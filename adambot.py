#!/usr/bin/env python3
import sys
import numpy as np
import time

from intransitive.IntransitiveLogic import Board
from intransitive.IntransitiveGame import Intransitive
from intransitive.pytorch.NNet import NNetWrapper
from MCTS import MCTS
from utils import dotdict

def move_str_to_squares(move_str, n):
    orig_str = move_str[0:2]
    dest_str = move_str[3:]

    orig_col = ord(orig_str[0]) - ord('a')
    orig_row = n - int(orig_str[1])
    orig_square = orig_row * n + orig_col

    dest_col = ord(dest_str[0]) - ord('a')
    dest_row = n - int(dest_str[1])
    dest_square = dest_row * n + dest_col

    return orig_square, dest_square

def squares_to_move_str(orig_square, dest_square, n):
    orig_row, orig_col = divmod(orig_square, n)
    dest_row, dest_col = divmod(dest_square, n)

    orig_file = chr(ord('a') + orig_col)
    orig_rank = n - orig_row
    dest_file = chr(ord('a') + dest_col)
    dest_rank = n - dest_row

    return f"{orig_file}{orig_rank}-{dest_file}{dest_rank}"

def fen_to_board(fen_fields, n):
    board = Board(n)
    board.blue_rock = board.blue_paper = board.blue_scissors = 0
    board.red_rock = board.red_paper = board.red_scissors = 0

    piece_map = {
        'R': 'blue_rock', 'P': 'blue_paper', 'S': 'blue_scissors',
        'r': 'red_rock', 'p': 'red_paper', 's': 'red_scissors',
    }

    ranks = fen_fields[0].split('/')  # rank 1 first, matching board_to_fen's convention

    for i, rank_str in enumerate(ranks):
        display_row = i  # rank 1 = display_row 0, rank 9 = display_row 8
        row = n - 1 - display_row  # convert back to internal row
        col = 0
        for ch in rank_str:
            if ch.isdigit():
                col += int(ch)
            else:
                attr_name = piece_map[ch]
                square = row * n + col
                setattr(board, attr_name, getattr(board, attr_name) | (1 << square))
                col += 1

    board.blue_all = board.blue_rock | board.blue_paper | board.blue_scissors
    board.red_all = board.red_rock | board.red_paper | board.red_scissors
    board.all = board.blue_all | board.red_all
    board.moves_since_capture = 0  # will be corrected by replaying moves afterward

    return board


# game / model setup
game = Intransitive()
nnet = NNetWrapper(game)
# THIS IS IMPORTANT
nnet.load_checkpoint(folder='./temp/', filename='temp.pth.tar')

# this 25 is very important
mcts_args = dotdict({'numMCTSSims': 25, 'cpuct': 1})
mcts = MCTS(game, nnet, mcts_args)

# default/running state so Python doesn't freak out
board = Board(9)  # will hold the current Board object, rebuilt on each 'position' command
moves_list = []
n = 9

#timing
warmup_board = game.getCanonicalForm(game.getInitBoard(), 1)
warmup_sims = 10
mcts_args['numMCTSSims'] = warmup_sims

start = time.time()
mcts.getActionProb(warmup_board, temp=0)
elapsed_ms = (time.time() - start) * 1000
ms_per_sim = elapsed_ms / warmup_sims

# THE BULK OF IT
for line in sys.stdin:
    word = line.split()
    if not word:
        continue
    if word[0] == "rpsi":
        print("id name AdamBot", flush=True)
        print("id author Cray", flush=True)
        print("protocol 1", flush=True)
        print("rules 2", flush=True)
        print("mode V6 Intransitive", flush=True)
        print("rpsiok", flush=True)
    elif word[0] == "isready":
        print("readyok", flush=True)
    elif word[0] == "position":
        if "moves" in word:
            moves_index = word.index("moves")
            fen_fields = word[2:moves_index]
            moves_list = word[moves_index + 1:]
        else:
            fen_fields = word[2:]
            moves_list = []

        n = len(fen_fields[0].split('/'))

        board = fen_to_board(fen_fields, n)
        for i, move_str in enumerate(moves_list):
            move = move_str_to_squares(move_str, n)
            player_str = 'blue' if i % 2 == 0 else 'red'
            board.execute_move(move, player_str)
    elif word[0] == "legalmoves":
        legal_moves = word[1:]
    elif word[0] == "go":
        current_player_str = 'blue' if len(moves_list) % 2 == 0 else 'red'
        current_player = 1 if current_player_str == 'blue' else -1

        # time management wizardry
        time_fields = {}
        for i in range(len(word) // 2):
            field_name = word[(i * 2) + 1]
            time_ms = word[(i * 2) + 2]

            time_fields[field_name] = time_ms

        if current_player_str == 'blue':
            my_time_ms = int(time_fields['btime'])
            my_inc_ms = int(time_fields['binc'])
        else:
            my_time_ms = int(time_fields['rtime'])
            my_inc_ms = int(time_fields['rinc'])

        # timing params can easily change
        time_budget_ms = my_time_ms / 20 + my_inc_ms * 0.8
        estimated_sims = max(1, int(time_budget_ms / ms_per_sim))
        mcts.args['numMCTSSims'] = estimated_sims

        canonical_board = game.getCanonicalForm(board, current_player)
        pi = mcts.getActionProb(canonical_board, temp=0)
        action = int(np.argmax(pi))

        orig_square, dest_square = board._decode_action(action)

        if current_player == -1:
            n2 = n * n
            orig_square = n2 - 1 - orig_square
            dest_square = n2 - 1 - dest_square

        move_dash = squares_to_move_str(orig_square, dest_square, n)
        move_x = move_dash.replace('-', 'x')

        if move_dash in legal_moves:
            matched_move_str = move_dash
        elif move_x in legal_moves:
            matched_move_str = move_x
        else:
            matched_move_str = legal_moves[0]  # fallback

        print(f"bestmove {matched_move_str}", flush=True)

    elif word[0] == "quit":
        break