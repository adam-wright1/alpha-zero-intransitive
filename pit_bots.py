# ai slop
import sys
sys.path.append('.')
import argparse
import numpy as np

from intransitive.IntransitiveGame import Intransitive
from intransitive.pytorch.NNet import NNetWrapper
from MCTS import MCTS
from utils import dotdict

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint1', type=str, help='e.g. temp_farmshare_v13/best.pth.tar', default='temp_farmshare_v13/best.pth.tar')
parser.add_argument('--checkpoint2', type=str, help='e.g. temp_farmshare_v22/temp.pth.tar', default='temp_farmshare_v22/best.pth.tar')
parser.add_argument('--numGames', type=int, default=20)
parser.add_argument('--numMCTSSims', type=int, default=70)
parser.add_argument('--cpuct', type=float, default=1.0)
parser.add_argument('--maxMoves', type=int, default=1000)
args = parser.parse_args()

game = Intransitive()
mcts_args = dotdict({'numMCTSSims': args.numMCTSSims, 'cpuct': args.cpuct,
                      'dirichlet_alpha': None, 'dirichlet_epsilon': 0.25})


def load_player(ckpt_path):
    import os
    folder, filename = os.path.split(ckpt_path)
    n = NNetWrapper(game)
    n.load_checkpoint(folder=folder, filename=filename)
    return MCTS(game, n, mcts_args)


print(f"Loading checkpoint1: {args.checkpoint1}")
mcts1 = load_player(args.checkpoint1)
print(f"Loading checkpoint2: {args.checkpoint2}")
mcts2 = load_player(args.checkpoint2)


def play_one_game(first_player_mcts, second_player_mcts):
    """first_player_mcts plays as blue (1), second_player_mcts plays as red (-1).
    Returns: 1 if blue won, -1 if red won, else the draw/stalemate value."""
    players = {1: first_player_mcts, -1: second_player_mcts}
    board = game.getInitBoard()
    player = 1
    move_count = 0
    captures_total = 0

    while game.getGameEnded(board, player) == 0 and move_count < args.maxMoves:
        canonical = game.getCanonicalForm(board, player)
        action = int(np.argmax(players[player].getActionProb(canonical, temp=0)))
        board, player, captured = game.getNextState(board, player, action)
        if captured:
            captures_total += 1
        move_count += 1

    result = game.getGameEnded(board, -player)  # result relative to the player who just moved is inverted; get from perspective of mover
    # getGameEnded expects (board, player) where player is "next to move" perspective;
    # since loop exits because either game ended for 'player' or move cap hit, re-derive cleanly:
    final_result_for_blue = game.getGameEnded(board, 1) if game.getGameEnded(board, 1) != 0 else game.getGameEnded(board, -1) * -1
    return final_result_for_blue, move_count, captures_total


p1_wins = 0  # checkpoint1 wins (regardless of color)
p2_wins = 0
draws = 0
game_logs = []

for i in range(args.numGames):
    # alternate who plays blue to remove color-order bias
    if i % 2 == 0:
        result, moves, captures = play_one_game(mcts1, mcts2)
        # result is relative to blue; checkpoint1 was blue
        if result == 1:
            p1_wins += 1
            winner = "checkpoint1"
        elif result == -1:
            p2_wins += 1
            winner = "checkpoint2"
        else:
            draws += 1
            winner = "draw"
    else:
        result, moves, captures = play_one_game(mcts2, mcts1)
        # result is relative to blue; checkpoint2 was blue
        if result == 1:
            p2_wins += 1
            winner = "checkpoint2"
        elif result == -1:
            p1_wins += 1
            winner = "checkpoint1"
        else:
            draws += 1
            winner = "draw"

    game_logs.append((i + 1, winner, moves, captures))
    print(f"Game {i + 1}/{args.numGames}: winner={winner}, moves={moves}, captures={captures}  "
          f"[running: ckpt1={p1_wins} ckpt2={p2_wins} draws={draws}]")

print("\n" + "=" * 50)
print("FINAL REPORT")
print("=" * 50)
print(f"checkpoint1 ({args.checkpoint1}): {p1_wins} wins")
print(f"checkpoint2 ({args.checkpoint2}): {p2_wins} wins")
print(f"Draws: {draws}")
print(f"Total games: {args.numGames}")
if p1_wins + p2_wins > 0:
    print(f"checkpoint1 win rate (excluding draws): {p1_wins / (p1_wins + p2_wins) * 100:.1f}%")
avg_moves = sum(m for _, _, m, _ in game_logs) / len(game_logs)
avg_captures = sum(c for _, _, _, c in game_logs) / len(game_logs)
print(f"Average moves per game: {avg_moves:.1f}")
print(f"Average captures per game: {avg_captures:.1f}")