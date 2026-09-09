from flask import Flask, jsonify, request, render_template
import numpy as np
import os
import glob
import torch

from intransitive.IntransitiveGame import Intransitive
from intransitive.pytorch.NNet import NNetWrapper
from MCTS import MCTS
from utils import dotdict

app = Flask(__name__)

game = Intransitive()
nnet = NNetWrapper(game)
nnet.load_checkpoint(folder='./temp_farmshare_v4/', filename='best.pth.tar')
mcts_args = dotdict({'numMCTSSims': 25, 'cpuct': 1})

state = {
    'board': game.getInitBoard(),
    'player': 1,
    'mcts': MCTS(game, nnet, mcts_args),
}

# --- bot vs bot state ---
bvb_state = {
    'active': False,
    'history': [],
    'view_index': 0,
    'players': None,
}

PIECE_LABELS = {
    ('blue', 'rock'): 'blue_rock', ('blue', 'paper'): 'blue_paper', ('blue', 'scissors'): 'blue_scissors',
    ('red', 'rock'): 'red_rock', ('red', 'paper'): 'red_paper', ('red', 'scissors'): 'red_scissors',
}

def board_to_json(board):
    n = board.n
    grid = [[None] * n for _ in range(n)]
    for (owner, ptype), label in PIECE_LABELS.items():
        bitboard = getattr(board, f"{owner}_{ptype}")
        for square in range(n * n):
            if (bitboard >> square) & 1:
                row, col = divmod(square, n)
                grid[row][col] = label
    return grid

def state_json():
    board, player = state['board'], state['player']
    result = game.getGameEnded(board, player)
    return jsonify({'grid': board_to_json(board), 'player': player,
                     'game_over': result != 0, 'result': result})

def list_checkpoints():
    pattern = os.path.join('temp*', '*.pth.tar')
    files = glob.glob(pattern)
    return sorted(files)

def load_player(ckpt_path):
    folder, filename = os.path.split(ckpt_path)
    n = NNetWrapper(game)
    n.load_checkpoint(folder=folder, filename=filename)
    m = MCTS(game, n, mcts_args)
    return m

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/state')
def get_state():
    return state_json()

@app.route('/move', methods=['POST'])
def make_move():
    data = request.get_json()
    board, player = state['board'], state['player']

    if game.getGameEnded(board, player) != 0:
        return jsonify({'error': 'Game already over'}), 400

    orig_square = board._square(*data['from'])
    dest_square = board._square(*data['to'])

    try:
        action = board._encode_action(orig_square, dest_square)
    except ValueError:
        return jsonify({'error': 'Invalid move direction'}), 400

    valids = game.getValidMoves(board, player)
    if not valids[action]:
        return jsonify({'error': 'Illegal move'}), 400

    board, player = game.getNextState(board, player, action)
    state['board'], state['player'] = board, player

    return state_json()

@app.route('/bot_move', methods=['POST'])
def bot_move():
    board, player = state['board'], state['player']

    if game.getGameEnded(board, player) != 0:
        return jsonify({'error': 'Game already over'}), 400

    canonical = game.getCanonicalForm(board, player)
    bot_action = int(np.argmax(state['mcts'].getActionProb(canonical, temp=0)))
    board, player = game.getNextState(board, player, bot_action)
    state['board'], state['player'] = board, player

    return state_json()

@app.route('/reset', methods=['POST'])
def reset():
    state['board'] = game.getInitBoard()
    state['player'] = 1
    state['mcts'] = MCTS(game, nnet, mcts_args)
    return jsonify({'status': 'ok'})

@app.route('/checkpoints')
def get_checkpoints():
    return jsonify(list_checkpoints())

@app.route('/start_bvb', methods=['POST'])
def start_bvb():
    try:
        data = request.get_json()
        ckpt1, ckpt2 = data['checkpoint1'], data['checkpoint2']

        p1 = load_player(ckpt1)
        p2 = load_player(ckpt2)
        players = {1: p1, -1: p2}

        board = game.getInitBoard()
        player = 1
        history = [(board, player)]

        max_moves = 300
        move_count = 0
        while game.getGameEnded(board, player) == 0 and move_count < max_moves:
            canonical = game.getCanonicalForm(board, player)
            action = int(np.argmax(players[player].getActionProb(canonical, temp=0)))
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            board, player = game.getNextState(board, player, action)
            history.append((board, player))
            move_count += 1
            print(f"Computed move {move_count}", flush=True)

        bvb_state['players'] = players
        bvb_state['history'] = history
        bvb_state['view_index'] = 1
        bvb_state['active'] = True

        return jsonify({'status': 'started', 'total_moves': len(history) - 1})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/bvb_forward', methods=['POST'])
def bvb_forward():
    if bvb_state['view_index'] < len(bvb_state['history']) - 1:
        bvb_state['view_index'] += 1
    return bvb_state_json()

@app.route('/bvb_backward', methods=['POST'])
def bvb_backward():
    if bvb_state['view_index'] > 0:
        bvb_state['view_index'] -= 1
    return bvb_state_json()

@app.route('/bvb_seek', methods=['POST'])
def bvb_seek():
    data = request.get_json()
    idx = max(0, min(int(data['index']), len(bvb_state['history']) - 1))
    bvb_state['view_index'] = idx
    return bvb_state_json()

def bvb_state_json():
    board, player = bvb_state['history'][bvb_state['view_index']]
    result = game.getGameEnded(board, player)
    return jsonify({
        'grid': board_to_json(board),
        'player': player,
        'game_over': result != 0,
        'result': result,
        'move_number': bvb_state['view_index'],
        'total_moves': len(bvb_state['history']) - 1,
        'at_frontier': bvb_state['view_index'] == len(bvb_state['history']) - 1,
    })

if __name__ == '__main__':
    app.run(port=5001, debug=False)