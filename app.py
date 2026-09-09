from flask import Flask, jsonify, request, render_template
import numpy as np

from intransitive.IntransitiveGame import Intransitive
from intransitive.pytorch.NNet import NNetWrapper
from MCTS import MCTS
from utils import dotdict

app = Flask(__name__)

game = Intransitive()
nnet = NNetWrapper(game)
nnet.load_checkpoint(folder='./temp_farmshare_v2/', filename='best.pth.tar')
mcts_args = dotdict({'numMCTSSims': 25, 'cpuct': 1})

state = {
    'board': game.getInitBoard(),
    'player': 1,
    'mcts': MCTS(game, nnet, mcts_args),
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

if __name__ == '__main__':
    app.run(debug=True, port=5000)