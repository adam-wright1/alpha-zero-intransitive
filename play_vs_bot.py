from intransitive.IntransitiveGame import Intransitive
from intransitive.IntransitivePlayers import RandomPlayer, HumanIntransitivePlayer
from intransitive.pytorch.NNet import NNetWrapper
from MCTS import MCTS
from utils import dotdict
import numpy as np

game = Intransitive()
board = game.getInitBoard()
Intransitive.display(board)
player = 1  # player starts

# --- Load trained bot ---
nnet = NNetWrapper(game)
nnet.load_checkpoint(folder='./temp_farmshare/', filename='best.pth.tar')

mcts_args = dotdict({'numMCTSSims': 25, 'cpuct': 1})
mcts = MCTS(game, nnet, mcts_args)

def bot_play(canonical_board):
    return int(np.argmax(mcts.getActionProb(canonical_board, temp=0)))

human_player = HumanIntransitivePlayer(game)

# plays game
while game.getGameEnded(board, player) == 0:
    canonical_board = game.getCanonicalForm(board, player)
    if player == 1:
        action = human_player.play(canonical_board, player)
    else:
        action = bot_play(canonical_board)

    board, player = game.getNextState(board, player, action)
    Intransitive.display(board)

result = game.getGameEnded(board, player)
print("Game over. Result:", result)