from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .IntransitiveLogic import Board
import numpy as np

"""

Game class implementation for the game of Intransitive.
Based on the board for the game of TicTacToe by Evgeny Tyurin.

"""
class Intransitive(Game):
    def __init__(self, n=9):
        self.n = n

    def getInitBoard(self):
        # return initial board
        return Board(self.n)

    def getBoardSize(self):
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return Board.ACTION_SIZE

    def getNextState(self, board, player, action):
        action = int(action) # guard against numpy integer types
        new_board = board.copy()
        orig_square, dest_square = new_board._decode_action(action)

        if player == -1:
            n2 = self.n * self.n
            orig_square = n2 - 1 - orig_square
            dest_square = n2 - 1 - dest_square

        player_str = 'blue' if player == 1 else 'red'
        new_board.execute_move((orig_square, dest_square), player_str)
        return new_board, -player

    def getValidMoves(self, board, player):
        player_str = 'blue' if player == 1 else 'red'
        return board._get_legal_moves(player_str)

    def getGameEnded(self, board, player):
        if player == 1:
            player_str = 'blue'
        else:
            player_str = 'red'
        return board.is_win(player_str)

    def getCanonicalForm(self, board, player):
        if player == 1:
            return board
        else:
            return board.swap_colors()  # new Board method: swaps blue<->red bitboards

    def getSymmetries(self, board, pi):
        # identity
        I = (board, pi)

        # antidiagonal reflection
        A_board = board.transform_board(board._antidiagonal_board_map)
        A_pi = board.transform_pi(pi, board._antidiagonal_pi_map)
        A = (A_board, A_pi)

        # 180 degree rotation + color swap
        R_board = board.transform_board(board._rotation_board_map, swap_colors=True)
        R_pi = board.transform_pi(pi, board._rotation_pi_map)
        R = (R_board, R_pi)

        # main diagonal transpose + color swap
        M_board = board.transform_board(board._maindiagonal_board_map, swap_colors=True)
        M_pi = board.transform_pi(pi, board._maindiagonal_pi_map)
        M = (M_board, M_pi)

        return [I, A, R, M]

    # TODO should moves since capture be here?
    def stringRepresentation(self, board):
        return (board.blue_rock, board.blue_paper, board.blue_scissors,
                board.red_rock, board.red_paper, board.red_scissors,
                board.moves_since_capture)

    @staticmethod
    def display(board):
        n = board.n
        symbols = {
            ('blue', 'rock'): '🔵', ('blue', 'paper'): '🟦', ('blue', 'scissors'): '🔷',
            ('red', 'rock'): '🟠', ('red', 'paper'): '🟧', ('red', 'scissors'): '🔶',
        }
        goal_squares = {8, 72}

        for row in range(n):
            print(f"{n - row:2} ", end="")
            for col in range(n):
                square = row * n + col
                bit = 1 << square

                piece = None
                for (owner, ptype), sym in symbols.items():
                    bitboard = getattr(board, f"{owner}_{ptype}")
                    if bitboard & bit:
                        piece = sym
                        break

                if piece:
                    print(piece, end="")
                elif square in goal_squares:
                    print("🔲", end="")
                else:
                    print("⬜️", end="")
            print()

        print("   " + "".join(f"{chr(ord('A') + c):2}" for c in range(n)))
