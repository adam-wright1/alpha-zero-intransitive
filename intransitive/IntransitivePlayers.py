import numpy as np
from intransitive.IntransitiveGame import Intransitive

"""
Random and Human-interacting players for the game of Intransitive.

Based on the TicTacToePlayers by Evgeny Tyurin.
"""
class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanIntransitivePlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board, real_player):
        valid = self.game.getValidMoves(board, 1)
        n = board.n

        while True:
            input_str = input("Enter move as <from> <to> (e.g. A4 B3)\n")
            parts = input_str.split(' ')
            if len(parts) != 2:
                print("Invalid move. Enter two squares separated by a space, e.g. A4 B3")
                continue

            try:
                from_row, from_col = self._parse_square(parts[0], n)
                to_row, to_col = self._parse_square(parts[1], n)
            except (ValueError, IndexError):
                print("Invalid square format. Use letter+number, e.g. A4")
                continue

            # convert true-orientation coords into canonical coords if needed
            if real_player == -1:
                from_row, from_col = n - 1 - from_row, n - 1 - from_col
                to_row, to_col = n - 1 - to_row, n - 1 - to_col

            orig_square = board._square(from_row, from_col)
            dest_square = board._square(to_row, to_col)

            try:
                action = board._encode_action(orig_square, dest_square)
            except ValueError:
                print("Invalid move (not a valid king-move direction).")
                continue

            if valid[action]:
                return action
            else:
                print("Invalid move, try again.")

    @staticmethod
    def _parse_square(s, n):
        col_letter = s[0].upper()
        row_number = int(s[1:])
        col = ord(col_letter) - ord('A')
        row = n - row_number  # undo display's row-reversal, independent of canonical form
        if not (0 <= row < n and 0 <= col < n):
            raise ValueError(f"Square '{s}' is out of range for a {n}x{n} board.")
        return row, col
