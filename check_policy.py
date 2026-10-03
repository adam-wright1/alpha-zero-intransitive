import sys
sys.path.append('.')
import numpy as np
from intransitive.IntransitiveGame import Intransitive
from intransitive.pytorch.NNet import NNetWrapper

game = Intransitive()
nnet = NNetWrapper(game)
nnet.load_checkpoint(folder='./temp_farmshare_v13/', filename='best.pth.tar')


def decode_and_describe(board, action):
    orig_square = action // 8
    direction_index = action % 8
    orig_row, orig_col = divmod(orig_square, board.n)
    dr, dc = board.DIRECTIONS[direction_index]
    dest_row, dest_col = orig_row + dr, orig_col + dc
    dest_square = board._square(dest_row, dest_col) if 0 <= dest_row < board.n and 0 <= dest_col < board.n else None
    piece_type = board._get_type(orig_square)
    return f"orig=({orig_row},{orig_col}) sq{orig_square} [{piece_type}] -> dest=({dest_row},{dest_col}) sq{dest_square}"


def run_test(label, board, expected_action):
    pi, v = nnet.predict(board)
    print(f"\n--- {label} ---")
    print(f"Value estimate (v): {v}")
    print(f"Expected (capturing) move: {decode_and_describe(board, expected_action)}")
    print(f"Policy probability assigned to it: {pi[expected_action]:.6f}")
    top_action = int(np.argmax(pi))
    print(f"Top move: {decode_and_describe(board, top_action)}")
    print(f"Top move probability: {pi[top_action]:.6f}")
    print(f"Rank of capturing move: {np.argsort(-pi).tolist().index(expected_action)}")


# --- Test 1: sparse board (original test) ---
board1 = game.getInitBoard()
board1.blue_rock = board1.blue_paper = board1.blue_scissors = 0
board1.red_rock = board1.red_paper = board1.red_scissors = 0
board1.blue_rock = 1 << 40
board1.red_scissors = 1 << 41
board1.blue_all = board1.blue_rock | board1.blue_paper | board1.blue_scissors
board1.red_all = board1.red_rock | board1.red_paper | board1.red_scissors
board1.all = board1.blue_all | board1.red_all
board1.moves_since_capture = 0
run_test("Sparse board", board1, board1._encode_action(40, 41))

# --- Test 2: full starting position, but move one red scissors adjacent to a blue rock ---
board2 = game.getInitBoard()
blue_rock_squares = [sq for sq in range(81) if (board2.blue_rock >> sq) & 1]
orig_square = blue_rock_squares[0]
orig_row, orig_col = divmod(orig_square, board2.n)
dest_square = None
for dr, dc in board2.DIRECTIONS:
    r, c = orig_row + dr, orig_col + dc
    if 0 <= r < board2.n and 0 <= c < board2.n:
        sq = board2._square(r, c)
        if not (board2.all >> sq) & 1:
            dest_square = sq
            break

board2.red_rock &= ~(1 << dest_square)
board2.red_paper &= ~(1 << dest_square)
board2.red_scissors &= ~(1 << dest_square)
board2.red_scissors |= (1 << dest_square)
board2.red_all = board2.red_rock | board2.red_paper | board2.red_scissors
board2.all = board2.blue_all | board2.red_all

run_test("Full board, one exposed red scissors", board2, board2._encode_action(orig_square, dest_square))

# --- Test 3: piece adjacent to goal, clear path ---
board3 = game.getInitBoard()
board3.blue_rock = board3.blue_paper = board3.blue_scissors = 0
board3.red_rock = board3.red_paper = board3.red_scissors = 0

goal_square = 8
adjacent_square = board3._square(1, 7)  # one king-move away from square 8
board3.blue_rock = 1 << adjacent_square

board3.blue_all = board3.blue_rock | board3.blue_paper | board3.blue_scissors
board3.red_all = board3.red_rock | board3.red_paper | board3.red_scissors
board3.all = board3.blue_all | board3.red_all
board3.moves_since_capture = 0

expected_action = board3._encode_action(adjacent_square, goal_square)
run_test("Piece adjacent to empty goal", board3, expected_action)