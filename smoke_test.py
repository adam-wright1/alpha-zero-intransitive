from intransitive.IntransitiveGame import Intransitive
from intransitive.IntransitivePlayers import RandomPlayer, HumanIntransitivePlayer
from intransitive.pytorch.NNet import NNetWrapper

game = Intransitive()
board = game.getInitBoard()
player = 1  # player starts

random_player = RandomPlayer(game)
human_player = HumanIntransitivePlayer(game)

# --- Neural net sanity check ---
nnet = NNetWrapper(game)

pi, v = nnet.predict(board)
print("Predicted pi shape:", pi.shape)
print("Predicted pi sum (should be ~1.0):", pi.sum())
print("Predicted v:", v)

# --- Symmetry round-trip sanity check ---
# Each of the 3 non-identity symmetries should be its own inverse:
# applying it twice should return to the original board/pi.

def boards_equal(b1, b2):
    return (b1.blue_rock == b2.blue_rock and
            b1.blue_paper == b2.blue_paper and
            b1.blue_scissors == b2.blue_scissors and
            b1.red_rock == b2.red_rock and
            b1.red_paper == b2.red_paper and
            b1.red_scissors == b2.red_scissors)

test_board = game.getInitBoard()
test_pi = [0.0] * game.getActionSize()
test_pi[0] = 0.5
test_pi[100] = 0.3
test_pi[647] = 0.2

sym = game.getSymmetries(test_board, test_pi)
labels = ["Identity", "Antidiagonal", "Rotation", "Maindiagonal"]

for label, (sym_board, sym_pi) in zip(labels, sym):
    if label == "Identity":
        continue

    # apply the same symmetry again to check it round-trips back to original
    sym_again = game.getSymmetries(sym_board, sym_pi)
    # find the matching transform in the second call's output by label position
    idx = labels.index(label)
    double_board, double_pi = sym_again[idx]

    board_ok = boards_equal(double_board, test_board)
    pi_ok = (double_pi == test_pi)

    print(f"{label}: board round-trip {'OK' if board_ok else 'FAILED'}, pi round-trip {'OK' if pi_ok else 'FAILED'}")
    if not board_ok:
        print(f"  original blue_rock={test_board.blue_rock}, after double-transform={double_board.blue_rock}")
    if not pi_ok:
        print(f"  original pi nonzero indices: {[i for i,v in enumerate(test_pi) if v != 0]}")
        print(f"  double-transformed pi nonzero indices: {[i for i,v in enumerate(double_pi) if v != 0]}")

#FEN TESTING

def board_to_fen(board):
    n = board.n
    symbol_map = {
        ('blue', 'rock'): 'R', ('blue', 'paper'): 'P', ('blue', 'scissors'): 'S',
        ('red', 'rock'): 'r', ('red', 'paper'): 'p', ('red', 'scissors'): 's',
    }

    rows = []
    for row in reversed(range(n)):
        rank_str = ""
        empty_count = 0
        for col in range(n):
            square = row * n + col
            bit = 1 << square
            piece_char = None
            for (owner, ptype), sym in symbol_map.items():
                bitboard = getattr(board, f"{owner}_{ptype}")
                if bitboard & bit:
                    piece_char = sym
                    break
            if piece_char:
                if empty_count > 0:
                    rank_str += str(empty_count)
                    empty_count = 0
                rank_str += piece_char
            else:
                empty_count += 1
        if empty_count > 0:
            rank_str += str(empty_count)
        rows.append(rank_str)

    # their rank 1 = Blue's home boundary; need to confirm which of your rows is "rank 1"
    return "/".join(rows)

test_board = game.getInitBoard()
print("Generated FEN (pieces only):", board_to_fen(test_board))

# plays game
while game.getGameEnded(board, player) == 0:
    canonical_board = game.getCanonicalForm(board, player)
    if player == 1:
        action = random_player.play(canonical_board)
        print("Random player (blue) plays action:", action)
    else:
        action = human_player.play(canonical_board, player)

    board, player = game.getNextState(board, player, action)
    Intransitive.display(board)

result = game.getGameEnded(board, player)
print("Game over. Result:", result)




