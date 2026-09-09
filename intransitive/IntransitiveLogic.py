'''

Based on the board for the game of TicTacToe by Evgeny Tyurin.
Note: player_str stored as 'blue' or 'red'

'''
class Board():

    # list of all 8 directions on the board, as (x,y) offsets
    DIRECTIONS = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

    A_DIRECTIONS = []
    R_DIRECTIONS = []
    M_DIRECTIONS = []
    for dr, dc in DIRECTIONS:
        A_DIRECTIONS.append(DIRECTIONS.index((-dc, -dr)))
        R_DIRECTIONS.append(DIRECTIONS.index((-dr, -dc)))
        M_DIRECTIONS.append(DIRECTIONS.index((dc, dr)))

    # dict of capture pairs
    CAPTURES = {'rock': 'scissors', 'paper': 'rock', 'scissors': 'paper'}

    # number of possible move encodings
    ACTION_SIZE = 81 * 8  # 81 = 9x9 board squares, 8 = king-move directions

    def __init__(self, n):
        # Initializing board state with bitboard!
        self.n = n

        self.blue_rock = 0
        self.blue_paper = 0
        self.blue_scissors = 0
        self.red_rock = 0
        self.red_paper = 0
        self.red_scissors = 0

        self.blue_all = 0
        self.red_all = 0
        self.all = 0

        #TODO unused
        FULL_MASK = (1 << (self.n ** 2)) - 1

        self._setup_initial_position()

        self.moves_since_capture = 0

    # given row and col, outputs associated index
    def _square(self, row, col):
        return row * self.n + col

    # assumes that there IS a piece at square
    def _get_type(self, square):
        bit = 1 << square
        if self.blue_rock & bit or self.red_rock & bit:
            return 'rock'
        if self.blue_paper & bit or self.red_paper & bit:
            return 'paper'
        if self.blue_scissors & bit or self.red_scissors & bit:
            return 'scissors'
        return None

    # sets up bitboards to initial state
    def _setup_initial_position(self):
        # blue rocks (diagonal from (5,1) to (7,3))
        for i in range(3):
            self.blue_rock |= (1 << self._square(5 + i, 1 + i))

        # blue papers (diagonal from (4,1) to (7,4))
        for i in range(4):
            self.blue_paper |= (1 << self._square(4 + i, 1 + i))

        # blue scissors (diagonal from (4,2) to (6,4))
        for i in range(3):
            self.blue_scissors |= (1 << self._square(4 + i, 2 + i))

        # red rocks (diagonal from (1,5) to (3,7))
        for i in range(3):
            self.red_rock |= (1 << self._square(1 + i, 5 + i))

        # red papers (diagonal from (1,4) to (4,7))
        for i in range(4):
            self.red_paper |= (1 << self._square(1 + i, 4 + i))

        # red scissors (diagonal from (2,4) to (4,6))
        for i in range(3):
            self.red_scissors |= (1 << self._square(2 + i, 4 + i))

        self.blue_all = self.blue_rock | self.blue_paper | self.blue_scissors
        self.red_all = self.red_rock | self.red_paper | self.red_scissors
        self.all = self.blue_all | self.red_all

    # given a specific piece type (and player, implicit in bitboard) finds all moves (disregarding piece overlap weirdness)
    def _get_pseudo_moves_for_type(self, type_bitboard):
        moves = set()
        for square in range(self.n * self.n):
            if not (type_bitboard >> square) & 1:
                continue
            row, col = divmod(square, self.n)
            for rd, cd in self.DIRECTIONS:
                to_row, to_col = row + rd, col + cd
                if to_row not in range(self.n) or to_col not in range(self.n):
                    continue
                moves.add((square, self._square(to_row, to_col)))
        return moves

    # given a specific piece type, and a player, and the corresponding bitboard, finds all valid moves
    def _get_legal_moves_for_type(self, type_bitboard, piece_type, player_str):
        pseudo_moves = self._get_pseudo_moves_for_type(type_bitboard)
        legal_moves = set()

        for move in pseudo_moves:
            dest_square = move[1]
            
            # move is into own piece
            if player_str == 'blue':
                if (self.blue_all & (1 << dest_square)) != 0:
                    continue
            elif (self.red_all & (1 << dest_square)) != 0:
                continue

            # move is into non-capturable
            dest_type = self._get_type(dest_square)
            if dest_type is not None and self.CAPTURES[piece_type] != dest_type:
                continue
            
            legal_moves.add(move)

        return legal_moves

    # given a specific player, finds all legal moves (using current bitboard states)
    def _get_legal_moves(self, player_str):
        legal_moves = [0] * self.ACTION_SIZE

        legal_move_tuples = set()

        if player_str == 'blue':
            legal_rock_moves = self._get_legal_moves_for_type(self.blue_rock, 'rock', 'blue')
            legal_paper_moves = self._get_legal_moves_for_type(self.blue_paper, 'paper', 'blue')
            legal_scissors_moves = self._get_legal_moves_for_type(self.blue_scissors, 'scissors', 'blue')
        else:
            legal_rock_moves = self._get_legal_moves_for_type(self.red_rock, 'rock', 'red')
            legal_paper_moves = self._get_legal_moves_for_type(self.red_paper, 'paper', 'red')
            legal_scissors_moves = self._get_legal_moves_for_type(self.red_scissors, 'scissors', 'red')

        legal_move_tuples = legal_rock_moves | legal_paper_moves | legal_scissors_moves

        for orig_square, dest_square in legal_move_tuples:
            action = self._encode_action(orig_square, dest_square)
            legal_moves[action] = 1

        return legal_moves

    def _encode_action(self, orig_square, dest_square):
        orig_row, orig_col = divmod(orig_square, self.n)
        dest_row, dest_col = divmod(dest_square, self.n)
        dr, dc = dest_row - orig_row, dest_col - orig_col
        direction_index = self.DIRECTIONS.index((dr, dc))
        return orig_square * 8 + direction_index

    def _decode_action(self, action):
        orig_square = action // 8
        direction_index = action % 8
        dr, dc = self.DIRECTIONS[direction_index]
        orig_row, orig_col = divmod(orig_square, self.n)
        dest_row, dest_col = orig_row + dr, orig_col + dc
        dest_square = self._square(dest_row, dest_col)
        return orig_square, dest_square

    # returns 1 if blue wins, -1 if red wins, 0 if game still going, and small positive number if draw
    def is_win(self, next_player_str):
        # check corner wins
        if next_player_str == 'blue':
            if (self.red_all & (1 << 72)) != 0:
                return -1
        else:
            if (self.blue_all & (1 << 8)) != 0:
                return 1

        # check 100 ply (200 move) rule
        # this is negative to disincentivize defensive play
        if self.moves_since_capture >= 200:
            return -0.2

        # check valid moves/stalemate
        if not any(self._get_legal_moves(next_player_str)):
            if next_player_str == 'blue':
                return -1
            else:
                return 1

        return 0

    # takes a move and the player doing it, changes bitboards accordingly
    def execute_move(self, move, player_str):

        orig_square, dest_square = move

        piece_type = self._get_type(orig_square)

        # detects capture
        if self.all & (1 << dest_square) != 0:
            self.moves_since_capture = 0
        else:
            self.moves_since_capture += 1

        if player_str == 'blue':
            if piece_type == 'rock':
                self.blue_rock &= ~(1 << orig_square)
                self.blue_rock |= (1 << dest_square)
            elif piece_type == 'paper':
                self.blue_paper &= ~(1 << orig_square)
                self.blue_paper |= (1 << dest_square)
            elif piece_type == 'scissors':
                self.blue_scissors &= ~(1 << orig_square)
                self.blue_scissors |= (1 << dest_square)

            self.blue_all &= ~(1 << orig_square)
            self.blue_all |= (1 << dest_square)

            self.red_all &= ~(1 << dest_square)

            self.red_rock &= ~(1 << dest_square)
            self.red_paper &= ~(1 << dest_square)
            self.red_scissors &= ~(1 << dest_square)
        else:
            if piece_type == 'rock':
                self.red_rock &= ~(1 << orig_square)
                self.red_rock |= (1 << dest_square)
            elif piece_type == 'paper':
                self.red_paper &= ~(1 << orig_square)
                self.red_paper |= (1 << dest_square)
            elif piece_type == 'scissors':
                self.red_scissors &= ~(1 << orig_square)
                self.red_scissors |= (1 << dest_square)

            self.red_all &= ~(1 << orig_square)
            self.red_all |= (1 << dest_square)

            self.blue_all &= ~(1 << dest_square)

            self.blue_rock &= ~(1 << dest_square)
            self.blue_paper &= ~(1 << dest_square)
            self.blue_scissors &= ~(1 << dest_square)

        self.all = self.blue_all | self.red_all

    def copy(self):
        new_board = Board.__new__(Board)

        new_board.n = self.n

        new_board.blue_rock = self.blue_rock
        new_board.blue_paper = self.blue_paper
        new_board.blue_scissors = self.blue_scissors
        new_board.red_rock = self.red_rock
        new_board.red_paper = self.red_paper
        new_board.red_scissors = self.red_scissors

        new_board.blue_all = self.blue_all
        new_board.red_all = self.red_all
        new_board.all = self.all

        new_board.moves_since_capture = self.moves_since_capture

        return new_board

    def _reflect_bitboard(self, bitboard):
        reflected = 0
        for i in range(self.n * self.n):
            if (bitboard >> i) & 1:
                reflected |= (1 << ((self.n * self.n - 1) - i))
        return reflected

    def swap_colors(self):
        new_board = Board.__new__(Board)
        new_board.n = self.n

        new_board.blue_rock = self._reflect_bitboard(self.red_rock)
        new_board.blue_paper = self._reflect_bitboard(self.red_paper)
        new_board.blue_scissors = self._reflect_bitboard(self.red_scissors)
        new_board.red_rock = self._reflect_bitboard(self.blue_rock)
        new_board.red_paper = self._reflect_bitboard(self.blue_paper)
        new_board.red_scissors = self._reflect_bitboard(self.blue_scissors)

        new_board.blue_all = new_board.blue_rock | new_board.blue_paper | new_board.blue_scissors
        new_board.red_all = new_board.red_rock | new_board.red_paper | new_board.red_scissors
        new_board.all = new_board.blue_all | new_board.red_all

        new_board.moves_since_capture = self.moves_since_capture
        return new_board

    def transform_board(self, square_map_fn, swap_colors=False):
        new_board = Board.__new__(Board)
        new_board.n = self.n

        new_board.blue_rock = self._transform_bitboard(self.blue_rock, square_map_fn)
        new_board.blue_paper = self._transform_bitboard(self.blue_paper, square_map_fn)
        new_board.blue_scissors = self._transform_bitboard(self.blue_scissors, square_map_fn)
        new_board.red_rock = self._transform_bitboard(self.red_rock, square_map_fn)
        new_board.red_paper = self._transform_bitboard(self.red_paper, square_map_fn)
        new_board.red_scissors = self._transform_bitboard(self.red_scissors, square_map_fn)

        new_board.blue_all = new_board.blue_rock | new_board.blue_paper | new_board.blue_scissors
        new_board.red_all = new_board.red_rock | new_board.red_paper | new_board.red_scissors
        new_board.all = new_board.blue_all | new_board.red_all
        new_board.moves_since_capture = self.moves_since_capture

        if swap_colors:
            new_board.blue_rock, new_board.red_rock = new_board.red_rock, new_board.blue_rock
            new_board.blue_paper, new_board.red_paper = new_board.red_paper, new_board.blue_paper
            new_board.blue_scissors, new_board.red_scissors = new_board.red_scissors, new_board.blue_scissors
            new_board.blue_all, new_board.red_all = new_board.red_all, new_board.blue_all
            return new_board

        return new_board

    def _transform_bitboard(self, bitboard, square_map_fn):
        transformed = 0
        for square in range(self.n * self.n):
            if (bitboard >> square) & 1:
                new_square = square_map_fn(square)
                transformed |= (1 << new_square)
        return transformed

    def _antidiagonal_board_map(self, square):
        row, col = divmod(square, self.n)
        return self._square(self.n - 1 - col, self.n - 1 - row)

    def _rotation_board_map(self, square):
        row, col = divmod(square, self.n)
        return self._square(self.n - 1 - row, self.n - 1 - col)

    def _maindiagonal_board_map(self, square):
        row, col = divmod(square, self.n)
        return self._square(col, row)

    def transform_pi(self, pi, pi_map_fn):
        new = [0] * (self.n * self.n * 8)
        for old_index in range(self.n * self.n * 8):
            old_square, old_dir = divmod(old_index, 8)

            new_square, new_dir = pi_map_fn(old_square, old_dir)

            new_action = new_square * 8 + new_dir

            new[new_action] = pi[old_index]
        return new

    def _antidiagonal_pi_map(self, square, dir):
        a_square = self._antidiagonal_board_map(square)
        a_dir = self.A_DIRECTIONS[dir]

        return a_square, a_dir

    def _rotation_pi_map(self, square, dir):
        r_square = self._rotation_board_map(square)
        r_dir = self.R_DIRECTIONS[dir]

        return r_square, r_dir

    def _maindiagonal_pi_map(self, square, dir):
        m_square = self._maindiagonal_board_map(square)
        m_dir = self.M_DIRECTIONS[dir]

        return m_square, m_dir