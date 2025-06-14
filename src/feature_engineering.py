import chess
import numpy as np

def get_piece_mobility(board):
    """
    Calculates the mobility of each piece on the board for both White and Black sides.
    Positive values indicate White pieces' mobility, and negative values indicate Black pieces' mobility.

    Args:
        board (chess.Board): The current state of the chess board.

    Returns:
        np.ndarray: An 8x8 matrix representing the mobility of each piece on the board.
    """
    mobility = np.zeros((8, 8), dtype=np.float32)

    # Define maximum moves per piece type for normalization
    max_moves_per_piece = {
        chess.PAWN: 4,
        chess.KNIGHT: 8,
        chess.BISHOP: 13,
        chess.ROOK: 14,
        chess.QUEEN: 27,
        chess.KING: 8
    }

    def calculate_mobility(board_copy, multiplier):
        """
        Calculates mobility for the current player in board_copy and updates the mobility matrix.

        Args:
            board_copy (chess.Board): A copy of the chess board.
            multiplier (int): 1 for White, -1 for Black.
        """
        legal_moves = list(board_copy.legal_moves)
        for move in legal_moves:
            from_square = move.from_square
            piece = board_copy.piece_at(from_square)
            if piece:
                # Correct row mapping: invert the row index
                original_row, col = divmod(from_square, 8)
                row = 7 - original_row  # Invert row index
                piece_type = piece.piece_type
                max_moves = max_moves_per_piece.get(piece_type, 27)
                mobility[row, col] += multiplier / max_moves

    # Calculate mobility for White
    calculate_mobility(board, 1)

    # Create a copy of the board and switch turn to calculate mobility for Black
    board_copy = board.copy()
    board_copy.turn = not board.turn
    calculate_mobility(board_copy, -1)

    mobility = mobility * -1

    return mobility

def analyze_pawn_structure(board):
    doubled = np.zeros((8, 8), dtype=np.float32)
    isolated = np.zeros((8, 8), dtype=np.float32)
    passed = np.zeros((8, 8), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type == chess.PAWN:
            file, rank = chess.square_file(square), chess.square_rank(square)
            color = piece.color

            # Doubled pawns
            if any(board.piece_at(chess.square(file, r)) == piece for r in range(rank + 1, 8)):
                doubled[rank, file] = 1

            # Isolated pawns
            if not any(board.piece_at(chess.square(f, r)) == piece
                       for f in [file - 1, file + 1] if 0 <= f < 8
                       for r in range(8)):
                isolated[rank, file] = 1

            # Passed pawns
            if not any(board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, not color)
                       for f in [file - 1, file, file + 1] if 0 <= f < 8
                       for r in range(rank + (1 if color else -1), 8 if color else -1, 1 if color else -1)):
                passed[rank, file] = 1

    return doubled, isolated, passed

def defended_and_vulnerable(board):
    defended = np.zeros((8, 8, 2), dtype=np.float32)  # [0] White, [1] Black
    vulnerable = np.zeros((8, 8, 2), dtype=np.float32)

    for color in [chess.WHITE, chess.BLACK]:
        opponent_color = not color
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                is_defended = any(board.attackers(color, attacker_square)
                                  for attacker_square in board.attackers(color, square))
                is_vulnerable = not is_defended
                row, col = divmod(square, 8)
                defended[row, col, int(color)] = 1 if is_defended else 0
                vulnerable[row, col, int(color)] = 1 if is_vulnerable else 0
    return defended, vulnerable

def piece_coordination(board):
    """
    Calculates a highly complex piece coordination score for both White and Black pieces on the board.

    Args:
        board (chess.Board): The current state of the chess board.

    Returns:
        np.ndarray: An 8x8x2 matrix representing the coordination of pieces.
                    [:,:,0] represents White's coordination, [:,:,1] represents Black's.
    """
    coordination = np.zeros((8, 8, 2), dtype=np.float32)

    for color in [chess.WHITE, chess.BLACK]:
        coordination[:,:,int(color)] = evaluate_color_coordination(board, color)

    return coordination

def evaluate_color_coordination(board, color):
    coordination = np.zeros((8, 8), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == color:
            row, col = chess.square_rank(square), chess.square_file(square)

            base_score = calculate_base_coordination(board, square, color)
            tactical_score = evaluate_tactical_coordination(board, square, color)
            strategic_score = evaluate_strategic_coordination(board, square, color)

            final_score = combine_coordination_scores(base_score, tactical_score, strategic_score)
            coordination[row, col] = final_score

    return coordination

def calculate_base_coordination(board, square, color):
    attackers = board.attackers(color, square)
    defenders = board.attackers(color, square)

    attacker_score = sum(piece_value(board.piece_at(attacker)) for attacker in attackers)
    defender_score = sum(piece_value(board.piece_at(defender)) for defender in defenders)

    return (attacker_score + defender_score) / 20  # Normalize

def evaluate_tactical_coordination(board, square, color):
    score = 0
    piece = board.piece_at(square)

    # Evaluate forks
    score += evaluate_forks(board, square, color) * 0.3

    # Evaluate pins and skewers
    score += evaluate_pins_and_skewers(board, square, color) * 0.25

    # Evaluate discovered attacks
    score += evaluate_discovered_attacks(board, square, color) * 0.2

    return score

def evaluate_strategic_coordination(board, square, color):
    score = 0
    piece = board.piece_at(square)

    # Evaluate piece mobility
    score += evaluate_mobility(board, square, color) * 0.15

    # Evaluate pawn structure support
    score += evaluate_pawn_structure_support(board, square, color) * 0.1

    # Evaluate king safety contribution
    score += evaluate_king_safety_contribution(board, square, color) * 0.2

    # Evaluate control of key squares
    score += evaluate_key_square_control(board, square, color) * 0.15

    return score

def combine_coordination_scores(base, tactical, strategic):
    # Combine scores with diminishing returns
    total = base + tactical * (1 - base/2) + strategic * (1 - (base + tactical)/3)
    return min(total, 2.0)  # Cap at 2.0

def piece_value(piece):
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
              chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 1}
    return values.get(piece.piece_type, 0) if piece else 0

def evaluate_forks(board, square, color):
    piece = board.piece_at(square)
    if piece.piece_type in [chess.KNIGHT, chess.PAWN]:
        attacks = list(board.attacks(square))
        valuable_targets = sum(1 for sq in attacks if board.piece_at(sq) and board.piece_at(sq).color != color and piece_value(board.piece_at(sq)) > piece_value(piece))
        return min(valuable_targets * 0.5, 1.0)
    return 0

def evaluate_pins_and_skewers(board, square, color):
    # Simplified evaluation, can be expanded
    return 0.5 if any(board.is_pinned(not color, sq) for sq in board.attacks(square)) else 0

def evaluate_discovered_attacks(board, square, color):
    # Simplified evaluation, can be expanded
    piece = board.piece_at(square)
    if piece.piece_type != chess.PAWN:
        behind_squares = list(board.attacks(square))
        return 0.5 if any(board.is_check() for move in behind_squares if board.is_legal(chess.Move(square, move))) else 0
    return 0

def evaluate_mobility(board, square, color):
    return len(list(board.legal_moves)) / 100  # Normalize

def evaluate_pawn_structure_support(board, square, color):
    piece = board.piece_at(square)
    if piece.piece_type != chess.PAWN:
        pawn_attacks = [sq for sq in board.attacks(square) if board.piece_at(sq) and board.piece_at(sq).piece_type == chess.PAWN and board.piece_at(sq).color == color]
        return len(pawn_attacks) * 0.25
    return 0

def evaluate_king_safety_contribution(board, square, color):
    king_square = board.king(color)
    if king_square:
        distance = chess.square_distance(square, king_square)
        return max(0, (8 - distance) / 8)
    return 0

def evaluate_key_square_control(board, square, color):
    central_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
    return 0.25 if square in central_squares else 0

def piece_square_tables(board):
    pst = np.zeros((8, 8), dtype=np.float32)
    piece_values = {
        chess.PAWN: [
            0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5,  5, 10, 25, 25, 10,  5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5, -5,-10,  0,  0,-10, -5,  5,
            5, 10, 10,-20,-20, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        ],
        chess.KNIGHT: [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50,
        ],
        chess.BISHOP: [
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10, 10, 10, 10, 10, 10, 10,-10,
            -10,  5,  0,  0,  0,  0,  5,-10,
            -20,-10,-10,-10,-10,-10,-10,-20,
        ],
        chess.ROOK: [
            0,  0,  0,  0,  0,  0,  0,  0,
            5, 10, 10, 10, 10, 10, 10,  5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            0,  0,  0,  5,  5,  0,  0,  0
        ],
        chess.QUEEN: [
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5,  5,  5,  5,  0,-10,
            -5,  0,  5,  5,  5,  5,  0, -5,
            0,  0,  5,  5,  5,  5,  0, -5,
            -10,  5,  5,  5,  5,  5,  0,-10,
            -10,  0,  5,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
        ],
        chess.KING: [
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -20,-30,-30,-40,-40,-30,-30,-20,
            -10,-20,-20,-20,-20,-20,-20,-10,
            20, 20,  0,  0,  0,  0, 20, 20,
            20, 30, 10,  0,  0, 10, 30, 20
        ]
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            piece_type = piece.piece_type
            color = piece.color
            if piece_type in piece_values:
                value = np.flipud(piece_values[piece_type])[square if color == chess.WHITE else 63 - square]
                pst[row, col] = value / 100

    return pst


def advanced_piece_square_tables(board):
    return piece_square_tables(board), piece_square_tables(board), piece_square_tables(board)

def is_outpost(board, square, color):
    opponent_pawns = chess.PAWN
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    opponent_color = not color
    pawn_attack_squares = [
        chess.square(file - 1, rank + 1) if color == chess.WHITE else chess.square(file - 1, rank - 1),
        chess.square(file + 1, rank + 1) if color == chess.WHITE else chess.square(file + 1, rank - 1)
    ]
    for pawn_square in pawn_attack_squares:
        if 0 <= pawn_square < 64 and board.piece_at(pawn_square) and board.piece_at(pawn_square).piece_type == opponent_pawns and board.piece_at(pawn_square).color == opponent_color:
            return False
    return True

def center_control(board):
    control = np.zeros((8, 8), dtype=np.float32)
    center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    for square in center_squares:
        row, col = divmod(square, 8)
        control[row, col] = len(board.attackers(chess.WHITE, square)) - len(board.attackers(chess.BLACK, square))
    return control / 4  # Normalize

def game_phase(board):
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9
    }

    total_material = sum(len(board.pieces(piece_type, color)) * value
                         for piece_type, value in piece_values.items()
                         for color in [chess.WHITE, chess.BLACK])

    # Assuming max material is 78 (16 pawns, 4 knights, 4 bishops, 4 rooks, 2 queens)
    max_material = 78

    # Normalize the phase between 0 (opening) and 1 (endgame)
    phase = 1 - (total_material / max_material)
    phase = np.full((8, 8), phase, dtype=np.float32)

    return phase



def king_safety(board):
    """
    Calculates a comprehensive king safety score for both White and Black.

    Args:
        board (chess.Board): The current state of the chess board.

    Returns:
        np.ndarray: An 8x8x1 matrix representing the king safety.
                    Positive values indicate better safety for White, negative for Black.
    """
    safety = np.zeros((8, 8, 1), dtype=np.float32)

    white_safety = evaluate_king_safety(board, chess.WHITE)
    black_safety = evaluate_king_safety(board, chess.BLACK)

    # Combine white and black safety scores
    combined_safety = white_safety - black_safety
    safety[:, :, 0] = np.clip(combined_safety, -1, 1)

    return safety

def evaluate_king_safety(board, color):
    king_square = board.king(color)
    if king_square is None:
        return np.zeros((8, 8), dtype=np.float32)

    safety_score = np.zeros((8, 8), dtype=np.float32)

    # Evaluate different aspects of king safety
    safety_score += evaluate_pawn_shield(board, king_square, color)
    safety_score += evaluate_king_zone_control(board, king_square, color)
    safety_score += evaluate_attacking_pieces(board, king_square, color)
    safety_score += evaluate_open_files(board, king_square, color)
    safety_score += evaluate_castling_status(board, king_square, color)

    return safety_score

def evaluate_pawn_shield(board, king_square, color):
    score = np.zeros((8, 8), dtype=np.float32)
    king_file, king_rank = chess.square_file(king_square), chess.square_rank(king_square)

    pawn_shield_squares = [
        (king_file - 1, king_rank + (1 if color == chess.WHITE else -1)),
        (king_file, king_rank + (1 if color == chess.WHITE else -1)),
        (king_file + 1, king_rank + (1 if color == chess.WHITE else -1)),
        (king_file - 1, king_rank + (2 if color == chess.WHITE else -2)),
        (king_file, king_rank + (2 if color == chess.WHITE else -2)),
        (king_file + 1, king_rank + (2 if color == chess.WHITE else -2))
    ]

    for file, rank in pawn_shield_squares:
        if 0 <= file < 8 and 0 <= rank < 8:
            square = chess.square(file, rank)
            if board.piece_at(square) == chess.Piece(chess.PAWN, color):
                score[rank, file] += 0.2
            elif board.piece_at(square) == chess.Piece(chess.PAWN, not color):
                score[rank, file] -= 0.1

    return score

def evaluate_king_zone_control(board, king_square, color):
    score = np.zeros((8, 8), dtype=np.float32)
    king_file, king_rank = chess.square_file(king_square), chess.square_rank(king_square)

    king_zone = [
        (f, r) for f in range(max(0, king_file-2), min(8, king_file+3))
        for r in range(max(0, king_rank-2), min(8, king_rank+3))
    ]

    for file, rank in king_zone:
        square = chess.square(file, rank)
        if board.is_attacked_by(color, square):
            score[rank, file] += 0.05
        if board.is_attacked_by(not color, square):
            score[rank, file] -= 0.05

    return score

def evaluate_attacking_pieces(board, king_square, color):
    score = np.zeros((8, 8), dtype=np.float32)
    king_file, king_rank = chess.square_file(king_square), chess.square_rank(king_square)

    attack_weight = {chess.PAWN: 0.1, chess.KNIGHT: 0.2, chess.BISHOP: 0.2,
                     chess.ROOK: 0.3, chess.QUEEN: 0.5}

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color != color:
            if board.is_attacked_by(not color, king_square):
                file, rank = chess.square_file(square), chess.square_rank(square)
                distance = max(abs(file - king_file), abs(rank - king_rank))
                score[rank, file] -= attack_weight.get(piece.piece_type, 0.1) * (1 / (distance + 1))

    return score

def evaluate_open_files(board, king_square, color):
    score = np.zeros((8, 8), dtype=np.float32)
    king_file, king_rank = chess.square_file(king_square), chess.square_rank(king_square)

    for file in range(max(0, king_file-1), min(8, king_file+2)):
        if not any(board.piece_at(chess.square(file, r)) for r in range(8)):
            score[king_rank, file] -= 0.3

    return score

def evaluate_castling_status(board, king_square, color):
    score = np.zeros((8, 8), dtype=np.float32)
    king_file, king_rank = chess.square_file(king_square), chess.square_rank(king_square)

    if board.has_castling_rights(color):
        score[king_rank, king_file] += 0.5
    elif (color == chess.WHITE and king_square in [chess.G1, chess.C1]) or \
         (color == chess.BLACK and king_square in [chess.G8, chess.C8]):
        score[king_rank, king_file] += 0.7

    return score