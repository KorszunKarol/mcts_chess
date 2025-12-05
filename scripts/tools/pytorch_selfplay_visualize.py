"""
PyTorch AI Self-Play with Board Visualization

Plays a game where the PyTorch model plays against itself and generates
PNG images of the game positions.
"""

import os
import sys
import chess
import chess.pgn
import numpy as np
import torch
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.transformer_model_pytorch import create_model
from src.encoder import Encoder
from src.move_mapping import move_to_index, index_to_move

# Try to import visualization libraries
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL/Pillow not available, will use SVG only")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import cairosvg
    CAIROSVG_AVAILABLE = True
except ImportError:
    CAIROSVG_AVAILABLE = False


class PyTorchChessPlayer:
    """Chess player using PyTorch model."""
    
    def __init__(self, model_path, name="PyTorch AI", device='cpu'):
        self.name = name
        self.device = device
        print(f"Loading PyTorch model from {model_path}...")
        
        checkpoint = torch.load(model_path, map_location=device)
        self.model = create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.to(device)
        
        self.encoder = Encoder()
        print(f"Model loaded successfully on {device}")
    
    def get_move(self, board: chess.Board, temperature=0.0):
        """
        Get the best move from the model.
        
        Args:
            board: Current chess board position
            temperature: Sampling temperature (0.0 = deterministic, >0 = stochastic)
        
        Returns:
            tuple: (best_move, value_probs, policy_logits, think_time)
        """
        start_time = time.time()
        
        # Encode board
        encoded_state = self.encoder.encode(board)
        
        # Convert to PyTorch format: (8, 8, 34) -> (1, 34, 8, 8) NCHW
        # Make a copy to avoid negative stride issues from encoder flip
        input_tensor = np.transpose(encoded_state.copy(), (2, 0, 1))  # (34, 8, 8)
        input_tensor = np.expand_dims(input_tensor, axis=0)  # (1, 34, 8, 8)
        input_tensor = torch.from_numpy(input_tensor).float().to(self.device)
        
        # Get predictions
        with torch.no_grad():
            value_probs, policy_logits = self.model(input_tensor)
        
        policy_logits = policy_logits[0].cpu().numpy()  # (4672,)
        value_probs = value_probs[0].cpu().numpy()  # (3,)
        
        # Mask illegal moves
        legal_move_indices = {move_to_index(move, board) for move in board.legal_moves}
        
        legal_mask = np.zeros_like(policy_logits)
        for index in legal_move_indices:
            if index is not None:
                legal_mask[index] = 1
        
        # Apply mask: set illegal moves to very negative values
        masked_policy = policy_logits + (legal_mask - 1) * 1e9
        
        # Select move
        if temperature > 0:
            # Stochastic sampling
            masked_policy = masked_policy / temperature
            masked_policy = masked_policy - np.max(masked_policy)  # Numerical stability
            probs = np.exp(masked_policy)
            probs = probs * legal_mask  # Zero out illegal
            probs = probs / probs.sum()  # Renormalize
            best_move_index = np.random.choice(len(probs), p=probs)
        else:
            # Deterministic: choose best legal move
            best_move_index = np.argmax(masked_policy)
        
        best_move = index_to_move(best_move_index)
        
        think_time = time.time() - start_time
        
        return best_move, value_probs, policy_logits, think_time


def draw_board_pil(board, move_number=None, last_move=None, save_path=None, size=800):
    """Draw chess board using PIL/Pillow."""
    if not PIL_AVAILABLE:
        return None
    
    square_size = size // 8
    img = Image.new('RGB', (size, size + 60), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw board squares
    light_color = (240, 217, 181)  # #f0d9b5
    dark_color = (181, 136, 99)     # #b58863
    
    for row in range(8):
        for col in range(8):
            is_light = (row + col) % 2 == 0
            color = light_color if is_light else dark_color
            
            x1 = col * square_size
            y1 = row * square_size
            x2 = x1 + square_size
            y2 = y1 + square_size
            
            draw.rectangle([x1, y1, x2, y2], fill=color, outline='black', width=2)
    
    # Highlight last move
    if last_move:
        from_sq = last_move.from_square
        to_sq = last_move.to_square
        for sq in [from_sq, to_sq]:
            row, col = divmod(sq, 8)
            x1 = col * square_size
            y1 = row * square_size
            x2 = x1 + square_size
            y2 = y1 + square_size
            draw.rectangle([x1, y1, x2, y2], outline='yellow', width=4)
    
    # Draw pieces (using Unicode symbols)
    piece_symbols = {
        'P': '♙', 'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔',
        'p': '♟', 'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚'
    }
    
    try:
        # Try to use a nice font
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 
                                 square_size // 2)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            symbol = piece_symbols.get(piece.symbol(), '?')
            color = 'white' if piece.color == chess.WHITE else 'black'
            
            x = col * square_size + square_size // 2
            y = row * square_size + square_size // 2
            
            # Draw piece
            if font:
                bbox = draw.textbbox((x, y), symbol, font=font, anchor='mm')
                draw.text((x, y), symbol, fill=color, font=font, anchor='mm')
            else:
                draw.text((x, y), symbol, fill=color, anchor='mm')
    
    # Add file labels (a-h)
    try:
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        label_font = ImageFont.load_default()
    
    for i in range(8):
        file_label = chr(97 + i)  # a-h
        x = i * square_size + square_size // 2
        draw.text((x, size + 10), file_label, fill='black', font=label_font, anchor='mm')
    
    # Add rank labels (1-8)
    for i in range(8):
        rank_label = str(8 - i)
        y = i * square_size + square_size // 2
        draw.text((10, y), rank_label, fill='black', font=label_font, anchor='mm')
    
    # Add move number
    if move_number is not None:
        move_text = f"Move {move_number}"
        draw.text((size // 2, size + 35), move_text, fill='black', font=label_font, anchor='mm')
    
    # Add result if game over
    if board.is_game_over():
        result = board.result()
        result_text = f"Game Over: {result}"
        draw.text((size // 2, size + 55), result_text, fill='red', font=label_font, anchor='mm')
    
    if save_path:
        img.save(save_path, 'PNG', dpi=(150, 150))
        return save_path
    return img


def draw_board_matplotlib(board, move_number=None, last_move=None, save_path=None):
    """Draw chess board using matplotlib."""
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Draw board squares
    for row in range(8):
        for col in range(8):
            is_light = (row + col) % 2 == 0
            color = '#f0d9b5' if is_light else '#b58863'
            rect = Rectangle((col, 7-row), 1, 1, facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
    
    # Highlight last move
    if last_move:
        from_sq = last_move.from_square
        to_sq = last_move.to_square
        from_row, from_col = divmod(from_sq, 8)
        to_row, to_col = divmod(to_sq, 8)
        
        for sq_row, sq_col in [(from_row, from_col), (to_row, to_col)]:
            rect = Rectangle((sq_col, 7-sq_row), 1, 1, 
                           facecolor='yellow', alpha=0.5, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
    
    # Draw pieces
    piece_symbols = {
        'P': '♙', 'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔',
        'p': '♟', 'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚'
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            symbol = piece_symbols.get(piece.symbol(), '?')
            color = 'white' if piece.color == chess.WHITE else 'black'
            ax.text(col + 0.5, 7 - row + 0.5, symbol, 
                   fontsize=40, ha='center', va='center',
                   color=color, weight='bold')
    
    # Add labels
    for i in range(8):
        # Files (a-h)
        ax.text(i + 0.5, -0.3, chr(97 + i), ha='center', va='top', fontsize=12)
        # Ranks (1-8)
        ax.text(-0.3, i + 0.5, str(8 - i), ha='right', va='center', fontsize=12)
    
    # Add move number
    if move_number is not None:
        ax.text(4, -0.8, f"Move {move_number}", ha='center', fontsize=14, weight='bold')
    
    # Add result if game over
    if board.is_game_over():
        result = board.result()
        ax.text(4, 8.5, f"Game Over: {result}", ha='center', fontsize=16, weight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        return fig




def self_play_game(model_path, output_dir='games', max_moves=200, device='cpu', save_every=1):
    """
    Play a game where the PyTorch model plays against itself.
    
    Args:
        model_path: Path to PyTorch model checkpoint
        output_dir: Directory to save game images
        max_moves: Maximum number of moves (to prevent infinite games)
        device: Device to run model on ('cpu' or 'cuda')
        save_every: Save image every N moves (1 = every move)
    """
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    game_dir = Path(output_dir) / f"selfplay_{timestamp}"
    game_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Game output directory: {game_dir}")
    
    # Create players (both use same model)
    white_player = PyTorchChessPlayer(model_path, "White (PyTorch)", device)
    black_player = PyTorchChessPlayer(model_path, "Black (PyTorch)", device)
    
    # Initialize game
    board = chess.Board()
    game_pgn = chess.pgn.Game()
    game_pgn.headers["Event"] = "PyTorch Self-Play"
    game_pgn.headers["Site"] = "Local"
    game_pgn.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game_pgn.headers["White"] = "PyTorch AI"
    game_pgn.headers["Black"] = "PyTorch AI"
    game_pgn.setup(board)
    node = game_pgn
    
    move_count = 0
    last_move = None
    
    print("\n" + "="*60)
    print("PYTORCH AI SELF-PLAY GAME")
    print("="*60)
    print(f"\nStarting position:")
    print(board)
    print()
    
    # Save initial position
    if PIL_AVAILABLE:
        draw_board_pil(board, move_number=0, last_move=None, 
                      save_path=str(game_dir / "move_000_initial.png"))
    elif MATPLOTLIB_AVAILABLE:
        draw_board_matplotlib(board, move_number=0, last_move=None, 
                            save_path=str(game_dir / "move_000_initial.png"))
    
    # Game loop
    while not board.is_game_over() and move_count < max_moves:
        move_count += 1
        player = white_player if board.turn == chess.WHITE else black_player
        player_name = "White" if board.turn == chess.WHITE else "Black"
        
        print(f"\n{'='*60}")
        print(f"Move {move_count} - {player_name} to move")
        print(f"{'='*60}")
        
        # Get move
        move, value_probs, policy_logits, think_time = player.get_move(board)
        
        # Print evaluation and timing
        print(f"Evaluation: Win={value_probs[0]:.1%}, Draw={value_probs[1]:.1%}, Loss={value_probs[2]:.1%}")
        print(f"Move: {board.san(move)} ({move.uci()})")
        print(f"Think time: {think_time*1000:.1f}ms")
        
        # Make move
        board.push(move)
        node = node.add_variation(move)
        
        # Print board
        print(f"\nPosition after move {move_count}:")
        print(board)
        
        # Save position
        if move_count % save_every == 0 or board.is_game_over():
            move_str = f"move_{move_count:03d}"
            
            # Save PNG
            if PIL_AVAILABLE:
                png_path = game_dir / f"{move_str}.png"
                draw_board_pil(board, move_number=move_count, 
                              last_move=move, save_path=str(png_path))
                print(f"  Saved PNG: {png_path}")
            elif MATPLOTLIB_AVAILABLE:
                png_path = game_dir / f"{move_str}.png"
                draw_board_matplotlib(board, move_number=move_count, 
                                    last_move=move, save_path=str(png_path))
                print(f"  Saved PNG: {png_path}")
        
        last_move = move
        
        # Check for game over
        if board.is_game_over():
            result = board.result()
            print(f"\n{'='*60}")
            print("GAME OVER")
            print(f"{'='*60}")
            print(f"Result: {result}")
            print(f"Reason: {board.result(claim_draw=True)}")
            print(f"\nFinal position:")
            print(board)
            break
    
    # Save final position
    if PIL_AVAILABLE:
        final_png = game_dir / "move_final.png"
        draw_board_pil(board, move_number=move_count, 
                      last_move=last_move, save_path=str(final_png))
    elif MATPLOTLIB_AVAILABLE:
        final_png = game_dir / "move_final.png"
        draw_board_matplotlib(board, move_number=move_count, 
                            last_move=last_move, save_path=str(final_png))
    
    # Save PGN
    game_pgn.headers["Result"] = board.result()
    pgn_path = game_dir / "game.pgn"
    with open(pgn_path, 'w') as f:
        exporter = chess.pgn.FileExporter(f)
        game_pgn.accept(exporter)
    
    print(f"\n{'='*60}")
    print("GAME SUMMARY")
    print(f"{'='*60}")
    print(f"Total moves: {move_count}")
    print(f"Result: {board.result()}")
    print(f"Game saved to: {game_dir}")
    print(f"PGN saved to: {pgn_path}")
    print(f"{'='*60}\n")
    
    return game_dir, game_pgn


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='PyTorch AI Self-Play with Visualization')
    parser.add_argument('--model', type=str, default='saved_models/best_model_pytorch.pt',
                       help='Path to PyTorch model checkpoint')
    parser.add_argument('--output', type=str, default='games',
                       help='Output directory for game files')
    parser.add_argument('--max-moves', type=int, default=200,
                       help='Maximum number of moves')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to run model on')
    parser.add_argument('--save-every', type=int, default=1,
                       help='Save image every N moves')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        sys.exit(1)
    
    self_play_game(
        model_path=args.model,
        output_dir=args.output,
        max_moves=args.max_moves,
        device=args.device,
        save_every=args.save_every
    )

