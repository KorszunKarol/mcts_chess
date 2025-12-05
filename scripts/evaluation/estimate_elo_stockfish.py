"""
ELO Estimation Script - PyTorch Engine vs Stockfish (Limited Strength)

Plays the PyTorch engine against Stockfish with limited strength to estimate ELO.
Uses binary search to find the Stockfish ELO that matches the PyTorch engine's strength.
"""

import os
import sys
import chess
import chess.engine
import chess.pgn
import numpy as np
import torch
import time
from pathlib import Path
from typing import Optional, Tuple

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.transformer_model_pytorch import create_model
from src.encoder import Encoder
from src.move_mapping import move_to_index, index_to_move


class PyTorchEngine:
    """UCI-like engine wrapper for PyTorch model."""
    
    def __init__(self, model_path, device='cpu'):
        self.device = device
        print(f"Loading PyTorch model from {model_path}...")
        
        checkpoint = torch.load(model_path, map_location=device)
        self.model = create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.to(device)
        
        self.encoder = Encoder()
        print(f"Model loaded successfully on {device}")
    
    def get_move(self, board: chess.Board) -> chess.Move:
        """Get the best move from the model.
        
        Note: No mirroring needed - model handles both perspectives directly
        (matches training where positions were encoded as-is).
        """
        # Encode board directly - NO mirroring (matches training)
        encoded_state = self.encoder.encode(board)
        
        # Convert to PyTorch format: (8, 8, 34) -> (1, 34, 8, 8) NCHW
        input_tensor = np.transpose(encoded_state.copy(), (2, 0, 1))  # (34, 8, 8)
        input_tensor = np.ascontiguousarray(input_tensor)  # Handle negative strides
        input_tensor = np.expand_dims(input_tensor, axis=0)  # (1, 34, 8, 8)
        input_tensor = torch.from_numpy(input_tensor).float().to(self.device)
        
        # Get predictions
        with torch.no_grad():
            value_probs, policy_logits = self.model(input_tensor)
        
        # Policy logits are already in correct format (no unmirroring needed)
        policy_logits = policy_logits[0].cpu().numpy()  # (4672,)
        
        # Mask illegal moves
        legal_move_indices = {move_to_index(move, board) for move in board.legal_moves}
        
        legal_mask = np.zeros_like(policy_logits)
        for index in legal_move_indices:
            if index is not None:
                legal_mask[index] = 1
        
        # Apply mask: set illegal moves to very negative values
        masked_policy = policy_logits + (legal_mask - 1) * 1e9
        
        # Select best move
        best_move_index = np.argmax(masked_policy)
        best_move = index_to_move(best_move_index)
        
        # Safety check
        if best_move not in board.legal_moves:
            print(f"WARNING: Illegal move selected, choosing random legal move")
            best_move = list(board.legal_moves)[0]
        
        return best_move


def play_game(pytorch_engine: PyTorchEngine, stockfish_engine, 
              stockfish_elo: int, pytorch_is_white: bool = True,
              time_limit: float = 2.0, save_pgn: Optional[Path] = None) -> Optional[str]:
    """
    Play a single game between PyTorch engine and Stockfish.
    
    Args:
        save_pgn: Optional path to save PGN file
    
    Returns:
        Result: "1-0" (PyTorch wins), "0-1" (Stockfish wins), "1/2-1/2" (draw), or None (error)
    """
    board = chess.Board()
    
    # Create PGN game object
    game = chess.pgn.Game()
    game.headers["Event"] = "ELO Estimation Match"
    game.headers["Site"] = "Local"
    game.headers["Date"] = time.strftime("%Y.%m.%d")
    game.headers["White"] = "PyTorch AI" if pytorch_is_white else "Stockfish"
    game.headers["Black"] = "Stockfish" if pytorch_is_white else "PyTorch AI"
    game.headers["StockfishElo"] = str(stockfish_elo)
    game.setup(board)
    node = game
    
    # Limit Stockfish strength
    stockfish_engine.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})
    
    move_count = 0
    max_moves = 200  # Prevent infinite games
    
    while not board.is_game_over() and move_count < max_moves:
        move_count += 1
        
        if (board.turn == chess.WHITE and pytorch_is_white) or \
           (board.turn == chess.BLACK and not pytorch_is_white):
            # PyTorch engine's turn
            try:
                move = pytorch_engine.get_move(board)
                board.push(move)
                node = node.add_variation(move)
            except Exception as e:
                print(f"Error in PyTorch engine: {e}")
                return None
        else:
            # Stockfish's turn
            try:
                result = stockfish_engine.play(board, chess.engine.Limit(time=time_limit))
                board.push(result.move)
                node = node.add_variation(result.move)
            except Exception as e:
                print(f"Error in Stockfish: {e}")
                return None
    
    # Determine result
    if board.is_game_over():
        result = board.result()
        # Convert result to PyTorch perspective
        if result == "1/2-1/2":
            final_result = "1/2-1/2"
        elif (result == "1-0" and pytorch_is_white) or (result == "0-1" and not pytorch_is_white):
            final_result = "1-0"  # PyTorch wins
        else:
            final_result = "0-1"  # Stockfish wins
    else:
        final_result = "1/2-1/2"  # Draw by move limit
    
    # Set game result
    game.headers["Result"] = final_result
    
    # Save PGN if requested
    if save_pgn:
        save_pgn.parent.mkdir(parents=True, exist_ok=True)
        with open(save_pgn, 'w') as f:
            exporter = chess.pgn.FileExporter(f)
            game.accept(exporter)
    
    return final_result


def play_match(pytorch_engine: PyTorchEngine, stockfish_path: str,
               stockfish_elo: int, num_games: int = 10,
               time_limit: float = 2.0, output_dir: Optional[Path] = None) -> Tuple[int, int, int]:
    """
    Play a match of multiple games.
    
    Args:
        output_dir: Directory to save PGN files (if None, no PGN saved)
    
    Returns:
        (wins, losses, draws) from PyTorch's perspective
    """
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as stockfish:
        wins = 0
        losses = 0
        draws = 0
        
        print(f"\nPlaying {num_games} games against Stockfish (ELO {stockfish_elo})...")
        if output_dir:
            print(f"PGN files will be saved to: {output_dir}")
        print("=" * 60)
        
        for game_num in range(1, num_games + 1):
            # Alternate colors
            pytorch_is_white = (game_num % 2 == 1)
            color_str = "White" if pytorch_is_white else "Black"
            
            # Prepare PGN path if saving
            pgn_path = None
            if output_dir:
                pgn_path = output_dir / f"game_{game_num:03d}_elo{stockfish_elo}_{color_str.lower()}.pgn"
            
            print(f"\nGame {game_num}/{num_games} - PyTorch plays as {color_str}...", end=" ", flush=True)
            
            result = play_game(pytorch_engine, stockfish, stockfish_elo, 
                             pytorch_is_white, time_limit, save_pgn=pgn_path)
            
            if result == "1-0":
                wins += 1
                print("WIN", end="")
            elif result == "0-1":
                losses += 1
                print("LOSS", end="")
            elif result == "1/2-1/2":
                draws += 1
                print("DRAW", end="")
            else:
                print("ERROR", end="")
            
            if pgn_path:
                print(f" (saved: {pgn_path.name})", end="")
            print()
        
        print("\n" + "=" * 60)
        print(f"Match Results: {wins} wins, {losses} losses, {draws} draws")
        print(f"Score: {wins + draws/2:.1f}/{num_games}")
        
        return wins, losses, draws


def estimate_elo(pytorch_engine: PyTorchEngine, stockfish_path: str,
                initial_elo: int = 1350, num_games: int = 10,
                time_limit: float = 2.0, tolerance: float = 0.1,
                save_pgn: bool = True) -> int:
    """
    Estimate ELO by binary search against Stockfish.
    
    Args:
        initial_elo: Starting ELO to test (minimum 1350 for Stockfish)
        num_games: Number of games per test
        time_limit: Time limit per move (seconds)
        tolerance: Win rate tolerance (0.1 = 10%)
    
    Returns:
        Estimated ELO
    """
    print("\n" + "=" * 60)
    print("ELO ESTIMATION - Binary Search Method")
    print("=" * 60)
    print(f"Initial ELO: {initial_elo}")
    print(f"Games per test: {num_games}")
    print(f"Time limit per move: {time_limit}s")
    print("=" * 60)
    
    # Create output directory for PGN files
    output_dir = None
    if save_pgn:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_dir = Path('games') / f'elo_estimation_{timestamp}'
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"PGN files will be saved to: {output_dir}")
    
    # Binary search bounds (Stockfish minimum is 1350)
    low_elo = 1350
    high_elo = 2000
    
    # Start with initial guess
    current_elo = initial_elo
    
    iteration = 0
    max_iterations = 10
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        print(f"Testing against Stockfish ELO: {current_elo}")
        
        # Create subdirectory for this iteration's games
        iter_output_dir = None
        if output_dir:
            iter_output_dir = output_dir / f"iteration_{iteration}_elo{current_elo}"
        
        wins, losses, draws = play_match(pytorch_engine, stockfish_path, 
                                        current_elo, num_games, time_limit,
                                        output_dir=iter_output_dir)
        
        win_rate = (wins + draws / 2) / num_games
        
        print(f"Win rate: {win_rate:.1%}")
        
        # Check if we're close to 50% (even match)
        if abs(win_rate - 0.5) < tolerance:
            print(f"\n{'='*60}")
            print(f"MATCH FOUND!")
            print(f"Estimated ELO: ~{current_elo}")
            print(f"Win rate: {win_rate:.1%} (target: 50%)")
            print(f"{'='*60}\n")
            return current_elo
        
        # Adjust ELO
        if win_rate > 0.5 + tolerance:
            # PyTorch is stronger, increase Stockfish ELO
            low_elo = current_elo
            if high_elo == 2000:
                # Expand search range
                current_elo = min(current_elo + 200, 2000)
            else:
                current_elo = (low_elo + high_elo) // 2
            print(f"PyTorch is stronger. Increasing Stockfish ELO to {current_elo}")
        else:
            # PyTorch is weaker, decrease Stockfish ELO
            high_elo = current_elo
            if low_elo == 1350:
                # Can't go below Stockfish minimum
                print(f"PyTorch is weaker than Stockfish at minimum ELO (1350)")
                print(f"Estimated ELO: <1350")
                return 1350
            else:
                current_elo = (low_elo + high_elo) // 2
            print(f"PyTorch is weaker. Decreasing Stockfish ELO to {current_elo}")
        
        # Safety check
        if high_elo - low_elo < 50:
            print(f"\n{'='*60}")
            print(f"Converged to ELO range: {low_elo} - {high_elo}")
            print(f"Estimated ELO: ~{current_elo}")
            print(f"{'='*60}\n")
            return current_elo
    
    print(f"\n{'='*60}")
    print(f"Reached max iterations. Final estimate: ~{current_elo} ELO")
    print(f"{'='*60}\n")
    return current_elo


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Estimate ELO by playing against Stockfish')
    parser.add_argument('--model', type=str, default='saved_models/best_model_pytorch.pt',
                       help='Path to PyTorch model checkpoint')
    parser.add_argument('--stockfish', type=str, default='/usr/games/stockfish',
                       help='Path to Stockfish executable')
    parser.add_argument('--initial-elo', type=int, default=1350,
                       help='Initial Stockfish ELO to test (minimum 1350)')
    parser.add_argument('--games', type=int, default=10,
                       help='Number of games per test')
    parser.add_argument('--time-limit', type=float, default=2.0,
                       help='Time limit per move (seconds)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to run model on')
    parser.add_argument('--no-pgn', action='store_true',
                       help='Do not save PGN files')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.stockfish):
        print(f"Error: Stockfish not found at {args.stockfish}")
        sys.exit(1)
    
    # Load PyTorch engine
    pytorch_engine = PyTorchEngine(args.model, device=args.device)
    
    # Estimate ELO
    estimated_elo = estimate_elo(
        pytorch_engine,
        args.stockfish,
        initial_elo=args.initial_elo,
        num_games=args.games,
        time_limit=args.time_limit,
        save_pgn=not args.no_pgn
    )
    
    print(f"\nFinal ELO Estimate: ~{estimated_elo}")

