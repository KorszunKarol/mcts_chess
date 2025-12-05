#!/usr/bin/env python3
"""
Coach Tal Interactive CLI

An interactive command-line interface for testing Coach Tal's
cognitive asymmetry move selection.

Usage:
    python scripts/coach_tal_cli.py [--weights PATH] [--pytorch]
    
Commands (in the CLI):
    <move>      Play a move (e.g., "e4", "Nf3", "e2e4")
    analyze     Show Coach Tal's analysis of current position
    undo        Take back the last move
    new         Start a new game
    fen <fen>   Set position from FEN string
    flip        Switch sides (you play as the other color)
    config      Show/modify Coach Tal configuration
    help        Show help
    quit        Exit
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import chess

from src.coach_tal.selector import CoachTalSelector, CoachTalConfig
from src.coach_tal.explainer import Explainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CoachTalCLI:
    """Interactive CLI for Coach Tal."""
    
    def __init__(self, config: CoachTalConfig):
        self.config = config
        self.selector = CoachTalSelector(config)
        self.explainer = Explainer()
        self.board = chess.Board()
        self.move_history = []
        self.user_color = chess.WHITE
    
    def run(self):
        """Main CLI loop."""
        print("\n" + "=" * 50)
        print("  Coach Tal - Cognitive Asymmetry Chess Engine")
        print("=" * 50)
        print("\nType 'help' for commands, 'quit' to exit.\n")
        
        self._print_board()
        
        while True:
            try:
                prompt = f"\n{'White' if self.board.turn == chess.WHITE else 'Black'} to move > "
                cmd = input(prompt).strip()
                
                if not cmd:
                    continue
                
                self._handle_command(cmd)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except EOFError:
                break
    
    def _handle_command(self, cmd: str):
        """Handle a CLI command."""
        parts = cmd.split()
        command = parts[0].lower()
        args = parts[1:]
        
        if command in ("quit", "exit", "q"):
            print("Goodbye!")
            sys.exit(0)
        
        elif command == "help":
            self._print_help()
        
        elif command == "new":
            self.board = chess.Board()
            self.move_history = []
            print("\nNew game started.")
            self._print_board()
        
        elif command == "undo":
            if self.move_history:
                self.board.pop()
                self.move_history.pop()
                print("\nMove undone.")
                self._print_board()
            else:
                print("No moves to undo.")
        
        elif command == "fen":
            if args:
                fen = " ".join(args)
                try:
                    self.board.set_fen(fen)
                    self.move_history = []
                    print(f"\nPosition set from FEN.")
                    self._print_board()
                except ValueError as e:
                    print(f"Invalid FEN: {e}")
            else:
                print(f"Current FEN: {self.board.fen()}")
        
        elif command == "flip":
            self.user_color = not self.user_color
            color_name = "White" if self.user_color == chess.WHITE else "Black"
            print(f"\nYou are now playing as {color_name}.")
        
        elif command == "analyze":
            self._analyze_position()
        
        elif command == "config":
            self._show_config(args)
        
        elif command == "board":
            self._print_board()
        
        else:
            # Try to parse as a move
            self._try_move(cmd)
    
    def _try_move(self, move_str: str):
        """Try to play a move."""
        try:
            # Try UCI notation first (e.g., "e2e4")
            try:
                move = chess.Move.from_uci(move_str)
                if move not in self.board.legal_moves:
                    raise ValueError("Illegal move")
            except ValueError:
                # Try SAN notation (e.g., "e4", "Nf3")
                move = self.board.parse_san(move_str)
            
            # Make the move
            san = self.board.san(move)
            self.board.push(move)
            self.move_history.append(move)
            
            print(f"\nPlayed: {san}")
            self._print_board()
            
            # Check for game over
            if self.board.is_game_over():
                self._print_game_over()
                return
            
            # If it's the engine's turn, get Coach Tal's recommendation
            if self.board.turn != self.user_color:
                self._get_engine_move()
        
        except ValueError as e:
            print(f"Invalid move: {move_str} ({e})")
    
    def _get_engine_move(self):
        """Get and play Coach Tal's recommended move."""
        print("\nCoach Tal is thinking...")
        
        try:
            result = self.selector.select_from_board(self.board)
            analysis = self.explainer.explain(result, self.board)
            
            # Show the recommendation
            print("\n" + self.explainer.format_summary(analysis, verbose=True))
            
            # Play the move
            san = self.board.san(result.chosen_move)
            self.board.push(result.chosen_move)
            self.move_history.append(result.chosen_move)
            
            print(f"\nCoach Tal plays: {san}")
            self._print_board()
            
            if self.board.is_game_over():
                self._print_game_over()
        
        except Exception as e:
            logger.error(f"Error getting engine move: {e}", exc_info=True)
            print(f"Error: {e}")
    
    def _analyze_position(self):
        """Analyze the current position with Coach Tal."""
        if self.board.is_game_over():
            self._print_game_over()
            return
        
        print("\nAnalyzing position...")
        
        try:
            result = self.selector.select_from_board(self.board, top_k=5)
            analyses = self.explainer.explain_all(result, self.board)
            
            print("\n" + self.explainer.format_comparison(analyses, top_n=5))
            
            # Show detailed analysis of top choice
            print("\nDetailed analysis of recommended move:")
            print(self.explainer.format_summary(analyses[0], verbose=True))
        
        except Exception as e:
            logger.error(f"Error analyzing position: {e}", exc_info=True)
            print(f"Error: {e}")
    
    def _print_board(self):
        """Print the current board state."""
        print()
        
        # Print from user's perspective
        if self.user_color == chess.WHITE:
            print(self.board)
        else:
            # Flip board for black
            print(self.board.transform(chess.flip_vertical).transform(chess.flip_horizontal))
        
        print(f"\nFEN: {self.board.fen()}")
        
        if self.move_history:
            moves_str = " ".join(
                f"{i//2 + 1}.{'' if i % 2 == 0 else '..'}{self.board.move_stack[i].uci()}"
                for i in range(len(self.move_history))
            )
            print(f"Moves: {moves_str}")
    
    def _print_game_over(self):
        """Print game over message."""
        result = self.board.result()
        
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            print(f"\n*** CHECKMATE! {winner} wins. ***")
        elif self.board.is_stalemate():
            print("\n*** STALEMATE! Draw. ***")
        elif self.board.is_insufficient_material():
            print("\n*** Draw by insufficient material. ***")
        elif self.board.can_claim_fifty_moves():
            print("\n*** Draw by fifty-move rule. ***")
        elif self.board.can_claim_threefold_repetition():
            print("\n*** Draw by threefold repetition. ***")
        else:
            print(f"\n*** Game over: {result} ***")
    
    def _show_config(self, args):
        """Show or modify configuration."""
        if not args:
            print("\nCoach Tal Configuration:")
            print(f"  lambda_psych: {self.config.lambda_psych}")
            print(f"  gamma_confusion: {self.config.gamma_confusion}")
            print(f"  delta_soundness: {self.config.delta_soundness}")
            print(f"  top_k_candidates: {self.config.top_k_candidates}")
            print(f"  opponent_temperature: {self.config.opponent_temperature}")
            print(f"  enabled: {self.config.enabled}")
            print("\nUse 'config <param> <value>' to change.")
        elif len(args) >= 2:
            param = args[0]
            value = args[1]
            
            try:
                if param == "lambda_psych":
                    self.config.lambda_psych = float(value)
                elif param == "gamma_confusion":
                    self.config.gamma_confusion = float(value)
                elif param == "delta_soundness":
                    self.config.delta_soundness = float(value)
                elif param == "top_k_candidates":
                    self.config.top_k_candidates = int(value)
                elif param == "opponent_temperature":
                    self.config.opponent_temperature = float(value)
                elif param == "enabled":
                    self.config.enabled = value.lower() in ("true", "1", "yes")
                else:
                    print(f"Unknown parameter: {param}")
                    return
                
                print(f"Set {param} = {getattr(self.config, param)}")
                
                # Reinitialize selector with new config
                self.selector = CoachTalSelector(self.config)
            except ValueError as e:
                print(f"Invalid value: {e}")
    
    def _print_help(self):
        """Print help message."""
        print("""
Coach Tal CLI Commands:
-----------------------
<move>          Play a move (e.g., "e4", "Nf3", "e2e4")
analyze         Show Coach Tal's analysis of current position
undo            Take back the last move
new             Start a new game
fen [<fen>]     Show current FEN or set position from FEN
flip            Switch sides (you play as the other color)
board           Show the current board
config [p v]    Show config or set parameter p to value v
help            Show this help
quit            Exit

Configuration Parameters:
  lambda_psych        Weight for psychological factors (default: 0.3)
  gamma_confusion     Weight for opponent confusion (default: 0.5)
  delta_soundness     Max allowed value drop (default: 0.15)
  top_k_candidates    Moves to consider (default: 5)
  opponent_temperature  Opponent model temperature (default: 1.2)
  enabled             Enable/disable Coach Tal (default: true)
""")


def main():
    parser = argparse.ArgumentParser(
        description="Coach Tal Interactive CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="/home/karolito/DL/chess_2.0/src/weights/best_model.keras",
        help="Path to model weights file",
    )
    parser.add_argument(
        "--pytorch",
        action="store_true",
        help="Use PyTorch backend instead of Keras",
    )
    parser.add_argument(
        "--lambda-psych",
        type=float,
        default=0.3,
        help="Weight for psychological factors",
    )
    parser.add_argument(
        "--gamma-confusion",
        type=float,
        default=0.5,
        help="Weight for opponent confusion",
    )
    parser.add_argument(
        "--delta-soundness",
        type=float,
        default=0.15,
        help="Maximum allowed value drop",
    )
    
    args = parser.parse_args()
    
    # Check weights file exists
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Error: Weights file not found: {weights_path}")
        print("Please provide a valid path with --weights")
        sys.exit(1)
    
    config = CoachTalConfig(
        weights_path=str(weights_path),
        use_pytorch=args.pytorch,
        lambda_psych=args.lambda_psych,
        gamma_confusion=args.gamma_confusion,
        delta_soundness=args.delta_soundness,
    )
    
    cli = CoachTalCLI(config)
    cli.run()


if __name__ == "__main__":
    main()





