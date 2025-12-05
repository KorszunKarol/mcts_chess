#!/usr/bin/env python3
"""
Coach Tal UCI Engine

A UCI-compatible chess engine wrapper for Coach Tal that can be used
with any standard chess GUI (Arena, Cute Chess, Lucas Chess, etc.).

Usage:
    python coach_tal_uci.py
    
Then configure your GUI to use this script as an engine.
"""

import sys
import os
import logging
from pathlib import Path

# Suppress TensorFlow/PyTorch warnings before imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent  # bin/ is one level down from root
sys.path.insert(0, str(project_root))

import chess

# Configure logging to file (UCI uses stdin/stdout)
LOG_FILE = project_root / 'coach_tal_uci.log'
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w',
)
logger = logging.getLogger(__name__)

# Default paths - adjust these for your setup
DEFAULT_WEIGHTS = str(script_dir / 'saved_models' / 'best_model_pytorch.pt')


class CoachTalUCI:
    """UCI protocol handler for Coach Tal engine."""
    
    def __init__(self):
        self.board = chess.Board()
        self.selector = None
        self.explainer = None
        
        # Configuration options (can be set via UCI)
        self.weights_path = DEFAULT_WEIGHTS
        self.use_pytorch = True
        self.lambda_psych = 0.3
        self.gamma_confusion = 0.5
        self.delta_soundness = 0.15
        self.top_k = 5
        self.coach_tal_enabled = True
        
        logger.info("CoachTalUCI initialized")
    
    def _ensure_initialized(self):
        """Lazy-load the selector on first use."""
        if self.selector is not None:
            return
        
        logger.info(f"Loading model from {self.weights_path}")
        
        from src.coach_tal import CoachTalSelector, CoachTalConfig, Explainer
        
        config = CoachTalConfig(
            weights_path=self.weights_path,
            use_pytorch=self.use_pytorch,
            lambda_psych=self.lambda_psych,
            gamma_confusion=self.gamma_confusion,
            delta_soundness=self.delta_soundness,
            top_k_candidates=self.top_k,
            enabled=self.coach_tal_enabled,
        )
        
        self.selector = CoachTalSelector(config)
        self.explainer = Explainer()
        logger.info("Model loaded successfully")
    
    def uci(self):
        """Handle 'uci' command."""
        print("id name Coach Tal v0.1")
        print("id author Karolito")
        print("")
        print(f"option name WeightsPath type string default {DEFAULT_WEIGHTS}")
        print("option name UsePyTorch type check default true")
        print("option name LambdaPsych type spin default 30 min 0 max 100")
        print("option name GammaConfusion type spin default 50 min 0 max 100")
        print("option name DeltaSoundness type spin default 15 min 0 max 50")
        print("option name TopK type spin default 5 min 1 max 20")
        print("option name CoachTalEnabled type check default true")
        print("uciok")
        sys.stdout.flush()
    
    def setoption(self, name: str, value: str):
        """Handle 'setoption' command."""
        name_lower = name.lower()
        
        if name_lower == "weightspath":
            self.weights_path = value
            self.selector = None  # Force reload
        elif name_lower == "usepytorch":
            self.use_pytorch = value.lower() == "true"
            self.selector = None
        elif name_lower == "lambdapsych":
            self.lambda_psych = int(value) / 100.0
            self.selector = None
        elif name_lower == "gammaconfusion":
            self.gamma_confusion = int(value) / 100.0
            self.selector = None
        elif name_lower == "deltasoundness":
            self.delta_soundness = int(value) / 100.0
            self.selector = None
        elif name_lower == "topk":
            self.top_k = int(value)
            self.selector = None
        elif name_lower == "coachtalenabled":
            self.coach_tal_enabled = value.lower() == "true"
            self.selector = None
        
        logger.info(f"Set option {name} = {value}")
    
    def isready(self):
        """Handle 'isready' command."""
        self._ensure_initialized()
        print("readyok")
        sys.stdout.flush()
    
    def ucinewgame(self):
        """Handle 'ucinewgame' command."""
        self.board = chess.Board()
        logger.info("New game started")
    
    def position(self, args: list):
        """Handle 'position' command."""
        idx = 0
        
        if args[idx] == "startpos":
            self.board = chess.Board()
            idx += 1
        elif args[idx] == "fen":
            idx += 1
            fen_parts = []
            while idx < len(args) and args[idx] != "moves":
                fen_parts.append(args[idx])
                idx += 1
            fen = " ".join(fen_parts)
            self.board = chess.Board(fen)
        
        # Apply moves if present
        if idx < len(args) and args[idx] == "moves":
            idx += 1
            while idx < len(args):
                move = chess.Move.from_uci(args[idx])
                self.board.push(move)
                idx += 1
        
        logger.info(f"Position set: {self.board.fen()}")
    
    def go(self, args: list):
        """Handle 'go' command."""
        self._ensure_initialized()
        
        logger.info(f"Thinking on position: {self.board.fen()}")
        
        try:
            result = self.selector.select_from_board(self.board, top_k=self.top_k)
            best_move = result.chosen_move
            
            # Log the analysis
            analysis = self.explainer.explain(result, self.board)
            logger.info(f"Best move: {best_move.uci()} ({analysis.move_san})")
            logger.info(f"J-score: {analysis.j_score:.3f}, Value: {analysis.value:+.3f}")
            logger.info(f"Reason: {analysis.primary_reason}")
            
            # Send info string with explanation (some GUIs show this)
            info_str = f"{analysis.move_san}: {analysis.primary_reason}"
            print(f"info string {info_str}")
            
            # Send best move
            print(f"bestmove {best_move.uci()}")
            sys.stdout.flush()
            
        except Exception as e:
            logger.error(f"Error during search: {e}", exc_info=True)
            # Fallback to first legal move
            fallback = list(self.board.legal_moves)[0]
            print(f"bestmove {fallback.uci()}")
            sys.stdout.flush()
    
    def quit(self):
        """Handle 'quit' command."""
        logger.info("Quit command received")
        sys.exit(0)
    
    def run(self):
        """Main UCI loop."""
        logger.info("Coach Tal UCI engine started")
        
        # Ensure stdout is line-buffered
        sys.stdout.reconfigure(line_buffering=True)
        
        while True:
            try:
                line = input().strip()
                if not line:
                    continue
                
                logger.info(f"Received: {line}")
                
                parts = line.split()
                cmd = parts[0].lower()
                args = parts[1:]
                
                if cmd == "uci":
                    self.uci()
                elif cmd == "isready":
                    self.isready()
                elif cmd == "ucinewgame":
                    self.ucinewgame()
                elif cmd == "position":
                    self.position(args)
                elif cmd == "go":
                    self.go(args)
                elif cmd == "setoption":
                    # Parse "setoption name X value Y"
                    if "name" in args:
                        name_idx = args.index("name") + 1
                        if "value" in args:
                            value_idx = args.index("value") + 1
                            name = " ".join(args[name_idx:args.index("value")])
                            value = " ".join(args[value_idx:])
                        else:
                            name = " ".join(args[name_idx:])
                            value = ""
                        self.setoption(name, value)
                elif cmd == "quit":
                    self.quit()
                elif cmd == "stop":
                    pass  # We don't do async search, so nothing to stop
                else:
                    logger.warning(f"Unknown command: {cmd}")
                    
            except EOFError:
                break
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error processing command: {e}", exc_info=True)


def main():
    engine = CoachTalUCI()
    engine.run()


if __name__ == "__main__":
    main()





