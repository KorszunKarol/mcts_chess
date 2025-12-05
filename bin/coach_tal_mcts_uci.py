#!/usr/bin/env python3
"""
Coach Tal + MCTS UCI Engine

A UCI-compatible chess engine that combines:
1. MCTS search for strong move candidates
2. Coach Tal's cognitive asymmetry re-ranking

This version is significantly stronger than the raw policy version
because it uses actual tree search.

Usage:
    python coach_tal_mcts_uci.py
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
LOG_FILE = script_dir / 'coach_tal_mcts_uci.log'
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w',
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_WEIGHTS = str(script_dir / 'saved_models' / 'best_model_pytorch.pt')


class CoachTalMCTSEngine:
    """UCI protocol handler for Coach Tal + MCTS engine."""
    
    def __init__(self):
        self.board = chess.Board()
        
        # MCTS components (lazy loaded)
        self.mcts_controller = None
        self.mcts_started = False
        
        # Coach Tal components (lazy loaded)
        self.selector = None
        self.explainer = None
        
        # Configuration options
        self.weights_path = DEFAULT_WEIGHTS
        self.use_pytorch = True
        self.lambda_psych = 0.3
        self.gamma_confusion = 0.5
        self.delta_soundness = 0.15
        self.top_k = 5
        self.coach_tal_enabled = True
        
        # MCTS options
        self.num_simulations = 800  # Default simulations per move
        self.num_workers = 2  # MCTS worker processes
        
        logger.info("CoachTalMCTSEngine initialized")
    
    def _ensure_mcts_initialized(self):
        """Lazy-load MCTS controller."""
        if self.mcts_started:
            return
        
        logger.info("Initializing MCTS controller...")
        
        from src.mcts.controller import MCTSController
        
        self.mcts_controller = MCTSController(
            num_workers=self.num_workers,
            model_weights_path=self.weights_path,
            batch_size=32,
            max_wait_time_ms=10.0,
            buffer_count=64,
            use_mock_model=False,
        )
        self.mcts_controller.start()
        self.mcts_started = True
        logger.info("MCTS controller started")
    
    def _ensure_coach_tal_initialized(self):
        """Lazy-load Coach Tal selector."""
        if self.selector is not None:
            return
        
        logger.info("Initializing Coach Tal selector...")
        
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
        logger.info("Coach Tal selector initialized")
    
    def uci(self):
        """Handle 'uci' command."""
        print("id name Coach Tal MCTS v0.1")
        print("id author Karolito")
        print("")
        print(f"option name WeightsPath type string default {DEFAULT_WEIGHTS}")
        print("option name UsePyTorch type check default true")
        print("option name NumSimulations type spin default 800 min 100 max 10000")
        print("option name NumWorkers type spin default 2 min 1 max 8")
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
            self._reset_engines()
        elif name_lower == "usepytorch":
            self.use_pytorch = value.lower() == "true"
            self._reset_engines()
        elif name_lower == "numsimulations":
            self.num_simulations = int(value)
        elif name_lower == "numworkers":
            self.num_workers = int(value)
            self._reset_engines()
        elif name_lower == "lambdapsych":
            self.lambda_psych = int(value) / 100.0
            self.selector = None  # Force reload
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
    
    def _reset_engines(self):
        """Reset MCTS and Coach Tal when config changes."""
        if self.mcts_started and self.mcts_controller:
            self.mcts_controller.shutdown()
            self.mcts_started = False
            self.mcts_controller = None
        self.selector = None
    
    def isready(self):
        """Handle 'isready' command."""
        self._ensure_mcts_initialized()
        self._ensure_coach_tal_initialized()
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
        """Handle 'go' command with MCTS search."""
        self._ensure_mcts_initialized()
        self._ensure_coach_tal_initialized()
        
        # Parse time controls (simplified)
        num_sims = self.num_simulations
        
        # Check for movetime or other constraints
        for i, arg in enumerate(args):
            if arg == "movetime" and i + 1 < len(args):
                # Rough heuristic: more time = more simulations
                movetime_ms = int(args[i + 1])
                num_sims = max(100, min(5000, movetime_ms // 2))
            elif arg == "nodes" and i + 1 < len(args):
                num_sims = int(args[i + 1])
        
        logger.info(f"Searching position: {self.board.fen()} with {num_sims} simulations")
        
        try:
            # Step 1: Run MCTS search
            mcts_result = self.mcts_controller.run_search(
                fen=self.board.fen(),
                num_simulations=num_sims,
            )
            
            if mcts_result.error:
                logger.error(f"MCTS error: {mcts_result.error}")
            
            mcts_policy = mcts_result.policy
            mcts_value = mcts_result.q_value
            
            logger.info(f"MCTS returned {len(mcts_policy)} moves, Q={mcts_value:.3f}")
            
            if not mcts_policy:
                # Fallback to first legal move
                fallback = list(self.board.legal_moves)[0]
                print(f"bestmove {fallback.uci()}")
                sys.stdout.flush()
                return
            
            # Step 2: Re-rank with Coach Tal (if enabled)
            if self.coach_tal_enabled and self.selector:
                result = self.selector.select(self.board, mcts_policy)
                best_move = result.chosen_move
                
                # Log the analysis
                analysis = self.explainer.explain(result, self.board)
                logger.info(f"Coach Tal selected: {best_move.uci()} ({analysis.move_san})")
                logger.info(f"J-score: {analysis.j_score:.3f}, Value: {analysis.value:+.3f}")
                logger.info(f"Type: {analysis.move_type}, Reason: {analysis.primary_reason}")
                
                # Send info string with explanation
                info_str = f"{analysis.move_san} ({analysis.move_type}): {analysis.primary_reason}"
                print(f"info string {info_str}")
                print(f"info score cp {int(mcts_value * 100)} depth {num_sims // 100}")
            else:
                # Just use MCTS best move
                best_move = max(mcts_policy.items(), key=lambda x: x[1])[0]
                print(f"info score cp {int(mcts_value * 100)} depth {num_sims // 100}")
            
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
        if self.mcts_started and self.mcts_controller:
            self.mcts_controller.shutdown()
        sys.exit(0)
    
    def run(self):
        """Main UCI loop."""
        logger.info("Coach Tal MCTS UCI engine started")
        
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
                    pass  # We don't do async search yet
                else:
                    logger.warning(f"Unknown command: {cmd}")
                    
            except EOFError:
                break
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error processing command: {e}", exc_info=True)


def main():
    engine = CoachTalMCTSEngine()
    engine.run()


if __name__ == "__main__":
    main()





