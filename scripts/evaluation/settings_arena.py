#!/usr/bin/env python3
"""
Settings Arena - Systematic comparison of engine configurations.

This script runs round-robin matches between different engine settings
to find optimal configurations. It supports:
- Multiple starting positions (middlegame positions)
- Configurable engine parameters (temperature, MCTS simulations, etc.)
- JSON output for analysis
- Graceful interruption with partial results saved

Usage:
    python scripts/settings_arena.py --config configs/arena_config.yaml --games 2 --output results/
"""

import os
import sys
import json
import signal
import argparse
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from itertools import combinations

import chess
import chess.pgn
import numpy as np

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
# Script is in scripts/evaluation/, so go up 2 levels to reach project root
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("Warning: PyYAML not installed. Using JSON config format.")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EngineConfig:
    """Configuration for an engine instance."""
    name: str
    temperature: float = 1.0
    use_mcts: bool = False
    mcts_simulations: int = 800
    mcts_cpuct: float = 4.0
    # Coach Tal parameters
    use_coach_tal: bool = False
    lambda_psych: float = 0.3
    gamma_confusion: float = 0.5
    delta_soundness: float = 0.15
    top_k_candidates: int = 5
    
    @classmethod
    def from_dict(cls, data: dict) -> "EngineConfig":
        """Create config from dictionary, ignoring unknown keys."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class GameResult:
    """Result of a single game."""
    white_config: str
    black_config: str
    starting_fen: str
    result: str  # "1-0", "0-1", "1/2-1/2"
    moves: int
    termination: str  # "checkmate", "stalemate", "insufficient", "repetition", "50moves", "timeout"
    pgn: Optional[str] = None


@dataclass
class ConfigStats:
    """Aggregated statistics for a configuration."""
    wins: int = 0
    losses: int = 0
    draws: int = 0
    
    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws
    
    @property
    def score(self) -> float:
        if self.games == 0:
            return 0.0
        return (self.wins + self.draws * 0.5) / self.games
    
    def to_dict(self) -> dict:
        return {
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "games": self.games,
            "score": round(self.score, 3)
        }


@dataclass
class ArenaResults:
    """Complete results from an arena run."""
    timestamp: str
    model_path: str
    positions_count: int
    games_per_matchup: int
    total_games: int
    completed_games: int
    configs: List[Dict[str, Any]]
    results: Dict[str, Dict]  # config_name -> stats
    head_to_head: Dict[str, Dict]  # "A vs B" -> stats
    games: List[Dict]  # Individual game records
    interrupted: bool = False


# =============================================================================
# Engine Wrapper
# =============================================================================

class ConfigurableEngine:
    """
    Engine wrapper that can switch configurations without reloading the model.
    
    Uses TransformerEvaluator for inference, optionally with MCTS.
    """
    
    def __init__(self, model_path: str, use_pytorch: bool = True, device: str = "cuda"):
        self.model_path = model_path
        self.use_pytorch = use_pytorch
        self.device = device
        
        # Lazy load components
        self._evaluator = None
        self._mcts_controller = None
        self._coach_tal_selector = None
        self._current_config: Optional[EngineConfig] = None
        
    def _ensure_evaluator(self) -> None:
        """Lazy load the evaluator."""
        if self._evaluator is not None:
            return
            
        from src.coach_tal.evaluator import TransformerEvaluator
        
        self._evaluator = TransformerEvaluator(
            weights_path=self.model_path,
            use_pytorch=self.use_pytorch,
            temperature=1.0,
        )
        # Force initialization
        self._evaluator._ensure_initialized()
        print(f"Loaded model from {self.model_path}")
    
    def _ensure_mcts(self) -> None:
        """Lazy load MCTS controller if needed."""
        if self._mcts_controller is not None:
            return
        
        # Use SimpleMCTS (PyTorch-based, in-process) instead of the
        # parallel TensorFlow-based MCTSController to avoid NumPy version issues
        from src.mcts.simple_mcts import SimpleMCTSController
        
        self._mcts_controller = SimpleMCTSController(
            model_weights_path=self.model_path,
            use_pytorch=self.use_pytorch,
            c_puct=4.0,
            add_noise=False,
        )
        self._mcts_controller.start()
        print("SimpleMCTS controller started (PyTorch-based)")
    
    def _ensure_coach_tal(self, config: EngineConfig) -> None:
        """Lazy load Coach Tal selector if needed."""
        if self._coach_tal_selector is not None:
            return
        
        from src.coach_tal.selector import CoachTalSelector, CoachTalConfig
        
        coach_config = CoachTalConfig(
            weights_path=self.model_path,
            use_pytorch=self.use_pytorch,
            lambda_psych=config.lambda_psych,
            gamma_confusion=config.gamma_confusion,
            delta_soundness=config.delta_soundness,
            top_k_candidates=config.top_k_candidates,
            opponent_temperature=config.temperature * 1.2,  # Slightly higher for opponent
            user_temperature=config.temperature,
            enabled=True,
        )
        
        self._coach_tal_selector = CoachTalSelector(coach_config)
        print("Coach Tal selector initialized")
    
    def _get_move_coach_tal(self, board: chess.Board) -> chess.Move:
        """Get move using Coach Tal cognitive asymmetry optimization (raw policy)."""
        result = self._coach_tal_selector.select_from_board(board)
        return result.chosen_move
    
    def _get_move_mcts_coach_tal(self, board: chess.Board) -> chess.Move:
        """Get move using MCTS + Coach Tal re-ranking.
        
        This is the intended use of Coach Tal:
        1. Run MCTS to get high-quality candidates with stable Q-values
        2. Pass candidates to Coach Tal for cognitive asymmetry re-ranking
        """
        # Run MCTS search
        mcts_result = self._mcts_controller.run_search(
            fen=board.fen(),
            num_simulations=self._current_config.mcts_simulations
        )
        
        if mcts_result.error or not mcts_result.policy:
            print(f"MCTS error in Coach Tal path: {mcts_result.error}, falling back to policy")
            return self._get_move_policy(board)
        
        # Convert MCTS policy to Move -> float dict
        candidate_scores = {}
        for move_str, visit_proportion in mcts_result.policy.items():
            if isinstance(move_str, str):
                move = chess.Move.from_uci(move_str)
            else:
                move = move_str
            candidate_scores[move] = visit_proportion
        
        # Update Coach Tal config with MCTS Q-value for better soundness check
        # The root value from MCTS is more stable than raw network value
        # Coach Tal will use its evaluator for per-candidate values
        
        # Let Coach Tal re-rank the MCTS candidates
        result = self._coach_tal_selector.select(board, candidate_scores)
        
        return result.chosen_move
    
    def set_config(self, config: EngineConfig) -> None:
        """Set the current engine configuration."""
        self._current_config = config
        self._ensure_evaluator()
        self._evaluator.temperature = config.temperature
        
        if config.use_mcts:
            self._ensure_mcts()
        
        if config.use_coach_tal:
            self._ensure_coach_tal(config)
    
    def get_move(self, board: chess.Board) -> chess.Move:
        """Get the best move for the current position using current config."""
        if self._current_config is None:
            raise RuntimeError("Engine config not set. Call set_config() first.")
        
        if self._current_config.use_coach_tal and self._current_config.use_mcts:
            # MCTS + Coach Tal: The intended combination
            return self._get_move_mcts_coach_tal(board)
        elif self._current_config.use_coach_tal:
            # Coach Tal without MCTS (raw policy - not recommended)
            return self._get_move_coach_tal(board)
        elif self._current_config.use_mcts:
            return self._get_move_mcts(board)
        else:
            return self._get_move_policy(board)
    
    def _get_move_policy(self, board: chess.Board) -> chess.Move:
        """Get move directly from policy head (no search)."""
        _, policy = self._evaluator.evaluate(board)
        
        if not policy:
            # Fallback to first legal move
            return list(board.legal_moves)[0]
        
        # Select move based on temperature
        if self._current_config.temperature < 0.1:
            # Greedy selection
            return max(policy.items(), key=lambda x: x[1])[0]
        else:
            # Sample from policy
            moves = list(policy.keys())
            probs = np.array(list(policy.values()))
            probs = probs / probs.sum()  # Normalize
            idx = np.random.choice(len(moves), p=probs)
            return moves[idx]
    
    def _get_move_mcts(self, board: chess.Board) -> chess.Move:
        """Get move using MCTS search."""
        result = self._mcts_controller.run_search(
            fen=board.fen(),
            num_simulations=self._current_config.mcts_simulations
        )
        
        if result.error or not result.policy:
            print(f"MCTS error: {result.error}, falling back to policy")
            return self._get_move_policy(board)
        
        # Select best move by visit count
        best_move = max(result.policy.items(), key=lambda x: x[1])[0]
        
        # Convert string to Move if needed
        if isinstance(best_move, str):
            best_move = chess.Move.from_uci(best_move)
        
        return best_move
    
    def shutdown(self) -> None:
        """Clean up resources."""
        if self._mcts_controller is not None:
            self._mcts_controller.shutdown()
            self._mcts_controller = None
        # Coach Tal selector doesn't need explicit shutdown


# =============================================================================
# Arena
# =============================================================================

class SettingsArena:
    """
    Runs round-robin matches between engine configurations.
    """
    
    def __init__(
        self,
        model_path: str,
        configs: List[EngineConfig],
        positions: List[str],
        use_pytorch: bool = True,
        device: str = "cuda",
        max_moves: int = 200,
        save_pgn: bool = False,
    ):
        self.model_path = model_path
        self.configs = configs
        self.positions = positions
        self.use_pytorch = use_pytorch
        self.device = device
        self.max_moves = max_moves
        self.save_pgn = save_pgn
        
        self.engine = ConfigurableEngine(model_path, use_pytorch, device)
        
        # Results storage
        self.config_stats: Dict[str, ConfigStats] = {c.name: ConfigStats() for c in configs}
        self.head_to_head: Dict[str, ConfigStats] = {}
        self.games: List[GameResult] = []
        
        # Interruption handling
        self._interrupted = False
        signal.signal(signal.SIGINT, self._handle_interrupt)
    
    def _handle_interrupt(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\n\nInterrupted! Saving partial results...")
        self._interrupted = True
    
    def run_round_robin(self, games_per_matchup: int = 2) -> ArenaResults:
        """
        Run round-robin tournament between all configurations.
        
        Each pair plays `games_per_matchup` games from each starting position,
        alternating colors.
        
        Args:
            games_per_matchup: Number of games per position per matchup (typically 2 for color balance)
        
        Returns:
            ArenaResults with complete statistics
        """
        matchups = list(combinations(self.configs, 2))
        total_games = len(matchups) * len(self.positions) * games_per_matchup
        
        print(f"\n{'='*60}")
        print("SETTINGS ARENA - Round Robin Tournament")
        print(f"{'='*60}")
        print(f"Configurations: {len(self.configs)}")
        print(f"Starting positions: {len(self.positions)}")
        print(f"Games per matchup: {games_per_matchup}")
        print(f"Total games to play: {total_games}")
        print(f"{'='*60}\n")
        
        # Create progress iterator
        game_iter = self._generate_games(matchups, games_per_matchup)
        if TQDM_AVAILABLE:
            game_iter = tqdm(game_iter, total=total_games, desc="Playing games")
        
        completed = 0
        for config_a, config_b, position, game_num in game_iter:
            if self._interrupted:
                break
            
            # Alternate colors
            if game_num % 2 == 0:
                white_config, black_config = config_a, config_b
            else:
                white_config, black_config = config_b, config_a
            
            result = self._play_game(white_config, black_config, position)
            self.games.append(result)
            self._update_stats(result)
            completed += 1
            
            if not TQDM_AVAILABLE:
                self._print_progress(completed, total_games, result)
        
        return self._compile_results(total_games, completed)
    
    def _generate_games(self, matchups, games_per_matchup):
        """Generate all games to be played."""
        for config_a, config_b in matchups:
            for position in self.positions:
                for game_num in range(games_per_matchup):
                    yield config_a, config_b, position, game_num
    
    def _play_game(
        self,
        white_config: EngineConfig,
        black_config: EngineConfig,
        starting_fen: str
    ) -> GameResult:
        """Play a single game between two configurations."""
        board = chess.Board(starting_fen)
        
        # Create PGN game
        game = chess.pgn.Game()
        game.headers["Event"] = "Settings Arena"
        game.headers["Site"] = "Local"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["White"] = white_config.name
        game.headers["Black"] = black_config.name
        game.headers["WhiteTemp"] = str(white_config.temperature)
        game.headers["BlackTemp"] = str(black_config.temperature)
        game.setup(board)
        node = game
        
        move_count = 0
        
        while not board.is_game_over(claim_draw=True) and move_count < self.max_moves:
            # Select engine config based on turn
            if board.turn == chess.WHITE:
                self.engine.set_config(white_config)
            else:
                self.engine.set_config(black_config)
            
            try:
                move = self.engine.get_move(board)
                board.push(move)
                node = node.add_variation(move)
                move_count += 1
            except Exception as e:
                print(f"Error getting move: {e}")
                break
        
        # Determine result and termination reason
        if board.is_checkmate():
            result = "0-1" if board.turn == chess.WHITE else "1-0"
            termination = "checkmate"
        elif board.is_stalemate():
            result = "1/2-1/2"
            termination = "stalemate"
        elif board.is_insufficient_material():
            result = "1/2-1/2"
            termination = "insufficient"
        elif board.can_claim_threefold_repetition():
            result = "1/2-1/2"
            termination = "repetition"
        elif board.can_claim_fifty_moves():
            result = "1/2-1/2"
            termination = "50moves"
        elif move_count >= self.max_moves:
            result = "1/2-1/2"
            termination = "timeout"
        else:
            result = "1/2-1/2"
            termination = "unknown"
        
        game.headers["Result"] = result
        game.headers["Termination"] = termination
        
        # Convert PGN to string if saving
        pgn_str = None
        if self.save_pgn:
            import io
            pgn_io = io.StringIO()
            exporter = chess.pgn.FileExporter(pgn_io)
            game.accept(exporter)
            pgn_str = pgn_io.getvalue()
        
        return GameResult(
            white_config=white_config.name,
            black_config=black_config.name,
            starting_fen=starting_fen,
            result=result,
            moves=move_count,
            termination=termination,
            pgn=pgn_str,
        )
    
    def _update_stats(self, result: GameResult) -> None:
        """Update statistics based on game result."""
        white = result.white_config
        black = result.black_config
        
        # Update individual config stats
        if result.result == "1-0":
            self.config_stats[white].wins += 1
            self.config_stats[black].losses += 1
        elif result.result == "0-1":
            self.config_stats[white].losses += 1
            self.config_stats[black].wins += 1
        else:
            self.config_stats[white].draws += 1
            self.config_stats[black].draws += 1
        
        # Update head-to-head stats (always store as "A vs B" alphabetically)
        names = sorted([white, black])
        key = f"{names[0]} vs {names[1]}"
        
        if key not in self.head_to_head:
            self.head_to_head[key] = ConfigStats()
        
        # Stats are from perspective of first name (alphabetically)
        if result.result == "1-0":
            if white == names[0]:
                self.head_to_head[key].wins += 1
            else:
                self.head_to_head[key].losses += 1
        elif result.result == "0-1":
            if black == names[0]:
                self.head_to_head[key].wins += 1
            else:
                self.head_to_head[key].losses += 1
        else:
            self.head_to_head[key].draws += 1
    
    def _print_progress(self, completed: int, total: int, result: GameResult) -> None:
        """Print progress when tqdm is not available."""
        pct = completed / total * 100
        print(f"[{completed}/{total} ({pct:.1f}%)] {result.white_config} vs {result.black_config}: {result.result} ({result.termination}, {result.moves} moves)")
    
    def _compile_results(self, total_games: int, completed_games: int) -> ArenaResults:
        """Compile final results."""
        return ArenaResults(
            timestamp=datetime.now().isoformat(),
            model_path=self.model_path,
            positions_count=len(self.positions),
            games_per_matchup=2,  # This should be passed in
            total_games=total_games,
            completed_games=completed_games,
            configs=[asdict(c) for c in self.configs],
            results={name: stats.to_dict() for name, stats in self.config_stats.items()},
            head_to_head={key: stats.to_dict() for key, stats in self.head_to_head.items()},
            games=[asdict(g) for g in self.games],
            interrupted=self._interrupted,
        )
    
    def shutdown(self) -> None:
        """Clean up resources."""
        self.engine.shutdown()


# =============================================================================
# Config Loading
# =============================================================================

def load_config(config_path: str) -> Tuple[List[EngineConfig], List[str]]:
    """Load arena configuration from YAML or JSON file."""
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml'] and YAML_AVAILABLE:
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    
    # Parse engine configs
    configs = []
    for cfg_data in data.get('configs', []):
        configs.append(EngineConfig.from_dict(cfg_data))
    
    # Parse positions
    positions = data.get('positions', [])
    
    if not configs:
        raise ValueError("No engine configurations found in config file")
    
    if not positions:
        raise ValueError("No starting positions found in config file")
    
    return configs, positions


def print_results_summary(results: ArenaResults) -> None:
    """Print a summary of arena results to console."""
    print(f"\n{'='*60}")
    print("ARENA RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Completed: {results.completed_games}/{results.total_games} games")
    if results.interrupted:
        print("(Tournament was interrupted)")
    print()
    
    # Sort by score
    sorted_configs = sorted(
        results.results.items(),
        key=lambda x: x[1]['score'],
        reverse=True
    )
    
    print("Configuration Rankings:")
    print("-" * 50)
    print(f"{'Rank':<6}{'Config':<25}{'W-L-D':<12}{'Score':<8}")
    print("-" * 50)
    
    for rank, (name, stats) in enumerate(sorted_configs, 1):
        wld = f"{stats['wins']}-{stats['losses']}-{stats['draws']}"
        print(f"{rank:<6}{name:<25}{wld:<12}{stats['score']:.3f}")
    
    print()
    print("Head-to-Head Results:")
    print("-" * 50)
    for matchup, stats in results.head_to_head.items():
        wld = f"{stats['wins']}-{stats['losses']}-{stats['draws']}"
        print(f"  {matchup}: {wld}")
    
    print(f"{'='*60}\n")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Settings Arena - Compare engine configurations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/arena_config.yaml',
        help='Path to arena configuration file (YAML or JSON)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='saved_models/best_model_pytorch.pt',
        help='Path to model weights'
    )
    parser.add_argument(
        '--games', '-g',
        type=int,
        default=2,
        help='Number of games per matchup per position (use 2 for color balance)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cpu', 'cuda'],
        help='Device to run model on'
    )
    parser.add_argument(
        '--max-moves',
        type=int,
        default=200,
        help='Maximum moves per game before draw'
    )
    parser.add_argument(
        '--save-pgn',
        action='store_true',
        help='Include PGN strings in output'
    )
    parser.add_argument(
        '--no-pytorch',
        action='store_true',
        help='Use TensorFlow/Keras instead of PyTorch'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)
    
    # Load configuration
    print(f"Loading config from {args.config}...")
    configs, positions = load_config(args.config)
    print(f"  Loaded {len(configs)} configurations")
    print(f"  Loaded {len(positions)} starting positions")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create arena
    arena = SettingsArena(
        model_path=args.model,
        configs=configs,
        positions=positions,
        use_pytorch=not args.no_pytorch,
        device=args.device,
        max_moves=args.max_moves,
        save_pgn=args.save_pgn,
    )
    
    try:
        # Run tournament
        results = arena.run_round_robin(games_per_matchup=args.games)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f"arena_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(asdict(results), f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
        # Print summary
        print_results_summary(results)
        
    finally:
        arena.shutdown()


if __name__ == '__main__':
    main()

