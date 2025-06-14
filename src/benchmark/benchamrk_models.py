import chess
import chess.engine
import glob
import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Generator
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import csv
import io
from contextlib import redirect_stdout
from scipy.stats import pearsonr

# Go up two levels to the project root (e.g., chess_2.0/) and add it to the path.
# This allows for consistent, absolute imports from 'src' across all files.
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(PROJECT_ROOT)

from src.evaluator import SingleEvaluator


@dataclass
class EPDParser:
    """Parses all .epd files in a given folder."""

    folder_path: str
    file_paths: List[str] = field(init=False, default_factory=list)

    def __post_init__(self):
        """Finds all .epd files in the folder path upon initialization."""
        pattern = os.path.join(self.folder_path, "*.epd")
        self.file_paths = glob.glob(pattern)
        if not self.file_paths:
            print(f"Warning: No .epd files found in {self.folder_path}")

    def parse_files(self) -> Generator[Tuple[str, str], None, None]:
        """Yields the filename and FEN string for each position."""
        for file_path in self.file_paths:

            file_name = os.path.basename(file_path)
            yield from ((file_name, fen) for fen in self._parse_single_file(file_path))

    def _parse_single_file(self, file_path: str) -> Generator[str, None, None]:
        """Yields FENs from a single .epd file."""
        board = chess.Board()
        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()
                # Skip comments and empty lines. The '#' character is the standard for EPD comments.
                if line and not line.startswith("#"):
                    try:
                        board.set_epd(line)
                        yield board.fen()
                    except (ValueError, IndexError):

                        print(
                            f"Warning: Could not parse EPD line in {file_path}: {line}"
                        )


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for a single model to be benchmarked."""

    model_weights_path: str
    model_name: str


@dataclass(frozen=True)
class PositionResult:
    """Stores the evaluation results for a single position."""

    fen: str
    stockfish_eval: float
    model_eval: float
    error: float


class Benchmarker:
    """Orchestrates the benchmarking process against Stockfish."""

    def __init__(self, epd_folder_path: str, stockfish_path: str):
        self.parser = EPDParser(folder_path=epd_folder_path)
        self.stockfish_engine = self._create_stockfish_engine(stockfish_path)

    def _create_stockfish_engine(self, path: str) -> chess.engine.SimpleEngine | None:
        """Initializes the Stockfish engine from the provided path."""
        try:
            return chess.engine.SimpleEngine.popen_uci(path)
        except FileNotFoundError:
            print(f"Error: Stockfish engine not found at {path}.")
            print("Please install Stockfish or update the path in main().")
            return None

    def _get_stockfish_eval(self, fen: str) -> float:
        """Analyzes a FEN and returns a normalized centipawn score."""
        board = chess.Board(fen)

        info = self.stockfish_engine.analyse(board, chess.engine.Limit(time=0.05))
        score = info["score"].relative

        return score.score(mate_score=30000)

    def _validate_model_configs(
        self,
        models_to_benchmark: List[BenchmarkConfig],
        size_check_buffer_mb: float,
    ) -> bool:
        """Validates model paths and checks for significant size differences."""
        model_paths = [config.model_weights_path for config in models_to_benchmark]

        for path in model_paths:
            if not os.path.exists(path):
                print(f"Error: Model weights file not found at {path}.")
                print("Aborting benchmark.")
                return False

        if len(model_paths) < 2:
            return True

        buffer_bytes = size_check_buffer_mb * 1024 * 1024
        try:
            reference_size = os.path.getsize(model_paths[0])
            reference_model_name = models_to_benchmark[0].model_name

            for i in range(1, len(model_paths)):
                current_size = os.path.getsize(model_paths[i])
                current_model_name = models_to_benchmark[i].model_name
                diff = abs(current_size - reference_size)

                if diff > buffer_bytes:
                    ref_size_mb = reference_size / (1024 * 1024)
                    cur_size_mb = current_size / (1024 * 1024)
                    print(
                        f"Warning: Model '{current_model_name}' ({cur_size_mb:.2f} MB) "
                        f"differs in size from '{reference_model_name}' ({ref_size_mb:.2f} MB) "
                        f"by more than the {size_check_buffer_mb} MB buffer."
                    )
        except OSError as e:
            print(f"Error accessing file for size check: {e}")
            return False

        return True

    def run(
        self,
        models_to_benchmark: List[BenchmarkConfig],
        size_check_buffer_mb: float = 5.0,
    ) -> Dict[str, Dict[str, List[PositionResult]]]:
        """
        Runs the benchmark for all models against all EPD positions.

        Returns:
            A nested dictionary: {model_name: {file_name: [PositionResult, ...]}}
        """
        if not self.stockfish_engine:
            print("Aborting benchmark: Stockfish engine not available.")
            return {}

        if not self._validate_model_configs(models_to_benchmark, size_check_buffer_mb):
            return {}

        results = {
            config.model_name: defaultdict(list) for config in models_to_benchmark
        }

        try:
            all_positions = list(self.parser.parse_files())
            if not all_positions:
                print("Aborting benchmark: No valid EPD positions found.")
                return {}

            print(
                f"Found {len(all_positions)} positions across {len(self.parser.file_paths)} file(s)."
            )

            for config in models_to_benchmark:
                model_evaluator = SingleEvaluator(config.model_weights_path)

                progress_bar = tqdm(
                    all_positions,
                    desc=f"Benchmarking {config.model_name}",
                    unit="pos",
                )
                for file_name, fen in progress_bar:
                    stockfish_eval = self._get_stockfish_eval(fen)

                    # Create a board object from the FEN for the model evaluator
                    board = chess.Board(fen)
                    # The evaluator returns (value, uncertainty), we only need the value for now.
                    model_eval, _ = model_evaluator.evaluate(board)

                    error = abs(stockfish_eval - model_eval)

                    result = PositionResult(
                        fen=fen,
                        stockfish_eval=stockfish_eval,
                        model_eval=model_eval,
                        error=error,
                    )
                    results[config.model_name][file_name].append(result)
            return results
        finally:
            self.stockfish_engine.quit()


class ReportGenerator:
    """Generates and prints a summary report from benchmark results."""

    THEME_MAPPING = {
        "STS1.epd": "Undermining",
        "STS2.epd": "Open Files and Diagonals",
        "STS3.epd": "Knight Outposts",
        "STS4.epd": "Square Vacancy",
        "STS5.epd": "Bishop vs Knight",
        "STS6.epd": "Re-Capturing",
        "STS7.epd": "Offer of Simplification",
        "STS8.epd": "Advancement of f/g/h pawns",
        "STS9.epd": "Advancement of a/b/c Pawns",
        "STS10.epd": "Simplification",
        "STS11.epd": "Activity of the King",
        "STS12.epd": "Center Control",
        "STS13.epd": "Pawn Play in the Center",
        "STS14.epd": "Queens and Rooks to the 7th Rank",
        "STS15.epd": "Avoid Pointless Exchange",
    }

    def __init__(self, results: Dict[str, Dict[str, List[PositionResult]]]):
        self.results = results

    def _calculate_stats(
        self, position_results: List[PositionResult]
    ) -> Dict[str, float]:
        """Calculates performance metrics for a set of results."""
        if not position_results or len(position_results) < 2:
            return {"mae": 0, "mse": 0, "std_dev": 0, "pearson_r": 0}

        errors = [res.error for res in position_results]
        model_evals = [res.model_eval for res in position_results]
        stockfish_evals = [res.stockfish_eval for res in position_results]

        pearson_r, _ = pearsonr(model_evals, stockfish_evals)

        return {
            "mae": np.mean(errors),
            "mse": np.mean(np.square(errors)),
            "std_dev": np.std(errors),
            "pearson_r": pearson_r,
        }

    def print_summary(self):
        """Prints a formatted summary of the benchmark results."""
        if not self.results:
            print("No results available to generate a report.")
            return

        model_names = list(self.results.keys())
        all_file_names = sorted(
            {
                file_name
                for model_res in self.results.values()
                for file_name in model_res
            }
        )

        print("\n\n" + "=" * 25 + " BENCHMARK SUMMARY " + "=" * 25)

        for file_name in all_file_names:
            theme = self.THEME_MAPPING.get(file_name, "Custom Test")
            print(f"\n--- Concept: {theme} ({file_name}) ---")
            header = f"{'Model':<20} | {'MAE':<10} | {'MSE':<15} | {'Std Dev':<10} | {'Correlation':<12}"
            print(header)
            print("-" * len(header))

            for model_name in model_names:
                results_for_file = self.results.get(model_name, {}).get(file_name)
                if results_for_file:
                    stats = self._calculate_stats(results_for_file)
                    print(
                        f"{model_name:<20} | {stats['mae']:<10.2f} | {stats['mse']:<15.2f} | {stats['std_dev']:<10.2f} | {stats['pearson_r'] * 100:<11.2f}%"
                    )
                else:
                    print(
                        f"{model_name:<20} | {'N/A':<10} | {'N/A':<15} | {'N/A':<10} | {'N/A':<12}"
                    )

        print("\n\n" + "=" * 23 + " OVERALL MODEL PERFORMANCE " + "=" * 22)
        header = f"{'Model':<20} | {'Overall MAE':<15} | {'Overall MSE':<15} | {'Overall Std Dev':<15} | {'Correlation':<12}"
        print(header)
        print("-" * len(header))

        for model_name in model_names:
            all_model_results = [
                res
                for file_res in self.results.get(model_name, {}).values()
                for res in file_res
            ]
            if all_model_results:
                stats = self._calculate_stats(all_model_results)
                print(
                    f"{model_name:<20} | {stats['mae']:<15.2f} | {stats['mse']:<15.2f} | {stats['std_dev']:<15.2f} | {stats['pearson_r'] * 100:<11.2f}%"
                )
            else:
                print(
                    f"{model_name:<20} | {'N/A':<15} | {'N/A':<15} | {'N/A':<15} | {'N/A':<12}"
                )
        print("\n")

    def print_detailed_report(self, n: int = 5):
        """Prints the top N blunders and brilliancies for each model."""
        if not self.results:
            return

        print("\n\n" + "=" * 24 + " DETAILED ERROR ANALYSIS " + "=" * 24)
        for model_name, file_results in self.results.items():
            all_results = [
                res for res_list in file_results.values() for res in res_list
            ]
            if not all_results:
                continue

            print(f"\n--- Analysis for Model: {model_name} ---")

            # Blunders
            blunders = sorted(all_results, key=lambda r: r.error, reverse=True)[:n]
            print(f"\nTop {n} Blunders (Largest Errors):")
            for res in blunders:
                lichess_link = (
                    f"https://lichess.org/analysis/standard/{res.fen.replace(' ', '_')}"
                )
                print(
                    f"  - Error: {res.error:>8.2f} | Model: {res.model_eval:>8.2f} | Stockfish: {res.stockfish_eval:>8.2f}"
                )
                print(f"    Link: {lichess_link}")

            # Brilliancies
            brilliancies = sorted(all_results, key=lambda r: r.error)[:n]
            print(f"\nTop {n} Brilliancies (Smallest Errors):")
            for res in brilliancies:
                lichess_link = (
                    f"https://lichess.org/analysis/standard/{res.fen.replace(' ', '_')}"
                )
                print(
                    f"  - Error: {res.error:>8.2f} | Model: {res.model_eval:>8.2f} | Stockfish: {res.stockfish_eval:>8.2f}"
                )
                print(f"    Link: {lichess_link}")

    def save_results_to_csv(self, output_dir: str):
        """Saves the detailed results for each model to a CSV file."""
        if not self.results:
            return

        os.makedirs(output_dir, exist_ok=True)
        print(f"\nSaving detailed results to CSV files in '{output_dir}/'...")

        for model_name, file_results in self.results.items():
            all_results = [
                res for res_list in file_results.values() for res in res_list
            ]
            if not all_results:
                continue

            safe_filename = "".join(
                c for c in model_name if c.isalnum() or c in ("_", "-")
            ).rstrip()
            filepath = os.path.join(output_dir, f"{safe_filename}_results.csv")

            with open(filepath, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["fen", "model_eval", "stockfish_eval", "error"])
                for res in all_results:
                    writer.writerow(
                        [res.fen, res.model_eval, res.stockfish_eval, res.error]
                    )
            print(f" - Saved results for {model_name} to {filepath}")


def main():
    """Main entry point for the benchmarking script."""
    # Define paths relative to this script's location for robustness.
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

    # Dynamically discover models from the 'src' directory.
    models_dir = os.path.join(PROJECT_ROOT, "src")
    model_paths = glob.glob(os.path.join(models_dir, "*.weights.h5"))

    if not model_paths:
        print(f"Error: No model files matching '*.weights.h5' found in '{models_dir}'.")
        print("Please add some models to benchmark.")
        return

    models_to_test = []
    for path in model_paths:
        basename = os.path.basename(path)
        model_name = basename.replace("best_model_", "").replace(".weights.h5", "")
        models_to_test.append(
            BenchmarkConfig(model_weights_path=path, model_name=model_name)
        )

    print(f"Found {len(models_to_test)} models to benchmark:")
    for config in sorted(models_to_test, key=lambda c: c.model_name):
        print(f" - {config.model_name} (from {config.model_weights_path})")

    benchmarker = Benchmarker(
        epd_folder_path=os.path.join(SCRIPT_DIR, "epd_files/"),
        stockfish_path="/usr/games/stockfish",
    )
    results = benchmarker.run(models_to_test, size_check_buffer_mb=10)

    if results:
        report_generator = ReportGenerator(results)
        report_generator.print_summary()
        report_generator.print_detailed_report(n=5)

        # Define the output directory at the project root
        reports_dir = os.path.join(PROJECT_ROOT, "benchmark_reports")
        report_generator.save_results_to_csv(output_dir=reports_dir)


if __name__ == "__main__":
    main()
