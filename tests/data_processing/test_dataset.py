# tests/data_processing/test_dataset.py

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import unittest
import tempfile
import chess
import chess.pgn
import tensorflow as tf
import numpy as np

from src.data.dataset import (
    pgn_data_generator,
    create_dataset,
    STATE_SHAPE,
    POLICY_SHAPE,
)
from src.encoder import Encoder
from src.move_mapping import move_to_index


class TestPgnDataGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the path to the main test PGN file."""
        cls.main_test_pgn_path = os.path.join(
            os.path.dirname(__file__), "test_data.pgn"
        )
        if not os.path.exists(cls.main_test_pgn_path):
            raise FileNotFoundError(
                f"Main test PGN file not found at: {cls.main_test_pgn_path}"
            )
        cls.encoder = Encoder()

    def _create_temp_pgn(self, content: str) -> str:
        """Helper to create a temporary PGN file for isolated testing."""
        # The tempfile needs to be closed to be readable by another process/function
        # but kept around for the duration of the test.
        # So we create it, get its name, and then it will be cleaned up automatically.
        # For more control, we store them and clean up in tearDown.
        if not hasattr(self, "temp_files"):
            self.temp_files = []

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".pgn", encoding="utf-8"
        ) as f:
            f.write(content)
            path = f.name
        self.temp_files.append(path)
        return path

    def tearDown(self):
        """Clean up any temporary files created during tests."""
        if hasattr(self, "temp_files"):
            for f_path in self.temp_files:
                if os.path.exists(f_path):
                    os.remove(f_path)

    def test_generator_yields_correct_total_sample_count(self):
        """
        Tests the generator on the main test file to ensure the overall count of
        yielded samples from all valid games is correct.
        Game 1: 12 ply
        Game 4: 4 ply
        Game 5 (Draw): 4 ply
        Total expected samples: 12 + 4 + 4 = 20.
        """
        generator = pgn_data_generator(
            self.main_test_pgn_path, min_elo=2400, encoder=self.encoder
        )
        self.assertEqual(len(list(generator)), 20)

    def test_filter_low_elo_games(self):
        """Ensures games with at least one player below min_elo are filtered."""
        pgn_content = '[WhiteElo "2300"]\n[BlackElo "2800"]\n[Result "1-0"]\n[Termination "Normal"]\n\n1. e4 1-0'
        path = self._create_temp_pgn(pgn_content)
        generator = pgn_data_generator(path, min_elo=2400, encoder=self.encoder)
        self.assertEqual(len(list(generator)), 0)

    def test_filter_bad_termination_games(self):
        """Ensures games with non-standard terminations (e.g., 'Abandoned') are filtered."""
        pgn_content = '[WhiteElo "2500"]\n[BlackElo "2500"]\n[Result "1-0"]\n[Termination "Abandoned"]\n\n1. e4 1-0'
        path = self._create_temp_pgn(pgn_content)
        generator = pgn_data_generator(path, min_elo=2400, encoder=self.encoder)
        self.assertEqual(len(list(generator)), 0)

    def test_handle_missing_elo_header(self):
        """Ensures the generator handles games missing a required ELO header without crashing."""
        pgn_content = (
            '[BlackElo "2500"]\n[Result "1-0"]\n[Termination "Normal"]\n\n1. e4 1-0'
        )
        path = self._create_temp_pgn(pgn_content)
        generator = pgn_data_generator(path, min_elo=2400, encoder=self.encoder)
        self.assertEqual(len(list(generator)), 0)

    def test_handle_corrupted_game_data(self):
        """Ensures the generator handles malformed PGN movetext without crashing."""
        # Use a PGN with a clearly illegal move. `python-chess` will parse this
        # and add an error to `game.errors`, which our generator should catch.
        # Castling on move 1 is illegal.
        pgn_content = '[WhiteElo "2500"]\n[BlackElo "2500"]\n[Result "1-0"]\n[Termination "Normal"]\n\n1. O-O-O 1-0'
        path = self._create_temp_pgn(pgn_content)
        generator = pgn_data_generator(path, min_elo=2400, encoder=self.encoder)
        self.assertEqual(len(list(generator)), 0)

    def test_data_transformation_on_win_loss(self):
        """
        Tests that a win/loss is transformed correctly, checking the value flip
        between players' perspectives.
        """
        pgn_content = """
[Event "Test"]
[Result "1-0"]
[WhiteElo "2500"]
[BlackElo "2500"]
[Termination "Normal"]

1. e4 e5 1-0
"""
        path = self._create_temp_pgn(pgn_content)
        generator = pgn_data_generator(path, min_elo=2400, encoder=self.encoder)
        samples = list(generator)
        self.assertEqual(len(samples), 2)

        # 1. e4 (White's move, White won)
        _, _, value_white = samples[0]
        self.assertEqual(value_white, 1.0)

        # 1...e5 (Black's move, White won)
        _, _, value_black = samples[1]
        self.assertEqual(value_black, -1.0)

    def test_data_transformation_on_draw(self):
        """
        Tests that for a drawn game, the value target is always 0.0.
        """
        pgn_content = """
[Event "Test"]
[Result "1/2-1/2"]
[WhiteElo "2500"]
[BlackElo "2500"]
[Termination "Normal"]

1. d4 d5 2. Bf4 Bf5 1/2-1/2
"""
        path = self._create_temp_pgn(pgn_content)
        generator = pgn_data_generator(path, min_elo=2400, encoder=self.encoder)
        samples = list(generator)
        self.assertEqual(len(samples), 4)

        for _, _, value_target in samples:
            self.assertEqual(value_target, 0.0)

    def test_full_pipeline_output_structure(self):
        """
        Tests the `create_dataset` function to ensure the final batched output
        is correctly structured for model.fit().
        """
        dataset = create_dataset(self.main_test_pgn_path, batch_size=4, min_elo=2400)

        for element in dataset.take(1):
            inputs, outputs = element
            value_head, policy_head = outputs

            self.assertEqual(inputs.shape, (4, *STATE_SHAPE))
            self.assertEqual(inputs.dtype, tf.float32)
            self.assertEqual(value_head.shape, (4,))
            self.assertEqual(value_head.dtype, tf.float32)
            self.assertEqual(policy_head.shape, (4, *POLICY_SHAPE))
            self.assertEqual(policy_head.dtype, tf.float32)


if __name__ == "__main__":
    unittest.main()
