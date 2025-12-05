"""
Metrics and logging utilities for Tal-RL training.

This module provides WandB integration and custom metrics
for tracking cognitive asymmetry training progress.
"""

from src.training_ppo.metrics.logger import TalMetricsLogger, MetricAggregator

__all__ = [
    "TalMetricsLogger",
    "MetricAggregator",
]

