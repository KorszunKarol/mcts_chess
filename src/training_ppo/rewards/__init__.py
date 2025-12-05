"""
Tal Reward computation for cognitive asymmetry training.

This module implements the reward function that encourages the agent
to create positions that are:
- Easy for the user (clear best move)
- Confusing for the opponent (many plausible but suboptimal moves)
- Sound (don't sacrifice objective strength)
"""

from src.training_ppo.rewards.normalizer import RunningMeanStd


def __getattr__(name):
    """Lazy import for JAX-dependent modules."""
    if name == "TalRewardEngine":
        from src.training_ppo.rewards.tal_reward import TalRewardEngine
        return TalRewardEngine
    elif name == "TalRewardConfig":
        from src.training_ppo.rewards.tal_reward import TalRewardConfig
        return TalRewardConfig
    elif name == "TalRewardEngineJIT":
        from src.training_ppo.rewards.tal_reward import TalRewardEngineJIT
        return TalRewardEngineJIT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TalRewardEngine",
    "TalRewardConfig",
    "TalRewardEngineJIT",
    "RunningMeanStd",
]

