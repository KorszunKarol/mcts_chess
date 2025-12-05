"""
Metrics and logging utilities for Tal-RL training.

Provides WandB integration and custom metrics tracking for
monitoring cognitive asymmetry training progress.

Metric Categories:
    1. Cognitive (Tal/*)     - Survival mass, value gap, trap success
    2. Style (style/*)       - Material imbalance, chaos index
    3. Health (ppo/*)        - PPO algorithm stability
    4. Performance (perf/*) - SPS, VRAM usage
    5. Environment (env/*)   - Win rate, game length
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import time

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class MetricAggregator:
    """
    Aggregates metrics over multiple steps/episodes.
    
    Computes running statistics (mean, std, min, max) for
    each metric and supports periodic resetting.
    """
    
    _values: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    def add(self, metrics: Dict[str, float]) -> None:
        """Add a batch of metric values."""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self._values[key].append(float(value))
    
    def get_stats(self, reset: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all metrics.
        
        Args:
            reset: Whether to clear values after computing stats.
            
        Returns:
            Dict of metric_name -> {mean, std, min, max, count}.
        """
        stats = {}
        
        for key, values in self._values.items():
            if values:
                arr = np.array(values)
                stats[key] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "count": len(values),
                }
        
        if reset:
            self._values.clear()
        
        return stats
    
    def get_means(self, reset: bool = True) -> Dict[str, float]:
        """Get just the mean of each metric."""
        stats = self.get_stats(reset=False)
        means = {key: s["mean"] for key, s in stats.items()}
        if reset:
            self._values.clear()
        return means
    
    def reset(self) -> None:
        """Clear all stored values."""
        self._values.clear()


class TalMetricsLogger:
    """
    Logging utility for Tal-RL training.
    
    Integrates with WandB for experiment tracking and provides
    custom metrics specific to cognitive asymmetry training.
    
    Key Tal Metrics:
        - survival_mass_avg: Average probability victim places on sound moves
        - value_gap_avg: Average deception (Q_truth - V_victim)
        - trap_success_rate: % of games where victim blundered after high-gap position
        - agent_suicide_rate: % of games where agent lost due to unsound play
    
    Example:
        logger = TalMetricsLogger(
            project="tal-rl",
            config=training_config,
        )
        
        # During training
        logger.log_step(step_metrics)
        
        # After each iteration
        logger.log_iteration(iteration, ppo_metrics, tal_metrics)
        
        # At end
        logger.finish()
    """
    
    def __init__(
        self,
        project: str = "tal-rl",
        entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        use_wandb: bool = True,
    ):
        """
        Initialize the logger.
        
        Args:
            project: WandB project name.
            entity: WandB entity/team.
            config: Configuration to log.
            run_name: Optional run name.
            use_wandb: Whether to use WandB (False for testing).
        """
        self.use_wandb = use_wandb
        self.start_time = time.time()
        
        # Aggregators for different metric types
        self.step_aggregator = MetricAggregator()
        self.episode_aggregator = MetricAggregator()
        self.tal_aggregator = MetricAggregator()
        self.style_aggregator = MetricAggregator()
        self.safety_aggregator = MetricAggregator()
        
        # Initialize WandB
        if use_wandb:
            try:
                import wandb
                
                self.wandb = wandb
                
                wandb.init(
                    project=project,
                    entity=entity,
                    config=config,
                    name=run_name,
                )
                
                # Define custom metrics with step_metric for proper x-axis
                wandb.define_metric("iteration")
                wandb.define_metric("timesteps")
                
                # === Cognitive Asymmetry (The "Tal" Logic) ===
                wandb.define_metric("tal/*", step_metric="iteration")
                
                # === Style Metrics (Tal Personality Verification) ===
                wandb.define_metric("style/*", step_metric="iteration")
                
                # === Safety Checks (Hope Chess Prevention) ===
                wandb.define_metric("safety/*", step_metric="iteration")
                
                # === PPO Training Health ===
                wandb.define_metric("ppo/*", step_metric="iteration")
                
                # === Environment / Game Metrics ===
                wandb.define_metric("env/*", step_metric="iteration")
                
                # === Performance Metrics ===
                wandb.define_metric("perf/*", step_metric="iteration")
                
                logger.info(f"WandB initialized: {project}/{wandb.run.name}")
            except ImportError:
                logger.warning("WandB not installed. Logging to console only.")
                self.use_wandb = False
                self.wandb = None
        else:
            self.wandb = None
        
        # Counters
        self.total_timesteps = 0
        self.total_episodes = 0
        self.total_iterations = 0
    
    def log_step(
        self,
        metrics: Dict[str, float],
        num_envs: int = 1,
    ) -> None:
        """
        Log metrics for a single rollout step.
        
        Args:
            metrics: Step metrics (rewards, values, etc.).
            num_envs: Number of parallel environments.
        """
        self.step_aggregator.add(metrics)
        self.total_timesteps += num_envs
    
    def log_episode(
        self,
        metrics: Dict[str, float],
    ) -> None:
        """
        Log metrics for a completed episode.
        
        Args:
            metrics: Episode metrics (return, length, etc.).
        """
        self.episode_aggregator.add(metrics)
        self.total_episodes += 1
    
    def log_tal_metrics(
        self,
        metrics: Dict[str, float],
    ) -> None:
        """
        Log Tal-specific cognitive asymmetry metrics.
        
        Args:
            metrics: Tal metrics (survival mass, value gap, etc.).
        """
        self.tal_aggregator.add(metrics)
    
    def log_style_metrics(
        self,
        metrics: Dict[str, float],
    ) -> None:
        """
        Log style metrics for Tal personality verification.
        
        Args:
            metrics: Style metrics (material_imbalance, chaos_index, etc.).
        """
        self.style_aggregator.add(metrics)
    
    def log_safety_metrics(
        self,
        metrics: Dict[str, float],
    ) -> None:
        """
        Log safety metrics (hope chess prevention).
        
        Args:
            metrics: Safety metrics (agent_suicide_rate, soundness_violation, etc.).
        """
        self.safety_aggregator.add(metrics)
    
    def log_episode_batch(
        self,
        episode_stats: List[Dict[str, float]],
    ) -> None:
        """
        Log a batch of completed episodes from the rollout buffer.
        
        Args:
            episode_stats: List of episode stat dicts from buffer.get_episode_statistics().
        """
        for stats in episode_stats:
            self.episode_aggregator.add(stats)
            self.total_episodes += 1
    
    def log_iteration(
        self,
        iteration: int,
        ppo_metrics: Optional[Dict[str, float]] = None,
        tal_metrics: Optional[Dict[str, float]] = None,
        env_metrics: Optional[Dict[str, float]] = None,
        style_metrics: Optional[Dict[str, float]] = None,
        safety_metrics: Optional[Dict[str, float]] = None,
        episode_stats: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Log metrics at the end of a training iteration.
        
        Args:
            iteration: Current iteration number.
            ppo_metrics: PPO training metrics.
            tal_metrics: Tal reward metrics.
            env_metrics: Environment metrics.
            style_metrics: Style metrics (material, chaos).
            safety_metrics: Safety metrics (suicide rate).
            episode_stats: Episode-level statistics from buffer.
        """
        self.total_iterations = iteration
        
        # Compile all metrics
        log_dict = {
            "iteration": iteration,
            "timesteps": self.total_timesteps,
            "episodes": self.total_episodes,
            "time_elapsed": time.time() - self.start_time,
        }
        
        # Add aggregated step metrics
        step_means = self.step_aggregator.get_means()
        for key, value in step_means.items():
            log_dict[f"env/{key}"] = value
        
        # Add aggregated episode metrics
        episode_means = self.episode_aggregator.get_means()
        for key, value in episode_means.items():
            log_dict[f"env/{key}"] = value
        
        # Add aggregated Tal metrics
        tal_means = self.tal_aggregator.get_means()
        for key, value in tal_means.items():
            log_dict[f"tal/{key}"] = value
        
        # Add aggregated style metrics
        style_means = self.style_aggregator.get_means()
        for key, value in style_means.items():
            log_dict[f"style/{key}"] = value
        
        # Add aggregated safety metrics
        safety_means = self.safety_aggregator.get_means()
        for key, value in safety_means.items():
            log_dict[f"safety/{key}"] = value
        
        # Add direct metrics
        if ppo_metrics:
            for key, value in ppo_metrics.items():
                log_dict[f"ppo/{key}"] = value
        
        if tal_metrics:
            for key, value in tal_metrics.items():
                log_dict[f"tal/{key}"] = value
        
        if env_metrics:
            for key, value in env_metrics.items():
                log_dict[f"env/{key}"] = value
        
        if style_metrics:
            for key, value in style_metrics.items():
                log_dict[f"style/{key}"] = value
        
        if safety_metrics:
            for key, value in safety_metrics.items():
                log_dict[f"safety/{key}"] = value
        
        if episode_stats:
            for key, value in episode_stats.items():
                log_dict[f"env/{key}"] = value
        
        # === Performance Metrics ===
        elapsed = time.time() - self.start_time
        sps = self.total_timesteps / max(1, elapsed)
        log_dict["perf/steps_per_second"] = sps

        # GPU memory tracking (allocated vs reserved)
        vram_stats = self._get_vram_stats()
        if vram_stats is not None:
            alloc_gb, reserved_gb, total_gb = vram_stats
            log_dict["perf/vram_active_gb"] = alloc_gb
            log_dict["perf/vram_reserved_gb"] = reserved_gb
            log_dict["perf/vram_reserved_pct"] = (
                reserved_gb / total_gb * 100 if total_gb > 0 else 0.0
            )
        
        # Log to WandB
        if self.use_wandb and self.wandb is not None:
            self.wandb.log(log_dict)
        
        # Console logging
        self._log_to_console(iteration, log_dict)
    
    def _get_vram_stats(self) -> Optional[tuple]:
        """
        Get current GPU VRAM stats in GB.
        
        Returns:
            (allocated_gb, reserved_gb, total_gb) or None if CUDA unavailable.
        """
        try:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(device)
                total_gb = props.total_memory / (1024 ** 3)
                allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
                return allocated, reserved, total_gb
        except Exception:
            pass
        return None
    
    def _log_to_console(
        self,
        iteration: int,
        metrics: Dict[str, float],
    ) -> None:
        """Print key metrics to console."""
        elapsed = time.time() - self.start_time
        sps = self.total_timesteps / max(1, elapsed)
        
        # === Line 1: PPO Health ===
        policy_loss = metrics.get("ppo/policy_loss", 0)
        value_loss = metrics.get("ppo/value_loss", 0)
        entropy = metrics.get("ppo/entropy", 0)
        clip_frac = metrics.get("ppo/clip_fraction", 0)
        expl_var = metrics.get("ppo/explained_variance", 0)
        
        logger.info(
            f"Iter {iteration:4d} | "
            f"Steps: {self.total_timesteps:,} | "
            f"SPS: {sps:.0f} | "
            f"π_loss: {policy_loss:.4f} | "
            f"v_loss: {value_loss:.4f} | "
            f"H: {entropy:.3f} | "
            f"Clip: {clip_frac:.2f} | "
            f"ExpVar: {expl_var:.2f}"
        )
        
        # === Line 2: Tal Metrics (Cognitive Asymmetry) ===
        survival = metrics.get("tal/survival_mass_mean", 0)
        gap = metrics.get("tal/value_gap_mean", 0)
        material = metrics.get("style/material_imbalance_mean", 0)
        chaos = metrics.get("style/chaos_index_mean", 0)
        suicide_rate = metrics.get("safety/agent_suicide_rate", 0)
        
        logger.info(
            f"         TAL | "
            f"Surv: {survival:.3f} | "
            f"Gap: {gap:.3f} | "
            f"Mat: {material:+.1f} | "
            f"Chaos: {chaos:.1f} | "
            f"Suicide: {suicide_rate:.3f}"
        )
        
        # === Line 3: Game Stats (if available) ===
        win_rate = metrics.get("env/win_rate", None)
        game_len = metrics.get("env/game_length_mean", None)
        
        if win_rate is not None:
            vram_alloc = metrics.get("perf/vram_active_gb", 0)
            vram_res = metrics.get("perf/vram_reserved_gb", 0)
            logger.info(
                f"         ENV | "
                f"WinRate: {win_rate:.2%} | "
                f"GameLen: {game_len:.0f} | "
                f"VRAM: alloc {vram_alloc:.2f}GB / res {vram_res:.2f}GB"
        )
    
    def log_model_checkpoint(
        self,
        path: str,
        iteration: int,
    ) -> None:
        """Log model checkpoint to WandB."""
        if self.use_wandb and self.wandb is not None:
            artifact = self.wandb.Artifact(
                f"model-{iteration}",
                type="model",
            )
            artifact.add_file(path)
            self.wandb.log_artifact(artifact)
    
    def finish(self) -> None:
        """Finish logging and close WandB run."""
        if self.use_wandb and self.wandb is not None:
            self.wandb.finish()
        
        elapsed = time.time() - self.start_time
        logger.info(
            f"Training finished: {self.total_timesteps:,} steps, "
            f"{self.total_episodes:,} episodes, "
            f"{elapsed/3600:.2f} hours"
        )


def create_logger(
    config: Optional[Any] = None,
    use_wandb: bool = True,
) -> TalMetricsLogger:
    """
    Factory function to create TalMetricsLogger.
    
    Args:
        config: Training configuration.
        use_wandb: Whether to use WandB.
        
    Returns:
        TalMetricsLogger instance.
    """
    project = "tal-rl"
    entity = None
    
    if config is not None and hasattr(config, "training"):
        project = getattr(config.training, "wandb_project", project)
        entity = getattr(config.training, "wandb_entity", entity)
    
    # Convert config to dict for WandB
    config_dict = None
    if config is not None:
        try:
            from dataclasses import asdict
            config_dict = asdict(config)
        except (TypeError, ImportError):
            config_dict = dict(config) if hasattr(config, "__dict__") else None
    
    return TalMetricsLogger(
        project=project,
        entity=entity,
        config=config_dict,
        use_wandb=use_wandb,
    )

