"""
Configuration schemas for Tal-RL PPO training.

Uses Pydantic for validation and type safety.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path


@dataclass
class EnvConfig:
    """Environment configuration."""
    num_envs: int = 4096
    max_episode_steps: int = 512  # Maximum moves per game
    

@dataclass
class AgentConfig:
    """Agent model configuration."""
    weights_path: str = "saved_models/tal_v1.pt"
    input_channels: int = 34
    action_space_size: int = 4672
    

@dataclass  
class VictimConfig:
    """Victim model configuration (frozen, high temperature)."""
    temperature: float = 1.5
    # Use same weights as agent, just different temperature
    share_weights: bool = True
    

@dataclass
class MCTSConfig:
    """Batched MCTS configuration."""
    num_simulations: int = 50
    max_num_considered_actions: int = 16
    discount: float = 1.0  # No discounting for chess
    # Whether to use environment dynamics or learned model
    use_env_dynamics: bool = True
    

@dataclass
class TalRewardConfig:
    """Tal reward (cognitive asymmetry) configuration."""
    alpha: float = 0.3  # Weight for survival mass penalty (1 - M_surv)
    beta: float = 0.2   # Weight for value gap bonus
    delta_soundness: float = 0.15  # Maximum allowed value drop for soundness
    normalize_rewards: bool = True
    reward_clip: Optional[float] = 10.0  # Clip extreme rewards
    

@dataclass
class PPOConfig:
    """PPO algorithm configuration."""
    lr: float = 3e-4
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None  # None = no value clipping
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 256
    num_steps: int = 128  # Rollout length before update
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantage: bool = True
    

@dataclass
class TrainingConfig:
    """Overall training configuration."""
    total_timesteps: int = 10_000_000
    log_interval: int = 10  # Log every N iterations
    save_interval: int = 100  # Save checkpoint every N iterations
    eval_interval: int = 50  # Evaluate every N iterations
    checkpoint_dir: str = "checkpoints/ppo_tal"
    wandb_project: str = "tal-rl"
    wandb_entity: Optional[str] = None
    seed: int = 42
    use_mixed_precision: bool = False  # Use mixed precision training (FP16)
    

@dataclass
class PPOTalConfig:
    """Complete configuration for Tal-RL PPO training."""
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    victim: VictimConfig = field(default_factory=VictimConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    reward: TalRewardConfig = field(default_factory=TalRewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "PPOTalConfig":
        """Load configuration from YAML file."""
        import yaml
        
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        return cls(
            env=EnvConfig(**data.get("env", {})),
            agent=AgentConfig(**data.get("agent", {})),
            victim=VictimConfig(**data.get("victim", {})),
            mcts=MCTSConfig(**data.get("mcts", {})),
            reward=TalRewardConfig(**data.get("reward", {})),
            ppo=PPOConfig(**data.get("ppo", {})),
            training=TrainingConfig(**data.get("training", {})),
        )
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        import yaml
        from dataclasses import asdict
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

