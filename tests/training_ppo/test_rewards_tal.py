import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from src.training_ppo.rewards.tal_reward import TalRewardConfig, TalRewardEngine


def test_deception_gap_rewards_scale_with_value_gap():
    config = TalRewardConfig(
        alpha=0.3,
        beta=0.2,
        normalize_rewards=False,
        reward_clip=None,
    )
    engine = TalRewardEngine(config)

    pi_victim = jnp.array([[0.5, 0.5]], dtype=jnp.float32)
    sound_mask = jnp.array([[1, 1]], dtype=jnp.bool_)
    game_outcomes = jnp.array([0.0], dtype=jnp.float32)

    # Large deception gap -> higher reward
    reward_gap = engine.compute_rewards(
        q_truth=jnp.array([1.0], dtype=jnp.float32),
        v_victim=jnp.array([0.0], dtype=jnp.float32),
        pi_victim=pi_victim,
        game_outcomes=game_outcomes,
        sound_mask=sound_mask,
    )

    # No deception gap -> low/zero reward contribution from gap
    reward_no_gap = engine.compute_rewards(
        q_truth=jnp.array([1.0], dtype=jnp.float32),
        v_victim=jnp.array([1.0], dtype=jnp.float32),
        pi_victim=pi_victim,
        game_outcomes=game_outcomes,
        sound_mask=sound_mask,
    )

    assert reward_gap[0] > 0.0
    assert reward_gap[0] > reward_no_gap[0]
    assert reward_no_gap[0] <= 0.05  # near-zero when no deception


def test_survival_mass_rewards_penalize_sound_defenses():
    config = TalRewardConfig(
        alpha=0.3,
        beta=0.0,  # isolate survival component
        normalize_rewards=False,
        reward_clip=None,
    )
    engine = TalRewardEngine(config)

    game_outcomes = jnp.array([0.0], dtype=jnp.float32)
    q_truth = jnp.array([0.0], dtype=jnp.float32)
    v_victim = jnp.array([0.0], dtype=jnp.float32)

    # Scenario A: only one sound move, low probability mass on it
    pi_a = jnp.array([[0.7, 0.1, 0.1, 0.1]], dtype=jnp.float32)
    sound_mask_a = jnp.array([[0, 0, 0, 1]], dtype=jnp.bool_)
    reward_a = engine.compute_rewards(
        q_truth=q_truth,
        v_victim=v_victim,
        pi_victim=pi_a,
        game_outcomes=game_outcomes,
        sound_mask=sound_mask_a,
    )

    # Scenario B: all moves sound (high survival mass)
    pi_b = jnp.array([[0.25, 0.25, 0.25, 0.25]], dtype=jnp.float32)
    sound_mask_b = jnp.ones_like(pi_b, dtype=jnp.bool_)
    reward_b = engine.compute_rewards(
        q_truth=q_truth,
        v_victim=v_victim,
        pi_victim=pi_b,
        game_outcomes=game_outcomes,
        sound_mask=sound_mask_b,
    )

    assert reward_a[0] > reward_b[0]
    assert reward_a[0] > 0.0


