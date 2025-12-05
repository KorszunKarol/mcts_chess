import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from src.training_ppo.env.chess_env import VectorizedChessEnv

pytestmark = pytest.mark.gpu


def _sample_legal_actions(key, legal_mask):
    """Sample legal actions by masking out illegal logits."""
    logits = jnp.where(legal_mask, 0.0, -1e9)
    return jax.random.categorical(key, logits, axis=-1)


def _get_step_count(state):
    """Extract step counter, handling pgx naming differences."""
    for attr in ("step_count", "_step_count"):
        value = getattr(state, attr, None)
        if value is not None:
            arr = jnp.asarray(value)
            # Normalize to shape (B,) if stored as (B,1) or higher
            if arr.ndim > 1:
                arr = arr.reshape((arr.shape[0], -1))[:, 0]
            return arr
    raise AssertionError("State is missing step_count/_step_count")


def _with_updates(state, **updates):
    """Safely update pgx state fields for test manipulation."""
    if hasattr(state, "_replace"):
        return state._replace(**updates)
    if hasattr(state, "replace"):
        return state.replace(**updates)
    if hasattr(state, "_asdict"):
        data = state._asdict()
        data.update(updates)
        return state.__class__(**data)
    raise AssertionError("Unable to update state fields for test")


@pytest.mark.slow
def test_vectorization_stress_shapes(require_gpu):
    num_envs = 16
    env = VectorizedChessEnv(num_envs=num_envs, max_episode_steps=32)
    key = jax.random.PRNGKey(0)

    obs, state = env.reset(key)
    assert obs.shape == (num_envs, *env.observation_shape)

    legal_mask = state.legal_action_mask
    assert legal_mask.shape == (num_envs, env.action_space_size)

    step_key = key
    for _ in range(3):
        step_key, action_key = jax.random.split(step_key)
        actions = _sample_legal_actions(action_key, legal_mask)
        result, state = env.step(state, actions)

        legal_mask = result.legal_action_mask

        assert result.obs.shape == (num_envs, *env.observation_shape)
        assert legal_mask.shape == (num_envs, env.action_space_size)
        assert result.dones.shape == (num_envs,)
        assert result.truncated.shape == (num_envs,)


def test_auto_reset_resets_only_done_envs(require_gpu):
    num_envs = 4
    env = VectorizedChessEnv(num_envs=num_envs, max_episode_steps=8)
    key = jax.random.PRNGKey(1)

    _, state = env.reset(key)
    base_step_count = _get_step_count(state)

    terminated = state.terminated.at[0].set(True)
    marked_state = _with_updates(state, terminated=terminated)

    reset_state = env.auto_reset(marked_state, jax.random.PRNGKey(99))
    reset_steps = _get_step_count(reset_state)

    assert jnp.array_equal(reset_steps, base_step_count)
    assert jnp.array_equal(
        reset_state.current_player[1:],
        state.current_player[1:],
    )

    reset_obs = env._get_observations(reset_state)
    start_obs = env._get_observations(state)
    assert jnp.all(reset_obs[0] == start_obs[0])
    assert jnp.all(reset_obs[1:] == start_obs[1:])


def test_legal_action_mask_start_position(require_gpu):
    num_envs = 2
    env = VectorizedChessEnv(num_envs=num_envs, max_episode_steps=8)
    key = jax.random.PRNGKey(7)

    _, state = env.reset(key)
    legal_mask = env.get_legal_actions(state)

    assert legal_mask.shape == (num_envs, env.action_space_size)
    assert legal_mask.dtype == jnp.bool_

    move_counts = legal_mask.sum(axis=1)

    # Standard chess opening has 20 legal moves; ensure mask is populated and not empty
    assert jnp.all(move_counts == 20)
    assert jnp.all(legal_mask.any(axis=1))

