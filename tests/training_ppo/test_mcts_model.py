import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy
mctx = pytest.importorskip("mctx")

from src.training_ppo.mcts.batched_mcts import BatchedMCTS
from src.training_ppo.models.tal_jax import ModelOutput, TalModelJAX


class DummyModel:
    """Minimal model stub to drive deterministic MCTS behavior."""

    def __init__(self, action_dim: int, mate_action: int):
        self.action_dim = action_dim
        self.mate_action = mate_action

    def apply(self, params, obs, train: bool = False):
        batch = obs.shape[0]
        logits = jnp.full((batch, self.action_dim), -5.0)
        logits = logits.at[:, self.mate_action].set(5.0)

        # Favor win in value head (W/D/L ordering)
        value = jnp.tile(jnp.array([[0.0, 0.0, 1.0]]), (batch, 1))

        return ModelOutput(value=value, policy_logits=logits)


def test_tal_model_determinism():
    model = TalModelJAX()
    key = jax.random.PRNGKey(0)
    dummy_input = jnp.zeros((2, 34, 8, 8))

    params = model.init(key, dummy_input)

    out1 = model.apply(params, dummy_input, train=False)
    out2 = model.apply(params, dummy_input, train=False)

    assert jnp.array_equal(out1.value, out2.value)
    assert jnp.array_equal(out1.policy_logits, out2.policy_logits)


@pytest.mark.slow
@pytest.mark.gpu
def test_batched_mcts_prefers_forced_mate(require_gpu):
    action_dim = 4
    mate_action = 2
    model = DummyModel(action_dim=action_dim, mate_action=mate_action)

    mcts = BatchedMCTS(
        model=model,
        num_simulations=16,
        max_num_considered_actions=action_dim,
        discount=1.0,
        temperature=1.0,
        use_gumbel=True,
    )

    batch_size = 1
    obs = jnp.zeros((batch_size, 1, 1, 1))
    env_state = obs  # simple placeholder state
    legal_mask = jnp.array([[True, True, True, False]])
    key = jax.random.PRNGKey(0)

    def env_step_fn(state, action):
        done = action == mate_action
        reward = jnp.where(done, 1.0, 0.0)
        next_obs = jnp.zeros_like(obs)
        next_state = state
        return next_obs, reward, done, next_state

    output = mcts.search(
        params={},
        key=key,
        obs=obs,
        env_state=env_state,
        legal_mask=legal_mask,
        env_step_fn=env_step_fn,
    )

    mate_prob = float(output.policy[0, mate_action])

    assert output.policy.shape == (batch_size, action_dim)
    assert mate_prob > 0.8  # policy should mass on the forced mate


