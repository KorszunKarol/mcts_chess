import pytest


@pytest.mark.slow
def test_agent_turn_value_gap_integration():
    jax = pytest.importorskip("jax")
    jnp = jax.numpy

    from src.training_ppo.config import PPOTalConfig
    from src.training_ppo.env.chess_env import create_env
    from src.training_ppo.models.tal_jax import create_model as create_jax_model
    from src.training_ppo.models.victim import create_victim
    from src.training_ppo.mcts.batched_mcts import create_mcts
    from src.training_ppo.rewards.tal_reward import TalRewardEngineJIT

    # --- Config: tiny, deterministic, GPU-friendly ---
    config = PPOTalConfig()
    config.env.num_envs = 2
    config.env.max_episode_steps = 32
    config.mcts.num_simulations = 4
    config.mcts.max_num_considered_actions = 4
    config.reward.alpha = 0.0
    config.reward.beta = 1.0  # isolate gap
    seed = 0

    # --- Setup models and env ---
    env = create_env(config)
    jax_key = jax.random.PRNGKey(seed)
    jax_key, init_key, mcts_key, reset_key = jax.random.split(jax_key, 4)

    # Init JAX model/params
    jax_model = create_jax_model()
    dummy_input = jnp.zeros((1, 8, 8, config.agent.input_channels))
    variables = jax_model.init(init_key, dummy_input)
    jax_params = {
        "params": variables["params"],
        "batch_stats": variables.get("batch_stats", {}),
    }

    victim = create_victim(jax_model, jax_params, config.victim)
    mcts = create_mcts(jax_model, config.mcts, use_simplified=False)

    # Reset env
    obs, state = env.reset(reset_key)
    legal_mask = env.get_legal_actions(state)
    is_agent = env.is_agent_turn(state)
    assert is_agent.all(), "Test expects agent to move first"

    # Env dynamics wrapper for MCTS
    def env_step_fn(state_in, actions_batch):
        step_result, next_state = env.step(state_in, actions_batch)
        return step_result.obs, step_result.rewards, step_result.dones, next_state

    # --- Agent turn: run MCTS to get q_truth ---
    mcts_output = mcts.search(
        jax_params,
        mcts_key,
        obs,
        state,
        legal_mask,
        env_step_fn=env_step_fn,
    )
    q_truth = mcts_output.q_value
    q_truth_adj = q_truth + 0.1  # ensure non-zero baseline

    # Victim perception on the same obs
    victim_output = victim(obs)
    # Shift to guarantee a measurable gap while still using victim forward pass
    v_victim = victim_output.value + 0.25
    pi_victim = victim_output.policy

    # Use legal mask as sound mask; alpha=0 so survival term is inactive
    sound_mask = legal_mask
    game_outcomes = jnp.zeros_like(q_truth_adj)

    rewards_real, _ = TalRewardEngineJIT.compute_rewards(
        q_truth_adj,
        v_victim,
        pi_victim,
        game_outcomes,
        sound_mask,
        alpha=config.reward.alpha,
        beta=config.reward.beta,
    )

    rewards_bugged, _ = TalRewardEngineJIT.compute_rewards(
        q_truth_adj,
        jnp.zeros_like(q_truth_adj),
        pi_victim,
        game_outcomes,
        sound_mask,
        alpha=config.reward.alpha,
        beta=config.reward.beta,
    )

    # If victim value is honored, rewards should drop by v_victim vs the zeroed baseline
    gap_from_rewards = rewards_bugged - rewards_real

    # Gap should change rewards noticeably (direction can vary with victim value)
    assert jnp.any(jnp.abs(gap_from_rewards) > 0.1), "Gap should meaningfully affect rewards"
    assert jnp.allclose(
        gap_from_rewards,
        v_victim,
        atol=1e-5,
    ), "Reward gap should equal victim value contribution when beta=1, alpha=0"

