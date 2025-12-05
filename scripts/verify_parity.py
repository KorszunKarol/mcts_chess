"""
Verify PyTorch <-> Flax weight parity for the Tal model.
"""

import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import torch
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze, unfreeze
from src.training_ppo.models.tal_jax import TalModelJAX
from src.training_ppo.models.jax_bridge import load_pytorch_weights_to_flax
from src.transformer_model_pytorch import create_model as create_pt_model


def _recursive_update(base, update):
    for k, v in update.items():
        if isinstance(v, dict):
            base[k] = _recursive_update(base.get(k, {}), v)
        else:
            base[k] = v
    return base


def main():
    print("Initializing parity check...")

    # 1) Create and save random PyTorch model
    pt_model = create_pt_model().eval()
    torch.save(pt_model.state_dict(), "temp_parity.pt")

    # 2) PyTorch forward pass
    pt_input = torch.randn(2, 34, 8, 8)
    with torch.no_grad():
        pt_val, pt_pol = pt_model(pt_input)

    # 3) Initialize Flax model and load converted weights
    jax_model = TalModelJAX()
    dummy = jnp.zeros((1, 8, 8, 34))
    init_vars = jax_model.init(jax.random.PRNGKey(0), dummy)
    loaded = load_pytorch_weights_to_flax("temp_parity.pt", jax_model)

    final_params = _recursive_update(unfreeze(init_vars["params"]), loaded["params"])
    final_stats = _recursive_update(unfreeze(init_vars["batch_stats"]), loaded["batch_stats"])
    variables = freeze({"params": final_params, "batch_stats": final_stats})

    # 4) Flax forward pass
    jax_input = jnp.transpose(pt_input.numpy(), (0, 2, 3, 1))
    jax_out = jax_model.apply(variables, jax_input, train=False)

    # 5) Compare outputs
    diff_val = np.abs(pt_val.numpy() - np.array(jax_out.value)).max()
    diff_pol = np.abs(pt_pol.numpy() - np.array(jax_out.policy_logits)).max()

    print(f"Value Max Diff:  {diff_val:.6f}")
    print(f"Policy Max Diff: {diff_pol:.6f}")

    if diff_val < 1e-4 and diff_pol < 1e-4:
        print("PARITY ACHIEVED")
    else:
        print("PARITY FAILED")


if __name__ == "__main__":
    main()

