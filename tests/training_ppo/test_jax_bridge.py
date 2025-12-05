import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy
torch = pytest.importorskip("torch")

from src.training_ppo.models.jax_bridge import jax_to_torch, torch_to_jax

pytestmark = pytest.mark.gpu


def test_jax_to_torch_dlpack_integrity(require_gpu):
    gpu_devices = jax.devices("gpu")
    if not gpu_devices:
        pytest.skip("GPU is required for zero-copy DLPack transfer")

    if not torch.cuda.is_available() or not getattr(torch.version, "cuda", None):
        pytest.skip("Torch is not compiled with CUDA enabled")

    device = gpu_devices[0]

    source = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    source_gpu = jax.device_put(source, device=device)

    tensor = jax_to_torch(source_gpu, device="cuda")

    assert tensor.is_cuda
    assert tensor.device.type == "cuda"
    assert tensor.dtype == torch.float32
    assert torch.allclose(tensor.cpu(), torch.tensor([1.0, 2.0, 3.0]))

    # Mutate torch view and ensure the JAX view sees the update (zero-copy)
    tensor[0] = 9.0
    round_trip = torch_to_jax(tensor)

    assert jnp.allclose(round_trip, jnp.array([9.0, 2.0, 3.0], dtype=jnp.float32))
    assert round_trip.device().platform == "gpu"

