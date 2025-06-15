from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory
import time
import numpy as np
import tensorflow as tf
import logging
from typing import List, Dict, Optional, TYPE_CHECKING
from unittest.mock import MagicMock

from src.model import create_model, ACTION_SPACE_SIZE

if TYPE_CHECKING:
    from src.mcts.controller import SharedMemoryConfig

logger = logging.getLogger(__name__)


def create_mock_model_for_manager():
    """Creates a simple, predictable mock model for testing."""
    inputs = tf.keras.Input(shape=(8, 8, 34))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(10)(x)
    value_output = tf.keras.layers.Dense(1, activation="tanh", name="value_head")(x)
    policy_output = tf.keras.layers.Dense(ACTION_SPACE_SIZE, name="policy_head")(x)
    return tf.keras.Model(inputs=inputs, outputs=[value_output, policy_output])


def create_error_raising_model():
    """Creates a mock model that raises an error on predict."""
    model = create_model()

    def error_predict(*args, **kwargs):
        raise RuntimeError("Test Inference Error")

    model.predict = MagicMock(side_effect=error_predict)
    return model


EvaluationRequest = Dict
EvaluationResponse = Dict


class EvaluationManager(Process):
    """
    Manages a GPU-based neural network model for batched evaluations.

    This process listens for evaluation requests from SearchWorkers, batches them
    together, runs them through the neural network, and sends the results back.

    Uses shared memory for high-throughput data transfer with SearchWorkers.
    """

    def __init__(
        self,
        request_q: Queue,
        response_qs: List[Queue],
        weights_path: str,
        batch_size: int = 32,
        max_wait_time_ms: float = 10.0,
        use_error_model: bool = False,
        use_mock_model: bool = False,
        shared_memory_config: Optional["SharedMemoryConfig"] = None,
    ):
        """
        Initializes the EvaluationManager.

        Args:
            request_q: Queue for receiving evaluation requests (buffer indices).
            response_qs: List of queues for sending evaluation results (buffer indices) to different workers.
            weights_path: Path to the model's weights file.
            batch_size: The maximum number of requests to batch together.
            max_wait_time_ms: Max time to wait for more requests before running
                              inference on a partial batch.
            use_error_model: If True, uses a model that raises errors.
            use_mock_model: If True, uses a simple, predictable mock model.
            shared_memory_config: Configuration for shared memory buffers.
        """
        super().__init__()
        self.request_q = request_q
        self.response_qs = response_qs
        self.weights_path = weights_path
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time_ms / 1000.0
        self.model = None
        self.use_error_model = use_error_model
        self.use_mock_model = use_mock_model
        self.shared_memory_config = shared_memory_config

        # Shared memory management
        self.shared_memory_blocks: Dict[str, SharedMemory] = {}
        self.input_arrays: Dict[str, np.ndarray] = {}
        self.output_arrays: Dict[str, np.ndarray] = {}

    def _setup_gpu(self):
        """Configure TensorFlow to not pre-allocate all GPU memory."""
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU memory growth enabled for {len(gpus)} GPU(s).")
        except RuntimeError as e:
            logger.error(f"Error setting up GPU: {e}")

    def _setup_shared_memory(self):
        """Attach to existing shared memory blocks created by the controller."""
        if not self.shared_memory_config:
            logger.warning(
                "No shared memory config provided, falling back to queue-based IPC"
            )
            return

        logger.info(
            f"Attaching to {len(self.shared_memory_config.buffer_names)} shared memory blocks..."
        )

        for buffer_name in self.shared_memory_config.buffer_names:
            try:
                # Attach to existing shared memory
                shm = SharedMemory(name=buffer_name)
                self.shared_memory_blocks[buffer_name] = shm

                # Create numpy arrays that view the shared memory
                input_size = self.shared_memory_config.get_input_size()
                output_size = self.shared_memory_config.get_output_size()

                # Input array (encoded board state)
                input_view = np.frombuffer(
                    shm.buf[:input_size], dtype=self.shared_memory_config.input_dtype
                ).reshape(self.shared_memory_config.input_shape)
                self.input_arrays[buffer_name] = input_view

                # Output array (value + policy logits)
                output_view = np.frombuffer(
                    shm.buf[input_size : input_size + output_size],
                    dtype=self.shared_memory_config.output_dtype,
                )
                self.output_arrays[buffer_name] = output_view

            except Exception as e:
                logger.error(f"Failed to attach to shared memory {buffer_name}: {e}")
                raise RuntimeError(f"Shared memory setup failed: {e}")

        logger.info(
            f"Successfully attached to {len(self.shared_memory_blocks)} shared memory blocks"
        )

    def _get_input_from_buffer(self, buffer_name: str) -> np.ndarray:
        """Get input data from a shared memory buffer."""
        return self.input_arrays[buffer_name].copy()  # Copy to avoid race conditions

    def _write_output_to_buffer(
        self, buffer_name: str, value: float, policy_logits: np.ndarray
    ):
        """Write output data to a shared memory buffer."""
        output_array = self.output_arrays[buffer_name]
        output_array[0] = value  # First element is the value
        output_array[1:] = policy_logits.flatten()  # Rest is policy logits

    def run(self):
        """
        Main loop to consume requests and produce evaluations.

        Uses shared memory for high-throughput data transfer:
        - Receives lightweight requests with buffer indices
        - Reads input data from shared memory buffers
        - Writes output data back to shared memory buffers
        - Sends lightweight responses with buffer indices
        """
        self._setup_gpu()
        self._setup_shared_memory()

        if self.use_error_model:
            self.model = create_error_raising_model()
            logger.info("Using error-raising mock model for testing.")
        elif self.use_mock_model:
            self.model = create_mock_model_for_manager()
            logger.info("Using simple mock model for testing.")
        else:
            self.model = create_model()
            if self.weights_path:
                try:
                    self.model.load_weights(self.weights_path)
                    logger.info("Model weights loaded successfully.")
                except Exception as e:
                    logger.error(
                        f"Could not load model weights from {self.weights_path}: {e}"
                    )
            else:
                logger.warning(
                    "No model weights path provided. Using a randomly initialized model."
                )

        requests_batch: List[EvaluationRequest] = []

        while True:
            # Get first request to start a batch
            if not requests_batch:
                try:
                    first_req = self.request_q.get()
                    if first_req is None:  # Sentinel check
                        logger.info("Received shutdown signal. Exiting.")
                        break
                    requests_batch.append(first_req)
                except (EOFError, BrokenPipeError):
                    logger.warning("Request queue closed. Shutting down manager.")
                    break

            # Collect additional requests for batching (up to batch_size or timeout)
            start_time = time.monotonic()
            while (
                len(requests_batch) < self.batch_size
                and (time.monotonic() - start_time) < self.max_wait_time
            ):
                if not self.request_q.empty():
                    try:
                        req = self.request_q.get_nowait()
                        if req is None:  # Sentinel check
                            self.request_q.put(
                                None
                            )  # Put it back for the outer loop to catch
                            break
                        requests_batch.append(req)
                    except Exception:
                        break
                else:
                    time.sleep(0.001)

            if not requests_batch:
                continue

            # Process batch using shared memory
            try:
                if self.shared_memory_config:
                    self._process_batch_with_shared_memory(requests_batch)
                else:
                    # Fallback to queue-based processing for backward compatibility
                    self._process_batch_with_queues(requests_batch)
            except Exception as e:
                logger.error(f"Error during batch processing: {e}", exc_info=True)
                # Send error responses
                for req in requests_batch:
                    worker_id = req["worker_id"]
                    self.response_qs[worker_id].put(
                        {
                            "worker_id": req["worker_id"],
                            "buffer_index": req.get("buffer_index") if req else None,
                            "error": str(e),
                        }
                    )

            requests_batch.clear()

    def _process_batch_with_shared_memory(
        self, requests_batch: List[EvaluationRequest]
    ):
        """Process a batch of requests using shared memory for data transfer."""
        batch_inputs = []

        # Read input data from shared memory buffers
        for req in requests_batch:
            buffer_name = self.shared_memory_config.buffer_names[req["buffer_index"]]
            input_data = self._get_input_from_buffer(buffer_name)
            batch_inputs.append(input_data)

        # Convert to tensor for model inference
        input_tensor = np.array(batch_inputs)
        logger.debug(
            f"Manager processing batch of {len(requests_batch)} using shared memory. "
            f"Input tensor shape: {input_tensor.shape}"
        )
        values, policy_logits = self.model.predict(input_tensor, verbose=0)

        # Write output data to shared memory and send response
        for i, req in enumerate(requests_batch):
            buffer_name = self.shared_memory_config.buffer_names[req["buffer_index"]]
            self._write_output_to_buffer(
                buffer_name, float(values[i]), policy_logits[i]
            )

            worker_id = req["worker_id"]
            response = {"worker_id": worker_id, "buffer_index": req["buffer_index"]}
            self.response_qs[worker_id].put(response)

    def _process_batch_with_queues(self, requests_batch: List[EvaluationRequest]):
        """Process a batch of requests using queues for data transfer (fallback)."""
        # This method is used when shared memory is not available
        batch_inputs = [req["encoded_state"] for req in requests_batch]
        input_tensor = np.array(batch_inputs)

        logger.debug(
            f"Manager processing batch of {len(requests_batch)} using queues. "
            f"Input tensor shape: {input_tensor.shape}"
        )

        values, policy_logits = self.model.predict(input_tensor, verbose=0)

        # Send responses back to workers via their dedicated queues
        for i, req in enumerate(requests_batch):
            worker_id = req["worker_id"]
            response = {
                "worker_id": worker_id,
                "value": float(values[i]),
                "policy_logits": policy_logits[i],
            }
            self.response_qs[worker_id].put(response)
