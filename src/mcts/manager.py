from multiprocessing import Process, Queue
import time
import numpy as np
import tensorflow as tf
import logging
from typing import List, Dict
from unittest.mock import MagicMock

from src.model import create_model, ACTION_SPACE_SIZE

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
    """

    def __init__(
        self,
        request_q: Queue,
        response_q: Queue,
        weights_path: str,
        batch_size: int = 32,
        max_wait_time_ms: float = 10.0,
        use_error_model: bool = False,
        use_mock_model: bool = False,
    ):
        """
        Initializes the EvaluationManager.

        Args:
            request_q: Queue for receiving evaluation requests.
            response_q: Queue for sending evaluation results.
            weights_path: Path to the model's weights file.
            batch_size: The maximum number of requests to batch together.
            max_wait_time_ms: Max time to wait for more requests before running
                              inference on a partial batch.
            use_error_model: If True, uses a model that raises errors.
            use_mock_model: If True, uses a simple, predictable mock model.
        """
        super().__init__()
        self.request_q = request_q
        self.response_q = response_q
        self.weights_path = weights_path
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time_ms / 1000.0
        self.model = None
        self.use_error_model = use_error_model
        self.use_mock_model = use_mock_model

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

    def run(self):
        """
        Main loop to consume requests and produce evaluations.
        """
        self._setup_gpu()

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

            if not requests_batch:
                try:
                    first_req = self.request_q.get()
                    requests_batch.append(first_req)
                except (EOFError, BrokenPipeError):
                    logger.warning("Request queue closed. Shutting down manager.")
                    break

            start_time = time.monotonic()
            while (
                len(requests_batch) < self.batch_size
                and (time.monotonic() - start_time) < self.max_wait_time
            ):
                if not self.request_q.empty():
                    try:
                        requests_batch.append(self.request_q.get_nowait())
                    except Exception:
                        break
                else:

                    time.sleep(0.001)

            if not requests_batch:
                continue

            encoded_states = [req["encoded_state"] for req in requests_batch]
            input_tensor = np.array(encoded_states)
            logger.debug(
                f"Manager processing batch of {len(requests_batch)}. "
                f"Input tensor shape: {input_tensor.shape}"
            )

            try:
                values, policy_logits = self.model.predict(input_tensor, verbose=0)
                logger.debug(
                    f"Manager received predictions. Values shape: {values.shape}, "
                    f"Policy logits shape: {policy_logits.shape}"
                )

                for i, req in enumerate(requests_batch):
                    response = {
                        "worker_id": req["worker_id"],
                        "value": float(values[i][0]),
                        "policy_logits": policy_logits[i],
                    }
                    self.response_q.put(response)
                    logger.debug(f"Manager sent response to worker {req['worker_id']}")
            except Exception as e:
                logger.error(f"Error during model inference: {e}", exc_info=True)

                for req in requests_batch:
                    self.response_q.put(
                        {"worker_id": req["worker_id"], "error": str(e)}
                    )

            requests_batch.clear()
