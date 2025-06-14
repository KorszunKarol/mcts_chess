import tensorflow as tf


def create_combined_loss(alpha: float = 0.5):
    """
    Creates a custom loss function that is a weighted sum of value and policy losses.

    This factory function returns a loss function compatible with the Keras API.
    The value 'alpha' weights the value_loss, and (1-alpha) weights the policy_loss.

    Args:
        alpha: A float between 0 and 1 that weights the value loss.

    Returns:
        A callable loss function.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")

    mse = tf.keras.losses.MeanSquaredError()
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    def combined_loss(y_true, y_pred):
        """
        Calculates the combined loss for the dual-head model.

        Args:
            y_true: A tuple of (game_outcome, move_policy_target).
                    - game_outcome: The ground-truth value (z).
                    - move_policy_target: The one-hot encoded target policy (pi).
            y_pred: A tuple of (predicted_value, policy_logits) from the model.
                    - predicted_value: The model's value head output (v).
                    - policy_logits: The model's policy head raw logits (p).

        Returns:
            A single scalar tensor representing the total weighted loss.
        """
        # Unpack the true values and the predictions
        value_true, policy_true = y_true
        value_pred, policy_pred = y_pred

        # Calculate individual losses
        value_loss = mse(y_true=value_true, y_pred=value_pred)
        policy_loss = cce(y_true=policy_true, y_pred=policy_pred)

        # Apply weighting
        total_loss = alpha * value_loss + (1 - alpha) * policy_loss

        return total_loss

    return combined_loss


# Example of how this might be used with model.compile():
#
# from tensorflow.keras.optimizers import Adam
#
# model = create_model()
# combined_loss_fn = create_combined_loss(alpha=0.5)
#
# model.compile(
#     optimizer=Adam(),
#     loss=combined_loss_fn,
#     # Since the loss function itself handles the two outputs, we don't need
#     # to specify loss weights or per-output losses here.
# )
