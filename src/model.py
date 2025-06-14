import tensorflow as tf
from src.move_mapping import ACTION_SPACE_SIZE


def create_model():
    """
    Creates the dual-head neural network model.

    The model consists of a shared "body" of residual blocks followed by two
    separate heads: a value head and a policy head.

    - Input: A (8, 8, 34) tensor representing the board state.
    - Value Head Output: A single scalar value in [-1, 1] predicting the game outcome.
    - Policy Head Output: A vector of raw logits of size ACTION_SPACE_SIZE,
      representing the relative probability of each possible move.

    Returns:
        A TensorFlow Keras model with one input and two outputs.
    """
    inputs = tf.keras.Input(shape=(8, 8, 34))

    # Initial convolution block
    x = tf.keras.layers.Conv2D(
        128, (3, 3), padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.0001)
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    # Residual blocks
    for i in range(2):
        residual = x
        filters = 128 * (2 ** min(i, 3))

        for _ in range(2):
            x = tf.keras.layers.Conv2D(
                filters,
                (3, 3),
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        if i > 0:
            residual = tf.keras.layers.Conv2D(
                filters,
                (1, 1),
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            )(residual)

        x = tf.keras.layers.Add()([x, residual])
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        # Squeeze-and-Excitation block
        se = tf.keras.layers.GlobalAveragePooling2D()(x)
        se = tf.keras.layers.Dense(filters // 4, activation="relu")(se)
        se = tf.keras.layers.Dense(filters, activation="sigmoid")(se)
        se = tf.keras.layers.Reshape((1, 1, filters))(se)
        x = tf.keras.layers.Multiply()([x, se])

        x = tf.keras.layers.Dropout(0.1)(x)

    # The output of the body, 'x', is the input to both heads.

    # Value Head
    value_tower = tf.keras.layers.GlobalAveragePooling2D()(x)
    value_tower = tf.keras.layers.Dense(256, activation="relu")(value_tower)
    value_output = tf.keras.layers.Dense(1, activation="tanh", name="value_head")(
        value_tower
    )

    # Policy Head
    policy_tower = tf.keras.layers.Conv2D(
        filters=32, kernel_size=1, padding="same", activation="relu"
    )(x)
    policy_tower = tf.keras.layers.Flatten()(policy_tower)
    policy_output = tf.keras.layers.Dense(ACTION_SPACE_SIZE, name="policy_head")(
        policy_tower
    )

    model = tf.keras.Model(inputs=inputs, outputs=[value_output, policy_output])
    return model
