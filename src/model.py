import tensorflow as tf
from src.move_mapping import ACTION_SPACE_SIZE

def create_model():
    """
    Creates a powerful but practical dual-head model for a solo project.
    - Depth: 8 Residual Blocks
    - Width: Capped at 256 filters
    """
    inputs = tf.keras.Input(shape=(8, 8, 34))

    x = tf.keras.layers.Conv2D(
        128, (3, 3), padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.0001)
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    # --- CHANGE 1: Reduced depth for faster training ---
    for i in range(8): # Using 8 blocks instead of 10
        residual = x

        # --- CHANGE 2: Capped filter width for speed and efficiency ---
        # Capping at 256 filters (2**min(i, 1)) is a huge efficiency gain.
        filters = 128 * (2 ** min(i, 1))

        for _ in range(2):
            x = tf.keras.layers.Conv2D(
                filters,
                (3, 3),
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

        # The projection convolution for the skip connection must also use the new filter count
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
        se = tf.keras.layers.Dense(filters // 4, activation=tf.nn.silu)(se)
        se = tf.keras.layers.Dense(filters, activation="sigmoid")(se)
        se = tf.keras.layers.Reshape((1, 1, filters))(se)
        x = tf.keras.layers.Multiply()([x, se])

        x = tf.keras.layers.Dropout(0.1)(x)

    # --- Value Head ---
    value_tower = tf.keras.layers.GlobalAveragePooling2D()(x)
    value_tower = tf.keras.layers.Dense(256, activation=tf.nn.silu)(value_tower)
    value_output = tf.keras.layers.Dense(3, activation="softmax", name="value_head")(
        value_tower
    )

    # --- Policy Head ---
    policy_tower = tf.keras.layers.Conv2D(
        filters=2, kernel_size=1, padding="same", activation=tf.nn.silu
    )(x)
    policy_tower = tf.keras.layers.Flatten()(policy_tower)
    policy_output = tf.keras.layers.Dense(ACTION_SPACE_SIZE, name="policy_head")(
        policy_tower
    )

    model = tf.keras.Model(inputs=inputs, outputs=[value_output, policy_output])
    return model