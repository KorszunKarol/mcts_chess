import tensorflow as tf
from src.move_mapping import ACTION_SPACE_SIZE

def _cnn_block(input_tensor, filters):
    """A single residual block for the CNN stem."""
    residual = input_tensor

    x = tf.keras.layers.Conv2D(
        filters, (3, 3), padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.0001)
    )(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.nn.silu)(x)

    x = tf.keras.layers.Conv2D(
        filters, (3, 3), padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.0001)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if residual.shape[-1] != filters:
        residual = tf.keras.layers.Conv2D(
            filters, (1, 1), padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.0001)
        )(residual)

    x = tf.keras.layers.Add()([x, residual])
    x = tf.keras.layers.Activation(tf.nn.silu)(x)
    return x

def _transformer_encoder(inputs, num_heads, key_dim, ff_dim, dropout_rate=0.1):
    """A single Transformer Encoder block."""
    # Multi-Head Self-Attention
    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate
    )(inputs, inputs)
    attention_output = tf.keras.layers.Dropout(dropout_rate)(attention_output)
    attention_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)

    # Feed-Forward Network
    ffn_output = tf.keras.layers.Dense(ff_dim, activation=tf.nn.silu)(attention_output)
    ffn_output = tf.keras.layers.Dropout(dropout_rate)(ffn_output)
    ffn_output = tf.keras.layers.Dense(inputs.shape[-1])(ffn_output)

    return tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)


def create_model():
    """
    Creates a novel Hybrid CNN-Transformer model (Upgraded & Corrected).
    - A CNN stem extracts spatial features.
    - Transformer encoders apply global reasoning.
    - A dedicated policy head pathway preserves spatial information.
    """
    hybrid_input = tf.keras.Input(shape=(8, 8, 34), name='hybrid_board_input')

    # --- 1. CNN Stem ---
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same")(hybrid_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.nn.silu)(x)

    x = _cnn_block(x, filters=128)
    x = _cnn_block(x, filters=128)
    x = _cnn_block(x, filters=256)
    x = _cnn_block(x, filters=256)

    # --- 2. Prepare for Transformer ---
    _, h, w, c = x.shape
    transformer_input = tf.keras.layers.Reshape((h * w, c))(x) # Shape: (None, 64, 256)

    # --- 3. Transformer Encoder Body ---
    transformer_output = transformer_input
    for _ in range(6):
        transformer_output = _transformer_encoder(transformer_output, num_heads=8, key_dim=32, ff_dim=1024)

    # --- 4. Separate Pathways for Policy and Value Heads ---

    # --- Value Head Pathway ---
    # For the value, we can average all spatial information to get a single score.
    value_representation = tf.keras.layers.GlobalAveragePooling1D()(transformer_output)
    value_tower = tf.keras.layers.Dense(256, activation=tf.nn.silu)(value_representation)
    value_output = tf.keras.layers.Dense(3, activation="softmax", name="value_head")(value_tower)

    # --- Policy Head Pathway (THE FIX) ---
    # For the policy, we MUST preserve the spatial information from the Transformer.
    # We reshape the (64, 256) sequence back into a (8, 8, 256) grid.
    policy_spatial_representation = tf.keras.layers.Reshape((h, w, c))(transformer_output)

    # Now we apply the proven, efficient convolutional policy head.
    policy_tower = tf.keras.layers.Conv2D(
        filters=2, kernel_size=1, padding="same", activation=tf.nn.silu
    )(policy_spatial_representation)
    policy_tower = tf.keras.layers.Flatten()(policy_tower)
    policy_output = tf.keras.layers.Dense(ACTION_SPACE_SIZE, name="policy_head")(policy_tower)

    # --- 5. Create the Final Model ---
    model = tf.keras.Model(
        inputs=hybrid_input,
        outputs=[value_output, policy_output] # Return list to match data pipeline
    )

    return model