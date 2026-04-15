import tensorflow as tf

from src.config import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES


def build_ann_model():
    """Build the required three-hidden-layer ANN model."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH), name="input_28x28_image"),
            tf.keras.layers.Flatten(name="flatten_784_features"),
            tf.keras.layers.Dense(64, name="hidden_dense_64_1"),
            tf.keras.layers.LeakyReLU(alpha=0.01, name="leaky_relu_1"),
            tf.keras.layers.Dense(64, name="hidden_dense_64_2"),
            tf.keras.layers.LeakyReLU(alpha=0.01, name="leaky_relu_2"),
            tf.keras.layers.Dense(32, name="hidden_dense_32"),
            tf.keras.layers.LeakyReLU(alpha=0.01, name="leaky_relu_3"),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="output_digit_probabilities"),
        ],
        name="mnist_three_hidden_layer_ann",
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

