import numpy as np
import tensorflow as tf

from src.config import TEST_SAMPLES, TRAIN_SAMPLES


def load_mnist_data():
    """Load MNIST, normalize pixels, and select the required sample counts."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train[:TRAIN_SAMPLES].astype("float32") / 255.0
    y_train = y_train[:TRAIN_SAMPLES]

    x_test = x_test[:TEST_SAMPLES].astype("float32") / 255.0
    y_test = y_test[:TEST_SAMPLES]

    return (x_train, y_train), (x_test, y_test)


def describe_dataset(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
    """Return the important dataset details requested in the notebook."""
    return {
        "train_shape": list(x_train.shape),
        "test_shape": list(x_test.shape),
        "train_samples": int(x_train.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "image_size": f"{x_train.shape[1]}x{x_train.shape[2]}",
        "color_mode": "black and white / grayscale",
        "classes": sorted([int(label) for label in np.unique(np.concatenate([y_train, y_test]))]),
    }

