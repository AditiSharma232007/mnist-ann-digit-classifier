import random

import numpy as np
import tensorflow as tf

from src.config import (
    BATCH_SIZE,
    EPOCHS,
    MODEL_DIR,
    MODEL_PATH,
    PLOT_DIR,
    RANDOM_SEED,
    VALIDATION_SPLIT,
)
from src.data_loader import describe_dataset, load_mnist_data
from src.evaluate import evaluate_model, plot_training_history, print_metrics
from src.model import build_ann_model


def set_reproducibility(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main():
    set_reproducibility()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    dataset_info = describe_dataset(x_train, y_train, x_test, y_test)

    print("Dataset Details")
    print("---------------")
    for key, value in dataset_info.items():
        print(f"{key}: {value}")

    model = build_ann_model()

    print("\nModel Architecture And Parameters")
    print("---------------------------------")
    model.summary()

    history = model.fit(
        x_train,
        y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    plot_training_history(history)
    metrics, matrix = evaluate_model(model, x_train, y_train, x_test, y_test)

    print_metrics(metrics)
    print("\nConfusion Matrix")
    print("----------------")
    print(matrix)

    model.save(MODEL_PATH)
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metrics and plots inside: {PLOT_DIR.parent}")


if __name__ == "__main__":
    main()

