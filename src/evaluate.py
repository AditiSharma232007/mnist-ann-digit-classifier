import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.config import CONFUSION_MATRIX_PLOT, METRICS_PATH, TRAINING_HISTORY_PLOT


def plot_training_history(history, output_path: Path = TRAINING_HISTORY_PLOT):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate_model(model, x_train, y_train, x_test, y_test):
    train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=0)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    probabilities = model.predict(x_test, verbose=0)
    y_pred = probabilities.argmax(axis=1)

    metrics = {
        "train_loss": float(train_loss),
        "train_accuracy": float(train_accuracy),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "accuracy_score": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_score_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }

    matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(matrix)
    save_metrics(metrics)

    return metrics, matrix


def plot_confusion_matrix(matrix, output_path: Path = CONFUSION_MATRIX_PLOT):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_metrics(metrics, output_path: Path = METRICS_PATH):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def print_metrics(metrics):
    print("\nEvaluation Metrics")
    print("------------------")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

