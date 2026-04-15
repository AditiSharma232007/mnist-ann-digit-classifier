import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from src.config import IMAGE_HEIGHT, IMAGE_WIDTH, MODEL_PATH


def preprocess_image(image_path: str | Path):
    image = Image.open(image_path).convert("L")
    image = ImageOps.invert(image)
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    image_array = np.asarray(image, dtype="float32") / 255.0
    return np.expand_dims(image_array, axis=0)


def predict_digit(image_path: str | Path, model_path: str | Path = MODEL_PATH):
    model = tf.keras.models.load_model(model_path)
    processed_image = preprocess_image(image_path)
    probabilities = model.predict(processed_image, verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))

    return predicted_digit, confidence, probabilities


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m src.predict path_to_digit_image.png")

    predicted_digit, confidence, probabilities = predict_digit(sys.argv[1])

    print(f"Predicted digit: {predicted_digit}")
    print(f"Confidence: {confidence:.4f}")
    print("Class probabilities:")
    for digit, probability in enumerate(probabilities):
        print(f"{digit}: {probability:.4f}")


if __name__ == "__main__":
    main()

