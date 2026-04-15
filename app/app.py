from pathlib import Path
import sys

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import IMAGE_HEIGHT, IMAGE_WIDTH, MODEL_PATH  # noqa: E402


model = None


def get_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. Train first with: python -m src.train"
            )
        model = tf.keras.models.load_model(MODEL_PATH)
    return model


def preprocess_canvas_image(image):
    if image is None:
        return None

    if isinstance(image, dict):
        image = image.get("composite")

    image = Image.fromarray(image).convert("L")
    image = ImageOps.invert(image)
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    image_array = np.asarray(image, dtype="float32") / 255.0
    return np.expand_dims(image_array, axis=0)


def classify_digit(image):
    processed_image = preprocess_canvas_image(image)
    if processed_image is None:
        return "Draw or upload a digit first.", {}

    classifier = get_model()
    probabilities = classifier.predict(processed_image, verbose=0)[0]
    labels = {str(index): float(probability) for index, probability in enumerate(probabilities)}
    prediction = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))

    return f"Predicted digit: {prediction} | Confidence: {confidence:.2%}", labels


demo = gr.Interface(
    fn=classify_digit,
    inputs=gr.Image(
        sources=["upload", "clipboard"],
        image_mode="L",
        type="numpy",
        label="Upload a handwritten digit image",
    ),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Label(label="Class probabilities"),
    ],
    title="MNIST ANN Digit Classifier",
    description="Upload a black-and-white handwritten digit image. The trained ANN predicts digits from 0 to 9.",
)


if __name__ == "__main__":
    demo.launch()
