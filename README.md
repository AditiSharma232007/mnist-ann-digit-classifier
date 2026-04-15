# MNIST ANN Digit Classifier

This project builds an Artificial Neural Network for the MNIST handwritten digit dataset, following the requirements from the notebook image.

## Requirements Covered

- Dataset: MNIST handwritten digits
- Image size: `28 x 28` black-and-white images
- Training samples used: `50,000`
- Testing samples used: `10,000`
- Model type: ANN / MLP
- Architecture:
  - Input layer: `784` features
  - Hidden layer 1: `64` units with LeakyReLU
  - Hidden layer 2: `64` units with LeakyReLU
  - Hidden layer 3: `32` units with LeakyReLU
  - Output layer: `10` classes
- Batch size: `32`
- Evaluation:
  - Loss
  - Train accuracy
  - Test accuracy
  - Precision
  - Recall
  - F1 score
  - Confusion matrix
- Plots:
  - Training accuracy/loss graph
  - Confusion matrix graph
- Deployment:
  - Gradio app, suitable for Hugging Face Spaces or other cloud platforms

## Folder Structure

```text
.
|-- app/
|   |-- __init__.py
|   `-- app.py
|-- artifacts/
|   `-- plots/
|-- data/
|-- models/
|-- notebooks/
|   `-- README.md
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data_loader.py
|   |-- evaluate.py
|   |-- model.py
|   |-- predict.py
|   `-- train.py
|-- app.py
|-- requirements.txt
`-- README.md
```

Generated after training:

```text
models/mnist_ann.keras
artifacts/metrics.json
artifacts/plots/training_history.png
artifacts/plots/confusion_matrix.png
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train And Evaluate

```bash
python -m src.train
```

This command will:

- Load MNIST
- Train the ANN model
- Print all model parameters
- Save the trained model to `models/mnist_ann.keras`
- Save metrics to `artifacts/metrics.json`
- Save graphs to `artifacts/plots/`

## Run Prediction From Python

```bash
python -m src.predict path_to_digit_image.png
```

## Run The Web App

Train the model first, then run:

```bash
python app.py
```

Open the local URL shown in the terminal.

For cloud deployment, upload this project to Hugging Face Spaces or another Gradio-compatible cloud platform. Make sure the trained file `models/mnist_ann.keras` is included after running `python -m src.train`.
