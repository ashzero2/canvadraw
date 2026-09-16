# CanvaDraw

CanvaDraw is a small neural-network experiment that predicts the next canvas UI frame from
the current frame and a click position. It uses a classifier and a FiLM-conditioned
encoder-decoder trained from scratch with PyTorch. The demo is served with FastAPI and an
HTML Canvas frontend.

## Requirements

- Python 3.12
- Apple Silicon is recommended for MPS acceleration; CPU is supported

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Generate data and train

```bash
python dataset/generate.py
python train_classifier.py
python train.py
```

Training writes `classifier.pth` and `canvadraw.pth` in the repository root. Model weights,
generated dataset images, and inference reports are ignored by Git.

## Run the demo

```bash
source .venv/bin/activate
uvicorn server:app --reload
```

Open <http://127.0.0.1:8000>. Enter a username and password in the canvas and select Submit.
The browser sends the rendered canvas and click coordinates over a WebSocket. The server
classifies the frame, conditions the image generator on the predicted class, and streams two
decoder activation maps followed by the final frame.

## Validate

```bash
ruff format --check .
ruff check .
python -m unittest discover -v
python predict.py
```

`python test_model.py` additionally creates a visual generalization report at
`test_results.png`.

## Current scope

The synthetic dataset contains a single login interaction and a small fixed vocabulary of
valid and invalid usernames. This is a proof of concept rather than a general-purpose UI
prediction model. Password semantics are not represented in the current training data.