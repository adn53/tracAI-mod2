# MNIST Classifier

A PyTorch model that classifies handwritten digits from the MNIST dataset. Training metrics and sample predictions are tracked with Weights & Biases.

## Run locally

```bash
pip install -r requirements.txt
wandb login
python train.py
```

The script saves the trained weights to `model.pt`.

## W&B report

https://wandb.ai/adityananjappa11-iowa-state-university/mnist-mlops/runs/m4q4yoc6
