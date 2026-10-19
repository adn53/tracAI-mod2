# Assignment: Model Training and Experiment Tracking

## Overview
In this assignment, you will train an MNIST classifier in PyTorch and monitor the training with Weights & Biases (W&B).

## Dataset
We'll use the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits (0-9).

## Tasks

### 1. Model Training with PyTorch (20 Points)
Use the following skeleton code to create your PyTorch model:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import wandb

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        # FIXME: Implement the model architecture
        # Hint: Consider adding hidden layers and dropout

    def forward(self, x):
        # FIXME: Implement the forward pass
        pass

# Hyperparameters to experiment with
config = {
    "learning_rate": 0.01,
    "epochs": 10,
    "batch_size": 64,
    "hidden_layers": [128, 64],  # Example: two hidden layers
    "dropout_rate": 0.2
}

# Initialize wandb
wandb.init(project="mnist-mlops", config=config)
config = wandb.config  # use this so values are read as config.batch_size, etc.

# Load and preprocess data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

# Initialize model, loss, and optimizer
model = LogisticRegression(input_dim=28*28, output_dim=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# Training loop
for epoch in range(config.epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # FIXME: Implement the training step
        # Hint: Remember to log the loss with wandb

    # Validation
    model.eval()
    correct = 0
    with torch.no_grad():
        # FIXME: Implement the validation step
        # Hint: Remember to log the accuracy with wandb

wandb.finish()
```

### 2. Model Monitoring and Logging with Weights and Biases (25 Points)
Extend your training code to include more detailed logging:

```python
# Add this after model initialization
wandb.watch(model, log="all")

# In the training loop, add more detailed logging
# FIXME: Log additional metrics (e.g., learning rate, gradient norms)

# After validation, log more metrics
# FIXME: Log additional validation metrics (e.g., precision, recall)

# Visualize sample predictions
def log_predictions(model, data, target, num_samples=10):
    # FIXME: Implement this function to log prediction samples
    pass

# Call this function at the end of each epoch or at the end of training
log_predictions(model, next(iter(test_loader))[0][:10], next(iter(test_loader))[1][:10])
```

Ensure you generate an automated report in wandb with these visualizations.

## Submission
Submit a GitHub repository containing:

- Your training code (train.py) and a requirements.txt.
- Your trained model weights, saved at the end of training with torch.save(model.state_dict(), "model.pt").
- A README.md with a project overview, instructions for running the code locally, and the URL of your wandb report.
- Keep this repository: assignment 2 builds on it.

## Additional Notes
- Experiment with different model architectures and hyperparameters.
- Add detailed logging and custom visualizations in Weights & Biases.
