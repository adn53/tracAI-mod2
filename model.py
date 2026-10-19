import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import wandb

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)

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
    total_train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * data.size(0)

    # Validation
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == target).sum().item()

    average_train_loss = total_train_loss / len(train_dataset)
    validation_accuracy = correct / len(test_dataset)
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": average_train_loss,
        "validation_accuracy": validation_accuracy,
    })

torch.save(model.state_dict(), "model.pt")

wandb.finish()