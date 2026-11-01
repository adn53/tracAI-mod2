import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import wandb

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, dropout_rate):
        super(LogisticRegression, self).__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)


def log_predictions(model, data, target, num_samples=10):
    model.eval()
    data = data[:num_samples]
    target = target[:num_samples]
    with torch.no_grad():
        predictions = model(data).argmax(dim=1)

    table = wandb.Table(columns=["image", "actual", "predicted"])
    for image, actual, predicted in zip(data, target, predictions):
        display_image = (image.cpu() * 0.3081 + 0.1307).clamp(0, 1)
        table.add_data(
            wandb.Image(display_image, caption=f"Predicted: {predicted.item()}"),
            actual.item(),
            predicted.item(),
        )
    wandb.log({"sample_predictions": table})


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
model = LogisticRegression(
    input_dim=28 * 28,
    output_dim=10,
    hidden_layers=config.hidden_layers,
    dropout_rate=config.dropout_rate,
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
wandb.watch(model, log="all")
prediction_data, prediction_targets = next(iter(test_loader))

# Training loop
for epoch in range(config.epochs):
    model.train()
    total_train_loss = 0
    total_gradient_norm = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        gradient_norm = torch.sqrt(sum(
            parameter.grad.detach().pow(2).sum()
            for parameter in model.parameters()
            if parameter.grad is not None
        ))
        total_gradient_norm += gradient_norm.item()
        optimizer.step()

        total_train_loss += loss.item() * data.size(0)

    # Validation
    model.eval()
    confusion = torch.zeros(10, 10, dtype=torch.int64)
    validation_targets = []
    validation_predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            predictions = outputs.argmax(dim=1)
            validation_targets.extend(target.tolist())
            validation_predictions.extend(predictions.tolist())
            confusion += torch.bincount(
                target * 10 + predictions, minlength=100
            ).reshape(10, 10)

    average_train_loss = total_train_loss / len(train_dataset)
    validation_accuracy = confusion.diag().sum().item() / len(test_dataset)
    true_positives = confusion.diag().float()
    precision = (true_positives / confusion.sum(dim=0).clamp_min(1)).mean().item()
    recall = (true_positives / confusion.sum(dim=1).clamp_min(1)).mean().item()
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": average_train_loss,
        "validation_accuracy": validation_accuracy,
        "validation_precision_macro": precision,
        "validation_recall_macro": recall,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "gradient_norm": total_gradient_norm / len(train_loader),
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=validation_targets,
            preds=validation_predictions,
            class_names=[str(digit) for digit in range(10)],
        ),
    })
    log_predictions(model, prediction_data, prediction_targets)

torch.save(model.state_dict(), "model.pt")

wandb.finish()