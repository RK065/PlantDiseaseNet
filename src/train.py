
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import create_dataloaders
from src.model import PlantDiseaseCNN
from src.utils import set_seed, save_checkpoint, plot_training_history


EPOCHS = 5
LEARNING_RATE = 0.001
CHECKPOINT_PATH = Path("models/best_model.pth")


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += batch_size

    return total_loss / total, correct / total


def train_model():
    set_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)

    train_loader, val_loader, _, classes = create_dataloaders()

    model = PlantDiseaseCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    best_val_accuracy = -1.0

    for epoch in range(EPOCHS):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )

        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val accuracy: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(
                model,
                CHECKPOINT_PATH,
                classes,
                epoch + 1,
                val_accuracy,
            )
            print("Saved best validation checkpoint.")

    plot_training_history(history)
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train_model()
