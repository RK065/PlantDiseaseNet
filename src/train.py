
from pathlib import Path
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import create_dataloaders
from src.model import PlantDiseaseModel
from src.utils import set_seed, plot_training_history


EPOCHS = 10
LEARNING_RATE = 0.001
CHECKPOINT_DIR = Path("models")


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None

    if training:
        model.train()
        # Keep the frozen backbone in evaluation mode.
        model.model.features.eval()
    else:
        model.eval()

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


def train_model(architecture):
    set_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)
    print("Architecture:", architecture)

    train_loader, val_loader, _, classes = create_dataloaders()

    model = PlantDiseaseModel(
        architecture=architecture,
        num_classes=len(classes)
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    # Train only parameters that are not frozen.
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    best_val_accuracy = -1.0

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = CHECKPOINT_DIR / f"{architecture}_best.pth"

    start_time = time.time()

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

            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "architecture": architecture,
                "epoch": epoch + 1,
                "val_accuracy": val_accuracy
            }, checkpoint_path)

            print("Saved best checkpoint:", checkpoint_path)

    training_time = time.time() - start_time

    plot_training_history(history)

    print("\nTraining completed!")
    print("Best validation accuracy:", best_val_accuracy)
    print("Training time:", round(training_time, 2), "seconds")
    print("Checkpoint:", checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["mobilenet_v2", "efficientnet_b0"],
        default="mobilenet_v2"
    )

    args = parser.parse_args()

    train_model(args.model)


