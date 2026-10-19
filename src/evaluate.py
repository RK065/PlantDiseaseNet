from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.dataset import create_dataloaders
from src.model import PlantDiseaseCNN
from src.utils import load_checkpoint


CHECKPOINT_PATH = Path("models/best_model.pth")
REPORT_PATH = Path("reports/week3_classification_report.txt")
MATRIX_PATH = Path("reports/week3_confusion_matrix.png")


def evaluate_model():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    _, _, test_loader, classes = create_dataloaders()

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}. "
            "Run python -m src.train first."
        )

    model = PlantDiseaseCNN(num_classes=len(classes))
    model, checkpoint = load_checkpoint(
        model, CHECKPOINT_PATH, device
    )

    if checkpoint["classes"] != classes:
        raise ValueError(
            "Checkpoint class order differs from the dataset class order."
        )

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu()

            y_true.extend(labels.tolist())
            y_pred.extend(predictions.tolist())

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0,
        digits=4,
    )

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"Best validation accuracy: {checkpoint['val_accuracy']:.4f}")
    print("\nClassification report:\n")
    print(report)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        f"Test accuracy: {accuracy:.4f}\n"
        f"Best epoch: {checkpoint['epoch']}\n"
        f"Best validation accuracy: "
        f"{checkpoint['val_accuracy']:.4f}\n\n"
        f"{report}",
        encoding="utf-8",
    )

    matrix = confusion_matrix(
        y_true, y_pred, labels=list(range(len(classes)))
    )
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=classes,
    )

    fig, ax = plt.subplots(figsize=(15, 15))
    display.plot(
        ax=ax,
        xticks_rotation=90,
        include_values=False,
        cmap="Blues",
        colorbar=False,
    )
    ax.set_title("PlantDiseaseNet — Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(MATRIX_PATH, dpi=150)
    plt.close(fig)

    print(f"Report saved to {REPORT_PATH}")
    print(f"Confusion matrix saved to {MATRIX_PATH}")


if __name__ == "__main__":
    evaluate_model()
