
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "PlantVillage"

BATCH_SIZE = 32
SEED = 42

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def create_dataloaders():
    # Separate dataset objects let each split use its own transform.
    train_source = ImageFolder(
        root=str(DATASET_PATH),
        transform=train_transform,
    )

    eval_source = ImageFolder(
        root=str(DATASET_PATH),
        transform=eval_transform,
    )

    if train_source.classes != eval_source.classes:
        raise RuntimeError("Training and evaluation class mappings differ.")

    total_size = len(train_source)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(total_size, generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_dataset = Subset(train_source, train_indices)
    val_dataset = Subset(eval_source, val_indices)
    test_dataset = Subset(eval_source, test_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    return train_loader, val_loader, test_loader, train_source.classes


if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = create_dataloaders()

    print("Number of classes:", len(classes))
    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))
    print("Test images:", len(test_loader.dataset))

    images, labels = next(iter(train_loader))
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
