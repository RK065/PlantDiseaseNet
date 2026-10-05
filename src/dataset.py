import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def create_dataloaders():
    dataset = ImageFolder(
    root="data/PlantVillage",
    transform=train_transform
    )
    
    total_size = len(dataset)
    
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    torch.manual_seed(42)
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        dataset.classes
    )

if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = create_dataloaders()

    print("Number of classes:", len(classes))
    print("Classes:", classes)
    
    images, labels = next(iter(train_loader))
    
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
