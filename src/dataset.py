import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Mapping for the new dataset
RECYCLABLE = {"cardboard", "glass", "metal", "paper", "plastic"}

def is_recyclable(class_name):
    return 1 if class_name in RECYCLABLE else 0

def get_loaders(data_dir, batch_size=32, split=0.8):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)

    # Make a binary label list based on folder names
    binary_targets = []
    inv_map = {v: k for k, v in dataset.class_to_idx.items()}
    for idx in dataset.targets:
        class_name = inv_map[idx]
        binary_targets.append(is_recyclable(class_name))

    # Replace original labels with binary labels
    dataset.targets = binary_targets

    # Train/val split
    total = len(dataset)
    train_size = int(total * split)
    val_size = total - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader