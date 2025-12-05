import torch
from torch import nn, optim
from tqdm import tqdm

from dataset import get_loaders
from model import build_model

DATA_DIR = "../data/garbage_dataset"
SAVE_PATH = "../models/resnet50_binary.pth"

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using:", device)

    train_loader, val_loader = get_loaders(DATA_DIR)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0

        for imgs, labels in tqdm(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(out, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                _, preds = torch.max(out, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print("Saved:", SAVE_PATH)

if __name__ == "__main__":
    train()