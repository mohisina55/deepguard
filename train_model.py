import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from prepare_dataset import get_dataloaders

def train_model(data_dir, epochs=5, lr=0.0001, batch_size=32):
    # Load train and validation dataloaders
    train_loader, val_loader = get_dataloaders(data_dir, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use pretrained ResNet18 and replace final layer for binary classification
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.2f}%")

    # Save model
    torch.save(model.state_dict(), "deepfake_model.pth")
    print("✅ Model saved as deepfake_model.pth")

if __name__ == "__main__":
    # 👇 CHANGE THIS to your processed directory
    train_model("processed_data", epochs=10)
