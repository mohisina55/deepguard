import torch
from prepare_dataset import get_dataloaders
from torchvision import models
from sklearn.metrics import classification_report
import numpy as np

def evaluate_model(data_dir, model_path, batch_size=32):
    # Load validation dataloader
    _, val_loader = get_dataloaders(data_dir, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model architecture and weights
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=["Fake", "Real"]))

if __name__ == "__main__":
    evaluate_model("processed_data", "deepfake_model.pth", batch_size=32)