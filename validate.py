import os
import torch
import logging
import argparse
import torch.nn as nn

from logs.logging_config import setup_logging
from utils.dataset import CustomDataset  # 你需要自己实现 dataset.py
from models.mymodel import mymodel  # 假设你的模型类名是 MyModel 而不是 model
from models.param import Config  # 你需要创建 param.py 文件


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = total_loss / len(val_loader)
    val_acc = correct / total
    logging.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
    return val_loss, val_acc

def main():
    model = mymodel()
    model_name = model.__class__.__name__
    log_filename = os.path.join(model_name, "validate")
    setup_logging(log_filename)

    config = Config()
    args = config.__dict__
    logging.info(f"Validation Parameters: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    val_dataset = CustomDataset(root_dir="data", flag="val")
    val_loader = CustomDataset.create_dataloaders(batch_size=32, num_workers=4)

    model.load_state_dict(torch.load("results/dict/mymodel_epoch_10.pth"))
    model.to(device)
    logging.info("Model loaded successfully")

    criterion = nn.CrossEntropyLoss()

    validate(model, val_loader, criterion, device)


if __name__ == "__main__":
    main()