import os
import torch
import logging
import argparse
import torch.nn as nn
import torch.optim as optim

from logs.logging_config import setup_logging
from utils.dataset import CustomDataset  # 你需要自己实现 dataset.py
from models.mymodel import mymodel  # 假设你的模型类名是 MyModel 而不是 model
from models.param import Config  # 你需要创建 param.py 文件

# 训练与验证的合并函数
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, model_name, start_epoch=0, patience=5):
    os.makedirs("results/dict", exist_ok=True)  # 创建模型保存目录
    best_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()  # 进入训练模式
        running_loss = 0.0

        # 训练过程
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()  # 清空梯度
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, masks)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                logging.info(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # 每20个 epoch 保存一次模型
        if (epoch + 1) % 20 == 0:
            model_filename = os.path.join("results/dict", f"{model_name}_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_filename)
            logging.info(f"第 {epoch+1} 个 epoch 模型已保存: {model_filename}")
        
        # 验证过程
        model.eval()  # 进入评估模式
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():  # 不计算梯度
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += masks.size(0)
                correct += (predicted == masks).sum().item()

        val_loss = total_loss / len(val_loader)
        val_acc = correct / total
        logging.info(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        # 如果当前验证损失低于历史最佳，则保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_filename = os.path.join("results/dict", f"{model_name}_best_model.pth")
            torch.save(model.state_dict(), best_model_filename)
            logging.info(f"Best model saved with loss: {best_loss:.4f} as {best_model_filename}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # 早停机制
        if early_stop_counter >= patience:
            logging.info(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
            break

# 恢复训练
def resume_training(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    model.to(device)
    return model, optimizer, start_epoch, best_loss

# 主函数
def main():
    model = mymodel()
    model_name = model.__class__.__name__
    log_filename = os.path.join(model_name, "train_validate")
    setup_logging(log_filename)  # 设置日志

    # 解析命令行参数
    config = Config()
    args = config.__dict__
    logging.info(f"训练参数: {args}")
    
    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")

    # 数据集 & 数据加载器
    # sub_dir = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
    # root_dir = f"data/MVTecAD/{sub_dir}"
    root_dir = "data/NEU"
    train_dataset = CustomDataset(root_dir, flag='train')  # 你需要自己实现 dataset.py
    train_loader = CustomDataset.create_dataloaders(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = CustomDataset(root_dir, flag='val')
    val_loader = CustomDataset.create_dataloaders(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 模型
    model = mymodel()  # 你需要自己实现 model.py
    logging.info(f"模型结构: {model}")

    # 损失函数 & 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    checkpoint_path = os.path.join("results/dict", f"{model_name}_epoch_{args.get('resume_epoch', 0)}.pth")
    
    if os.path.exists(checkpoint_path):
        logging.info(f"恢复训练，从 {checkpoint_path} 加载模型")
        model, optimizer, start_epoch, best_loss = resume_training(model, optimizer, checkpoint_path, device)
    else:
        start_epoch = 0
        best_loss = float('inf')

    # 训练与验证
    train_and_validate(model, train_loader, val_loader, criterion, optimizer, device, args.epochs, model_name, start_epoch)

if __name__ == "__main__":
    main()