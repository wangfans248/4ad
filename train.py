import os
import torch
import logging
import argparse
from datetime import datetime
import torch.nn as nn
import torch.optim as optim

# 自定义模块（你需要自己实现 dataset.py 和 model.py）
from logs.logging_config import setup_logging   # 你需要创建 logging_config.py 文件
from utils.dataset import CustomDataset  # 你需要创建 dataset.py 文件
from models.mymodel import mymodel   # 你需要创建 model.py 文件
from models.param import Config  # 你需要创建 param.py 文件
# 模型名称和日志名称和设置日志\


# 训练函数
save_dir = "results/dict"
 # 你需要自己实现 model.py
def train(model, train_loader, criterion, optimizer, device, num_epochs, model_name, start_epoch=0):
    os.makedirs(save_dir, exist_ok=True)  # 创建模型保存目录
    model.to(device)

    for epoch in range(start_epoch, num_epochs):
        model.train()  # 进入训练模式
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # 清空梯度
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                logging.info(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        if (epoch + 1) % 20 == 0:
            model_filename = os.path.join(save_dir, f"{model_name}_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_filename)
            logging.info(f"第 {epoch+1} 个 epoch 模型已保存: {model_filename}")
        
        if running_loss < best_loss:
            best_loss = running_loss
            best_model_filename = os.path.join("results/dict", f"{model_name}_best_model.pth")
            torch.save(model.state_dict(), best_model_filename)
            logging.info(f"Best model saved with loss: {best_loss:.4f} as {best_model_filename}")

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
    logname = 'train'
    log_filename = os.path.join(model_name, logname)
    setup_logging(log_filename)  # 设置日志

    # 解析命令行参数
    config = Config()
    args = config.__dict__
    logging.info(f"训练参数: {args}")
    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")

    # 数据集 & 数据加载器
    train_dataset = CustomDataset(flag='train')  # 你需要自己实现 dataset.py
    train_loader = CustomDataset.create_dataloaders(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 模型
    model = mymodel()  # 你需要自己实现 model.py
    logging.info(f"模型结构: {model}")

    # 损失函数 & 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    checkpoint_path = os.path.join(save_dir, f"{model_name}_epoch_{args['resume_epoch']}.pth")
    
    if os.path.exists(checkpoint_path):
        logging.info(f"恢复训练，从 {checkpoint_path} 加载模型")
        model, optimizer, start_epoch, best_loss = resume_training(model, optimizer, checkpoint_path, device)
    else:
        start_epoch = 0
        best_loss = float('inf')

    # 训练
    train(model, train_loader, criterion, optimizer, device, args.epochs, model_name, start_epoch)

if __name__ == "__main__":
    main()