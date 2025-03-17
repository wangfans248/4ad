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
model = mymodel()
model_name = model.__class__.__name__
logname = 'train'
log_filename = os.path.join(model_name, logname)

# 训练函数
train_loader = CustomDataset.create_dataloaders(batch_size=32, num_workers=4) 

 # 你需要自己实现 model.py
def train(model, train_loader, criterion, optimizer, device, num_epochs):
    os.makedirs("results/dict", exist_ok=True)  # 创建模型保存目录
    model.to(device)

    for epoch in range(num_epochs):
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

        # 每个 epoch 结束后保存模型
        torch.save(model.state_dict(), os.path.join("results/dict", f"{model_name}_epoch_{epoch+1}.pth"))
        logging.info(f"模型已保存: {model_name}_epoch_{epoch+1}.pth")

# 主函数
def main():
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

    # 训练
    train(model, train_loader, criterion, optimizer, device, args.epochs)

if __name__ == "__main__":
    main()