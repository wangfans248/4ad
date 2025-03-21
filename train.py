import os
import torch
import logging
import argparse
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import f1_score

from logs.logging_config import setup_logging
from utils.dataset import CustomDataset  # 你需要自己实现 dataset.py
from models.mymodel import mymodel  # 假设你的模型类名是 MyModel 而不是 model
from models.param import Config 
from utils.loss import CombinedLoss # 你需要创建 param.py 文件

# 训练与验证的合并函数
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, model_name, start_epoch=0, patience=100):
    os.makedirs("results/dict", exist_ok=True)  # 创建模型保存目录
    best_loss = float('inf')
    best_iou = 0.0
    early_stop_counter = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()  # 进入训练模式
        running_loss = 0.0

        # 训练过程
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            masks = masks.squeeze(1).float()  # 确保 masks 形状正确，并转换为 float

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
            torch.save({'model_state_dict': model.state_dict(),
                        'epoch': epoch,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_loss': best_loss
                        }, model_filename)
            logging.info(f"第 {epoch+1} 个 epoch 模型已保存: {model_filename}")
        
        # 验证过程
        model.eval()  # 进入评估模式
        total_loss = 0.0
        iou_sum = 0.0
        num_batches = 0

        best_f1 = 0.0
        best_threshold = 0.5  # 初始值

        with torch.no_grad():  # 不计算梯度
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                masks = masks.squeeze(1).float()  # 确保 masks 形状正确，并转换为 float

                outputs = model(images)  # (B,1,H,W) logits
                loss = criterion(outputs, masks)  # 计算 loss
                total_loss += loss.item()

        # 使用不同阈值计算 F1 分数
                for threshold in [x * 0.1 for x in range(1, 10)]:
                    predicted_binary = (torch.sigmoid(outputs) > threshold).long()
                    f1 = f1_score(masks.long().cpu().numpy().flatten(), predicted_binary.cpu().numpy().flatten())
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold

        # 使用最佳阈值进行二值化
                predicted_binary = (torch.sigmoid(outputs) > best_threshold).float()

        # 计算 IoU
            intersection = (predicted_binary * masks).sum(dim=(1, 2))  # 交集
            union = ((predicted_binary + masks).clamp(max=1.0)).sum(dim=(1, 2))  # 并集

            iou = (intersection + 1e-6) / (union + 1e-6)  # 计算 IoU
            iou = iou.mean().item()

        # 累加 IoU
            iou_sum += iou
            num_batches += 1

# 计算当前 epoch 的平均 IoU
        val_loss = total_loss / len(val_loader)
        val_iou = iou_sum / num_batches  # 计算整个验证集的平均 IoU
        logging.info(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Validation IoU: {val_iou:.4f}, Best Threshold: {best_threshold:.2f}")
# 早停判断
        if val_iou > best_iou and val_loss < best_loss:
            best_loss = val_loss
            best_iou = val_iou
            best_model_filename = os.path.join("results/dict", f"{model_name}_best_model.pth")
            torch.save({'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
                }, best_model_filename)
            logging.info(f"Best model saved with loss: {best_loss:.4f} and iou: {best_iou:.4f} as {best_model_filename}")
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
    logging.getLogger("PIL").setLevel(logging.WARNING)
    model = mymodel(n_channels=3, n_classes=1) # 你需要自己实现 model.py
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
    model.to(device)

    # 数据集 & 数据加载器
    # sub_dir = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
    # root_dir = f"data/MVTecAD/{sub_dir}"
    root_dir = "data/NEU"
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    train_dataset = CustomDataset(root_dir, transform, flag='train')  # 你需要自己实现 dataset.py
    train_loader = CustomDataset.create_dataloaders(train_dataset, batch_size=args['batch_size'], num_workers=4)
    val_dataset = CustomDataset(root_dir, transform, flag='val')
    val_loader = CustomDataset.create_dataloaders(val_dataset, batch_size=args['batch_size'], num_workers=4)

    # 模型
    logging.info(f"模型结构: {model}")

    # 损失函数 & 优化器
    criterion = CombinedLoss(alpha=args['alpha'])
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    checkpoint_path = os.path.join("results/dict", f"{model_name}_epoch_{args.get('resume_epoch', 0)}.pth")
    
    if os.path.exists(checkpoint_path):
        logging.info(f"恢复训练，从 {checkpoint_path} 加载模型")
        model, optimizer, start_epoch, best_loss = resume_training(model, optimizer, checkpoint_path, device)
    else:
        start_epoch = 0
        best_loss = float('inf')

    # 训练与验证
    train_and_validate(model, train_loader, val_loader, criterion, optimizer, device, args['epochs'], model_name, start_epoch)

if __name__ == "__main__":
    main()