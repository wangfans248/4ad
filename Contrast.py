import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import logging
from logs.logging_config import setup_logging
from torchvision import transforms
from utils.dataset import CustomDataset
from models.mymodel import mymodel
import os
import argparse



# 定义函数
def show_contrastive_loss(data_loader, model, save_path):
    normal_errors = []

    # 定义图像转换操作，将输入图像转为灰度图
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 转为灰度图
        transforms.ToTensor()
    ])

    # 遍历数据加载器
    for i, (input_image, _) in enumerate(data_loader):  # 假设 DataLoader 返回的是 (image, label) 对
        input_image = input_image.to("cuda")  # 转换为灰度图
        
        # 通过模型进行图像重建
        reconstructed_image = model(input_image)
        input_image = transform(input_image)
        # 计算均方误差（MSE）
        error = F.mse_loss(reconstructed_image, input_image).item()

        # 存储误差
        normal_errors.append(error)
        print(".", end="")

        # 打印进度
        if i % 100 == 0:  # 每 100 次输出一次进度
            logging.info(f"Processed {i} samples")

    # 转为 NumPy 数组
    normal_errors = np.array(normal_errors)

    # 绘制误差的直方图
    plt.hist(normal_errors, bins=50, alpha=0.7)
    plt.title("Reconstruction Error Distribution (Normal Images)")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")

    # 保存图像到指定路径
    plt.savefig(save_path)  # 保存图像
    logging.info(f"Histogram saved to {save_path}")

    # 显示图像
    plt.show()

    # 计算均值和标准差
    mean_error = np.mean(normal_errors)
    std_error = np.std(normal_errors)

    # 输出均值和标准差
    logging.info(f"Mean Error: {mean_error}")
    logging.info(f"Standard Deviation: {std_error}")
    print(f"Mean Error: {mean_error}")
    print(f"Standard Deviation: {std_error}")


def main():
     # 设置日志
    logging.getLogger("PIL").setLevel(logging.WARNING)
    model = mymodel(n_channels=3, n_classes=1)  # 假设模型输出是一个单通道的分割图
    model_name = model.__class__.__name__
    log_filename = os.path.join(model_name, 'contrast_loss_log.txt')
    setup_logging(log_filename)

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='对比损失计算')
    parser.add_argument('--data_path', type=str, required=True, help='数据集路径')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='模型保存目录')
    parser.add_argument('--result_dir', type=str, default='results/contrastive_loss', help='结果保存路径')
    args = parser.parse_args()

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")

    # 确保模型路径存在
    checkpoint_path = os.path.join(args.save_dir, f"{model_name}_best_model.pth")
    if not os.path.exists(checkpoint_path):
        logging.error(f"模型权重文件 {checkpoint_path} 不存在！请检查路径是否正确。")
        return

    # 加载模型
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.to(device)
        logging.info(f"模型 {model_name} 加载成功！")
    except Exception as e:
        logging.error(f"加载模型失败: {e}", exc_info=True)
        return

    # 假设你有一个加载数据集的函数
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    data_set = CustomDataset(args.data_path, transform, flag='train')
    data_loader = CustomDataset.create_dataloaders(data_set, batch_size=32, num_workers=4)  # 自定义数据加载器

    # 执行对比损失计算
    show_contrastive_loss(data_loader, model, save_path=args.result_dir)

if __name__ == "__main__":
    main()
   