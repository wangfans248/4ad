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
import seaborn as sns



# 定义函数
def show_contrastive_loss(data_loader, model, save_path):
    torch.cuda.empty_cache()
    # normal_errors = []
    anomaly_errors = []


    # 定义图像转换操作，将输入图像转为灰度图
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 转为灰度图
    ])

    # 遍历数据加载器
    for i, input_image in enumerate(data_loader):  # 假设 DataLoader 返回的是 (image, label) 对
        input_image = input_image.to("cuda") 
        
        # 通过模型进行图像重建
        with torch.no_grad():
            reconstructed_image = model(input_image)
        input_image = transform(input_image)
        
        # 计算均方误差（MSE）
        error = F.mse_loss(reconstructed_image, input_image).item()

        # 存储误差
        anomaly_errors.append(error)
        # normal_errors.append(error)
        print(".", end="")

        # 打印进度
        if i % 10 == 0:  # 每 100 次输出一次进度
            logging.info(f"Processed {i} samples")

    # 转为 NumPy 数组
    # normal_errors = np.array(normal_errors)
    anomaly_errors = np.array(anomaly_errors)

    # # 绘制误差的直方图
    # mean_error = np.mean(normal_errors)
    # std_error = np.std(normal_errors)

    # # **归一化 X 轴**（标准化误差）
    # normalized_errors = (normal_errors - mean_error) / std_error

    # # **绘制 KDE 曲线**
    # plt.figure(figsize=(8, 6))
    # sns.kdeplot(normalized_errors, bw_adjust=0.5, fill=True, color="blue", alpha=0.6, label="KDE Curve")
    
    # plt.title("Normalized Reconstruction Error Distribution")
    # plt.xlabel("Deviation from Mean (Standardized)")
    # plt.ylabel("Density")
    # plt.legend()

    # # 保存图像
    # plt.savefig(save_path)  
    # logging.info(f"KDE Plot saved to {save_path}")

    # # 显示图像
    # plt.show()

    # # 输出均值和标准差
    # logging.info(f"Mean Error: {mean_error}")
    # logging.info(f"Standard Deviation: {std_error}")
    # print(f"Mean Error: {mean_error}")
    # print(f"Standard Deviation: {std_error}")

    mean_normal = 83.95
    std_normal = 4.25

# 计算异常数据的标准化偏离程度（以标准差为单位）
    z_scores = [(x - mean_normal) / std_normal for x in anomaly_errors]

# 直方图绘制
    plt.figure(figsize=(8, 5))
    plt.hist(z_scores, bins=np.arange(-4, 5, 1), edgecolor='black', alpha=0.7)

# 设置坐标轴标签
    plt.xlabel("Deviation from Normal Mean (in std units)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Anomalous Image Mean Error")

# 设置X轴刻度，以标准差的整数倍为单位
    plt.xticks(np.arange(-4, 5, 1), labels=[f"{i}" for i in range(-4, 5)])

# 显示网格
    plt.grid(axis='y', linestyle='dashed', alpha=0.7)
    plt.savefig(save_path)  
    logging.info(f"Anomaly Plot saved to {save_path}")
# 显示图像
    plt.show()
def main():
     # 设置日志
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
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
    data_set = CustomDataset(args.data_path, transform, flag='test')
    logging.info(f"数据集加载成功！共有 {len(data_set)} 张图像。")
    data_loader = CustomDataset.create_dataloaders(data_set, batch_size=32, num_workers=0)  # 自定义数据加载器
    logging.info(f"data_loader 加载成功!{len(data_loader)}")  # 查看 data_loader 总共有多少个 batch

    # 执行对比损失计算
    show_contrastive_loss(data_loader, model, save_path=args.result_dir)

if __name__ == "__main__":
    main()
   