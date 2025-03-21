import os
import torch
import logging
import argparse
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import numpy as np


from utils.dataset import CustomDataset  # 你需要自己实现 dataset.py
from models.mymodel import mymodel  # 假设你的模型类名是 MyModel
from models.param import Config  # 你需要创建 param.py 文件
from logs.logging_config import setup_logging
from models.mymodel import visualize_feature_maps_with_hooks
# 设定日志格式
logger = logging.getLogger(__name__)

# 单张图片推理函数
def inference(model, image_path, device):
    """
    进行单张图片的推理，并返回预测结果。
    """
    try:
        model.eval()  # 进入评估模式
        image = Image.open(image_path).convert('RGB')  # 打开图片并转为RGB格式
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 假设模型输入大小为224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])
        image = transform(image).unsqueeze(0).to(device)  # 添加 batch 维度

        with torch.no_grad():  # 禁用梯度计算
            outputs = model(image)  # 前向传播
            # 处理二分类输出，使用sigmoid并与0.5进行阈值比较
            predicted_mask = (torch.sigmoid(outputs) > 0.9).long().squeeze(0).squeeze(0).cpu().numpy()  # 获取二值化分割结果

        return predicted_mask
    except Exception as e:
        logger.error(f"推理过程中发生错误: {e}", exc_info=True)
        return None

def visualize_segmentation(predicted_mask, save_path):
    """
    可视化分割结果，并保存图片。
    """
    plt.imshow(predicted_mask, cmap='gray')  # 显示为灰度图
    plt.title("Foreground/Background Segmentation")
    plt.axis("off")
    plt.savefig(save_path, bbox_inches='tight')  # 保存分割图
    plt.close()
    logger.info(f"分割结果已保存到: {save_path}")

# 主函数
def main():
    """
    解析命令行参数，加载模型，并进行推理。
    """
    # 设置日志
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    model = mymodel(n_channels=3, n_classes=1)  # 假设模型输出是一个单通道的分割图
    model_name = model.__class__.__name__
    log_filename = os.path.join(model_name, 'inference_log.txt')
    setup_logging(log_filename)

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='模型推理')
    parser.add_argument('--image_path', type=str, required=True, help='需要推理的图片路径')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='模型保存目录')
    parser.add_argument('--output_path', type=str, default='segmentation_output.png', help='分割结果保存路径')
    args = parser.parse_args()

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 确保模型路径存在
    checkpoint_path = os.path.join(args.save_dir, f"{model_name}_best_model.pth")
    if not os.path.exists(checkpoint_path):
        logger.error(f"模型权重文件 {checkpoint_path} 不存在！请检查路径是否正确。")
        return

    # 加载模型
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.to(device)
        logger.info(f"模型 {model_name} 加载成功！")
    except Exception as e:
        logger.error(f"加载模型失败: {e}", exc_info=True)
        return

    # 执行推理
    predicted_mask = inference(model, args.image_path, device)
    if predicted_mask is not None:
        logger.info(f"分割完成，结果已保存。")
        visualize_segmentation(predicted_mask, args.output_path)

    visualize_feature_maps_with_hooks(model, args.image_path)

if __name__ == "__main__":
    main()
