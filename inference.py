import os
import torch
import logging
import argparse
from PIL import Image
from torchvision import transforms
from utils.dataset import CustomDataset  # 你需要自己实现 dataset.py
from models.mymodel import mymodel  # 假设你的模型类名是 MyModel 而不是 model
from models.param import Config  # 你需要创建 param.py 文件
from logs.logging_config import setup_logging
import matplotlib.pyplot as plt
# 单张图片推理函数
def inference(model, image_path, device):
    model.eval()  # 进入评估模式
    image = Image.open(image_path).convert('RGB')  # 打开图片并转为RGB格式
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 假设模型输入大小为224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 常见的预处理方式
    ])
    image = transform(image).unsqueeze(0).to(device)  # 添加batch维度并移动到设备

    with torch.no_grad():  # 禁用梯度计算
        outputs = model(image)  # 前向传播
        _, predicted = torch.max(outputs, 1)  # 获取预测结果
        return predicted.item()

def visualize_segmentation(predicted, save_path):
    # 假设 predicted 是二值化后的前景背景分割图，形状为 [H, W]
    plt.imshow(predicted, cmap='gray')  # 显示为灰度图
    plt.title("Foreground/Background Segmentation")
    plt.savefig(save_path)  # 保存分割图
    plt.close()

# 主函数
def main():
    # 设置日志
    model = mymodel(n_channels=3, n_classes=1)
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
    logging.info(f"使用设备: {device}")

    # 模型加载
    checkpoint = torch.load(os.path.join(args.save_dir, f"{model_name}_best_model.pth"))
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # strict=False 忽略不匹配的键
  # 加载训练好的模型
    model.to(device)
    logging.info(f"模型结构: {model}")
    # 执行推理
    predicted_class, outputs = inference(model, args.image_path, device)
    logging.info(f"预测结果: 类别 {predicted_class}")

    # 假设模型的输出是前景背景二值化的分割图
    _, predicted = torch.max(outputs, 1)  # 获取分割结果，0为背景，1为前景
    predicted = predicted.squeeze(0).cpu().numpy()  # 转为 NumPy 数组，形状为 [H, W]

    # 可视化并保存分割结果
    visualize_segmentation(predicted, args.output_path)
    logging.info(f"分割结果已保存到: {args.output_path}")

if __name__ == "__main__":
    main()
