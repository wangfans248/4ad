from .modelinstance.UNet import UNet
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy as np

def mymodel(n_channels, n_classes):
    return UNet(n_channels, n_classes) 

def visualize_feature_maps_with_hooks(model, args_image):
    activations_maps = {}  # 存储激活图
    image = Image.open(args_image).convert('RGB')  # 打开图片并转为RGB格式
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 获取模型所在设备
    transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 假设模型输入大小为224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])
    image = transform(image).unsqueeze(0).to(device)

    # 2. 定义 forward hook
    def hook_fn(layer_name, module, input, output):
        activations_maps[layer_name] = output.detach().cpu()
    # 3. 选择要 hook 的层
    layers_to_hook = [
        ('inc', model.inc),
        ('down1', model.down1),
        ('down2', model.down2),
        ('down3', model.down3),
        ('down4', model.down4),
        ('up1', model.up1),
        ('up2', model.up2),
        ('up3', model.up3),
        ('up4', model.up4),
        ('outc', model.outc)
    ]

    # 4. 注册 hook
    hooks = [layer.register_forward_hook(lambda module, input, output, layer_name=layer_name: hook_fn(layer_name, module, input, output)) for layer_name, layer in layers_to_hook]

    # 5. 运行前向传播
    model.eval()  # 进入评估模式
    with torch.no_grad():  # 关闭梯度计算 # 获取模型所在设备
        input_image = image  # 确保输入数据在正确的设备上
        _ = model(input_image)  # 触发 forward hooks

    # 6. 取消所有 hook
    for hook in hooks:
        hook.remove()

    # 7. 可视化特征图
    for layer_name, feature_map in activations_maps.items():
        feature_map = feature_map.numpy()[0]  # 取 batch=0 的样本
        num_channels = feature_map.shape[0]  # 通道数

        fig, axes = plt.subplots(1, min(num_channels, 8), figsize=(20, 5), squeeze=False)
        axes = axes.flatten()  # ✅ 确保 axes 是一维数组

        fig.suptitle(f"Layer: {layer_name}")
        
        for j in range(min(num_channels, 8)):  
            ax = axes[j]  # ✅ axes 现在可以安全索引
            fm = feature_map[j]
            fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-5)
            ax.imshow(fm, cmap="gray")
            ax.axis("off")

        plt.savefig(f"{layer_name}.png")  # ✅ 保存图像
        plt.close()  # ✅ 关闭图像，防止内存泄漏


    print("Feature maps saved successfully!")