from .modelinstance.UNet import UNet
import torch
import matplotlib.pyplot as plt


def mymodel(n_channels, n_classes):
    return UNet(n_channels, n_classes) 

def visualize_feature_maps_with_hooks(input_image):
    activations_maps = {}  # 用于存储激活图
    
    # 1. 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mymodel(n_channels=3, n_classes=1)
    model.to(device)
    model.eval()  # 进入评估模式

    # 2. 定义 forward hook
    def hook_fn(module, input, output):
        activations_maps[module] = output.detach()  # ✅ 确保 output 是 Tensor，并且不计算梯度

    # 3. 选择要 hook 的层
    layers_to_hook = [
        model.inc,
        model.down1,
        model.down2,
        model.down3,
        model.down4,
        model.up1,
        model.up2,
        model.up3,
        model.up4,
        model.outc
    ]

    # 4. 注册 hook
    hooks = [layer.register_forward_hook(hook_fn) for layer in layers_to_hook]

    # 5. 运行前向传播
    with torch.no_grad():  # 关闭梯度计算，避免额外的显存消耗
        input_image = input_image  # 确保输入在正确的设备上
        _ = model(input_image)  # 触发 forward hooks

    # 6. 取消所有 hook
    for hook in hooks:
        hook.remove()

    # 7. 可视化特征图
    for i, (layer, feature_map) in enumerate(activations_maps.items()):
        feature_map = feature_map.cpu().numpy()[0]  # 取 batch=0 的样本
        num_channels = feature_map.shape[0]  # 通道数
        fig, axes = plt.subplots(1, min(num_channels, 8), figsize=(20, 5))  # 取前 8 个通道
        fig.suptitle(f"Layer: {layer.__class__.__name__}")
        
        for j in range(min(num_channels, 8)):  # 只显示前 8 个通道
            ax = axes[j]
            fm = feature_map[j]
            fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-5)  # 归一化到 [0,1]
            ax.imshow(fm, cmap="gray")
            ax.axis("off")
        
        plt.show()

# 假设 image 是输入图像
# 调用该函数进行推理和可视化