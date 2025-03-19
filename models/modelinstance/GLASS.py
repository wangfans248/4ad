import torch


def initialize_weights(module):
    """初始化不同类型层的权重。"""
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_normal_(module.weight)
    elif isinstance(module, torch.nn.BatchNorm2d):
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)
    elif isinstance(module, torch.nn.Conv2d):
        module.weight.data.normal_(0.0, 0.02)


class Discriminator(torch.nn.Module):
    def __init__(self, input_dim, num_layers=2, hidden_dim=None):
        super().__init__()

        # 确定隐藏层维度
        hidden_dim = input_dim if hidden_dim is None else hidden_dim
        self.model = torch.nn.Sequential()

        # 创建判别器的顺序模型
        for layer_index in range(num_layers - 1):
            current_input_dim = input_dim if layer_index == 0 else hidden_dim
            hidden_dim = int(hidden_dim // 1.5) if hidden_dim is None else hidden_dim

            layer_block = torch.nn.Sequential(
                torch.nn.Linear(current_input_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.LeakyReLU(0.2)
            )
            self.model.add_module(f'block{layer_index + 1}', layer_block)

        # 输出层
        self.tail = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 1, bias=False),
            torch.nn.Sigmoid()
        )

        # 应用权重初始化
        self.apply(initialize_weights)

    def forward(self, x):
        """经过判别器的前向传播。"""
        x = self.model(x)
        x = self.tail(x)
        return x


class Projection(torch.nn.Module):
    def __init__(self, input_dim, output_dim=None, num_layers=1, layer_type=0):
        super(Projection, self).__init__()

        # 如果没有提供输出维度则设置为输入维度
        if output_dim is None:
            output_dim = input_dim

        self.layers = torch.nn.Sequential()
        
        for layer_index in range(num_layers):
            current_input_dim = input_dim if layer_index == 0 else output_dim

            # 添加线性层
            self.layers.add_module(f"layer{layer_index}fc", torch.nn.Linear(current_input_dim, output_dim))
            if layer_index < num_layers - 1:
                if layer_type > 1:
                    self.layers.add_module(f"{layer_index}relu", torch.nn.LeakyReLU(.2))
                    
        # 应用权重初始化
        self.apply(initialize_weights)

    def forward(self, x):
        """前向传播。"""
        x = self.layers(x)
        return x


class PatchMaker:
    def __init__(self, patch_size, top_k=0, stride=None):
        self.patch_size = patch_size
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        """将张量转换为各个patch。
        Args:
            features: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patch_size, patch_size]
        """
        padding = int((self.patch_size - 1) / 2)
        unfolder = torch.nn.Unfold(kernel_size=self.patch_size, stride=self.stride, padding=padding, dilation=1)
        unfolded_features = unfolder(features)
        
        # 计算总的patch数量
        number_of_total_patches = []
        for spatial_size in features.shape[-2:]:
            n_patches = (spatial_size + 2 * padding - 1 * (self.patch_size - 1) - 1) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        
        # 重塑特征张量
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patch_size, self.patch_size, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batch_size):
        """将得分解构为原始的形状。"""
        return x.reshape(batch_size, -1, *x.shape[1:])

    def score(self, x):
        """计算得分。"""
        x = x[:, :, 0]  # 选择第一个通道
        x = torch.max(x, dim=1).values  # 取最大值
 
        return x
    

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.ndimage as ndimage

# ================== Glass 异常合成模块 ==================
class GlassAnomalySynthesizer:
    def __init__(self, intensity=0.5):
        self.intensity = intensity
    
    def generate_anomaly(self, image):
        """生成Glass异常"""
        pass  # 你需要实现
    
    def compute_gradient(self, image):
        """计算梯度"""
        pass  # 你需要实现
    
    def truncate_anomalies(self, anomaly):
        """截断异常值"""
        pass  # 你需要实现

# ================== Perlin 异常合成模块 ==================
class PerlinAnomalySynthesizer:
    def __init__(self, scale=10):
        self.scale = scale
    
    def generate_anomaly(self, image):
        """生成Perlin噪声异常"""
        pass  # 你需要实现

# ================== 判别器模块 ==================
class AnomalyDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        pass  # 你需要实现
    
    def forward(self, glass_score, perlin_score, normal_score):
        """结合三个分数，计算最终异常分数"""
        pass  # 你需要实现

# ================== 特征提取模块 ==================
class FeatureExtractor(nn.Module):
    def __init__(self, backbone, layers_to_extract_from):
        super().__init__()
        pass  # 你需要实现
    
    def forward(self, images):
        """提取图像特征"""
        pass  # 你需要实现

# ================== Patch 相关模块 ==================
class MeanMapper(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        pass  # 你需要实现
    
    def forward(self, features):
        """对特征进行均值映射"""
        pass  # 你需要实现

class PatchMaker:
    def __init__(self, patch_size):
        self.patch_size = patch_size
    
    def create_patches(self, image):
        """将图像划分为多个patch"""
        pass  # 你需要实现

# ================== 损失函数 ==================
def local_loss(pred, target):
    """计算基于局部损失的异常分数"""
    pass  # 你需要实现

# ================== 可视化模块 ==================
class AnomalyVisualizer:
    def __init__(self):
        pass
    
    def visualize(self, normal_img, glass_anomaly, perlin_anomaly):
        """可视化不同异常与正常图像"""
        pass  # 你需要实现