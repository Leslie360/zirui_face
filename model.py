import torch
import torch.nn as nn
from config import cfg

class MemristorCNN(nn.Module):
    def __init__(self, num_classes=cfg.NUM_CLASSES):  # 类别数来自cfg
        super().__init__()
        self.features = nn.Sequential(
            # 卷积层通道数来自cfg
            nn.Conv2d(3, cfg.FIRST_CONV_CHANNELS, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(cfg.FIRST_CONV_CHANNELS, cfg.SECOND_CONV_CHANNELS, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(cfg.SECOND_CONV_CHANNELS, cfg.THIRD_CONV_CHANNELS, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # 全连接层输入维度根据卷积层输出自动计算（避免硬编码）
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cfg.THIRD_CONV_CHANNELS * 4 * 4, 1024),  # 4x4是CIFAR-10池化后尺寸
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def extract_fc_weights(model: MemristorCNN):
    """Return the weights of the two FC layers as numpy arrays.

    Returns: [(W1, b1), (W2, b2)]
    """
    fc_layers = []
    for m in model.classifier:
        if isinstance(m, nn.Linear):
            w = m.weight.detach().cpu().numpy().copy()
            b = m.bias.detach().cpu().numpy().copy() if m.bias is not None else None
            fc_layers.append((w, b))
    return fc_layers

# 在model.py中添加单通道版本模型
class MemristorCNN_SingleChannel(nn.Module):
    def __init__(self, num_classes=cfg.NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            # 使用配置参数，与原始模型保持一致
            nn.Conv2d(1, cfg.FIRST_CONV_CHANNELS, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 修正第二层卷积输入通道
            nn.Conv2d(cfg.FIRST_CONV_CHANNELS, cfg.SECOND_CONV_CHANNELS, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 修正第三层卷积输入通道
            nn.Conv2d(cfg.SECOND_CONV_CHANNELS, cfg.THIRD_CONV_CHANNELS, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cfg.THIRD_CONV_CHANNELS * 4 * 4, 1024),  # 使用配置参数
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    