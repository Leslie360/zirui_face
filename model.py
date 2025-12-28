import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
import torchvision.models as tv_models

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

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


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        
        out += identity
        out = self.relu(out)
        return out

class MemristorCNN_Strong(nn.Module):
    """ResNet-like architecture with SE blocks and Projection Head."""
    def __init__(self, num_classes=cfg.NUM_CLASSES, use_se=True):
        super().__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Stacking ResBlocks
        self.layer1 = self._make_layer(64, 2, stride=1, use_se=use_se)
        self.layer2 = self._make_layer(128, 2, stride=2, use_se=use_se)
        self.layer3 = self._make_layer(256, 2, stride=2, use_se=use_se)
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512), # 32 -> 16 -> 8
            nn.ReLU(),
            nn.Linear(512, 128)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def _make_layer(self, out_channels, blocks, stride, use_se):
        layers = []
        layers.append(ResBlock(self.in_channels, out_channels, stride, use_se))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels, stride=1, use_se=use_se))
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        x = self.conv1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        features = x.flatten(1)
        
        if return_features:
            proj = self.projection_head(features)
            return self.classifier(x), proj
            
        return self.classifier(x)


class ResNet18CIFAR(nn.Module):
    """ResNet18 adapted for CIFAR-10 (3x3 conv, no first maxpool)."""
    def __init__(self, num_classes=cfg.NUM_CLASSES):
        super().__init__()
        self.model = tv_models.resnet18(weights=None, num_classes=num_classes)
        # adjust stem for CIFAR
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()

    def forward(self, x):
        return self.model(x)


def get_model(variant: str = "base", num_classes: int = cfg.NUM_CLASSES):
    variant = (variant or "base").lower()
    if variant == "base":
        return MemristorCNN(num_classes=num_classes)
    if variant == "strong":
        return MemristorCNN_Strong(num_classes=num_classes)
    if variant in ["resnet18", "resnet"]:
        return ResNet18CIFAR(num_classes=num_classes)
    raise ValueError(f"Unknown model variant: {variant}")
