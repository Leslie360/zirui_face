import torch
import torch.nn as nn

class MemristorCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Three conv layers with pooling
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Two fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
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
