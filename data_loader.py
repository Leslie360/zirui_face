import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
from config import cfg


def _quantize01(x01: np.ndarray, bits: int) -> np.ndarray:
    """Quantize [0,1] float array to bits, then map back to [0,1]."""
    levels = (1 << bits) - 1
    x01 = np.clip(x01, 0.0, 1.0)
    q = np.rint(x01 * levels) / levels
    return q.astype(np.float32)


def _rgb_overlap_mix(img01: np.ndarray, overlap: float) -> np.ndarray:
    """
    img01: HxWx3 in [0,1]
    overlap: alpha in [0,1]
    """
    a = float(np.clip(overlap, 0.0, 1.0))
    r = img01[..., 0]
    g = img01[..., 1]
    b = img01[..., 2]

    r2 = (1 - a) * r + a * (g + b) * 0.5
    g2 = (1 - a) * g + a * (r + b) * 0.5
    b2 = (1 - a) * b + a * (r + g) * 0.5

    out = np.stack([r2, g2, b2], axis=-1)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def preprocess_rgb_tensor(x: torch.Tensor, mode: str, bits: int, overlap: float) -> torch.Tensor:
    """
    x: torch Tensor (3,H,W), assumed in [0,1] from ToTensor()
    returns: torch Tensor (3,H,W) in [0,1]
    """
    mode = (mode or "none").lower()
    bits = int(bits)

    # to numpy HWC
    img = x.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)

    if mode in ["none", "raw"]:
        out = img
    elif mode in ["4bit_ideal", "ideal", "4bit"]:
        out = _quantize01(img, bits)
    elif mode in ["4bit_overlap", "overlap"]:
        mixed = _rgb_overlap_mix(img, overlap)
        out = _quantize01(mixed, bits)
    else:
        raise ValueError(f"Unknown RGB preprocess mode: {mode}")

    # back to CHW torch
    out_t = torch.from_numpy(out.transpose(2, 0, 1)).to(dtype=torch.float32)
    return out_t


class CIFAR10PreprocessDataset(Dataset):
    def __init__(self, base_ds, split: str):
        self.base = base_ds
        self.split = split.lower().strip()

        # train/test split configs (with backward-compatible fallback)
        if self.split == "train":
            self.mode = getattr(cfg, "RGB_PRE_MODE_TRAIN", getattr(cfg, "RGB_PREPROCESS_MODE", "4bit_ideal"))
            self.overlap = float(getattr(cfg, "RGB_OVERLAP_TRAIN", getattr(cfg, "RGB_OVERLAP", 0.0)))
        else:
            self.mode = getattr(cfg, "RGB_PRE_MODE_TEST", getattr(cfg, "RGB_PREPROCESS_MODE", "4bit_overlap"))
            self.overlap = float(getattr(cfg, "RGB_OVERLAP_TEST", getattr(cfg, "RGB_OVERLAP", 0.0)))

        self.bits = int(getattr(cfg, "BITS", 4))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]  # x is (3,H,W) from ToTensor
        x2 = preprocess_rgb_tensor(x, mode=self.mode, bits=self.bits, overlap=self.overlap)
        return x2, y


def get_cifar10_loaders(batch_size=None, num_workers=2, data_dir=None):
    if batch_size is None:
        batch_size = int(getattr(cfg, "BATCH_SIZE", 64))
    if data_dir is None:
        data_dir = getattr(cfg, "DATA_DIR", "data")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_base = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_base = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    train_ds = CIFAR10PreprocessDataset(train_base, split="train")
    test_ds = CIFAR10PreprocessDataset(test_base, split="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
