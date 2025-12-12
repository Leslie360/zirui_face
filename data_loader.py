import torch
from torchvision import datasets, transforms
import numpy as np

def chp_preprocess_channel(channel, bits=4):
    """Quantize a single channel image to `bits` resolution (0..2^bits-1).

    Input: channel as numpy array in range [0,255]
    Output: normalized float tensor in [0,1]
    """
    levels = 2 ** bits
    q = np.floor((channel / 255.0) * (levels - 1) + 0.5).astype(np.int32)
    # map back to [0,1]
    return q.astype(np.float32) / (levels - 1)

def chp_preprocess_image(img_pil, bits=4):
    """Apply CH-P like separation and 4-bit quantization per RGB channel.

    img_pil: PIL image RGB
    returns: tensor CxHxW float32
    """
    import numpy as np
    arr = np.array(img_pil)  # HWC, uint8
    r = chp_preprocess_channel(arr[:, :, 0], bits)
    g = chp_preprocess_channel(arr[:, :, 1], bits)
    b = chp_preprocess_channel(arr[:, :, 2], bits)
    out = np.stack([r, g, b], axis=0)
    return torch.from_numpy(out)

class CHPDataset(torch.utils.data.Dataset):
    """Top-level dataset to avoid multiprocessing pickling issues on Windows."""
    def __init__(self, ds, bits=4):
        self.ds = ds
        self.bits = bits

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        # img is tensor CxHxW in [0,1] from ToTensor; convert back to uint8 array
        arr = (img.numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        from PIL import Image
        pil = Image.fromarray(arr)
        t = chp_preprocess_image(pil, bits=self.bits)
        return t, label


def get_cifar10_loaders(batch_size=128, bits=4, data_dir='./data', num_workers=0):
    # transforms for training - include random crop/flip but keep our chp preprocess for final tensor
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(CHPDataset(trainset, bits=bits), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(CHPDataset(testset, bits=bits), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
