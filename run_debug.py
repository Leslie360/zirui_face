from model import MemristorCNN
from train_memristor_cnn import train_one_epoch, evaluate
from data_loader import get_cifar10_loaders
from memristor_model import MemristorModel
import torch
import config

def run_debug():
    device = torch.device('cpu')
    bs = 32
    train_loader, test_loader = get_cifar10_loaders(batch_size=bs, bits=4, data_dir=config.DATA_DIR, num_workers=0)
    model = MemristorCNN(num_classes=config.NUM_CLASSES).to(device)
    mem_model = MemristorModel(g_min=config.G_MIN, g_max=config.G_MAX)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    print('Running one quick train+eval epoch (debug)')
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, test_loader, criterion, device)
    print(f'debug train loss {train_loss:.4f} acc {train_acc:.4f} val {val_acc:.4f}')

if __name__ == '__main__':
    run_debug()
