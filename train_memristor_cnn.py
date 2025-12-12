import torch
import torch.nn as nn
import torch.optim as optim
from model import MemristorCNN, extract_fc_weights
from data_loader import get_cifar10_loaders
from memristor_model import MemristorModel
import numpy as np

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        _, pred = out.max(1)
        correct += pred.eq(y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            _, pred = out.max(1)
            correct += pred.eq(y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total

def map_weights_to_memristors(fc_weights, mem_model: MemristorModel):
    """Given list of FC weights (W,b), create G+ and G- arrays mapped from weights.

    We'll linearly map weight values to conductance differences.
    """
    mapped = []
    for W, b in fc_weights:
        # W shape: out x in
        max_abs = np.max(np.abs(W)) if W.size else 1.0
        # normalize to [-1,1]
        Wn = W / (max_abs + 1e-12)
        Gp = np.zeros_like(Wn)
        Gn = np.zeros_like(Wn)
        for i in range(Wn.shape[0]):
            for j in range(Wn.shape[1]):
                w = Wn[i, j]
                g_p, g_n = mem_model.map_weight_to_conductances(w, g_ref=(mem_model.g_min+mem_model.g_max)/2)
                Gp[i, j] = g_p
                Gn[i, j] = g_n
        mapped.append((Gp, Gn))
    return mapped

def main(epochs=None, batch_size=None, lr=None, device='cuda'):
    import config
    epochs = epochs if epochs is not None else config.EPOCHS
    batch_size = batch_size if batch_size is not None else config.BATCH_SIZE
    lr = lr if lr is not None else config.LR
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size, num_workers=config.NUM_WORKERS, data_dir=config.DATA_DIR)
    model = MemristorCNN(num_classes=config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mem_model = MemristorModel(g_min=config.G_MIN, g_max=config.G_MAX)

    # try auto-loading color LTP/LTD files from ./data
    import os
    data_dir = config.DATA_DIR
    # support either CSV or TXT combined file
    ltp_ltd_path = None
    for fname in (cfg.LTP_LTD_CSV, 'ltp_ltd.txt'):
        p = os.path.join(data_dir, fname)
        if os.path.exists(p):
            ltp_ltd_path = p
            break
    ltp_rgb_path = os.path.join(data_dir, 'ltp_rgb.csv')
    ltd_rgb_path = os.path.join(data_dir, 'ltd_rgb.csv')
    ltp_npy = os.path.join(data_dir, 'ltp_rgb.npy')
    ltd_npy = os.path.join(data_dir, 'ltd_rgb.npy')
    try:
        if os.path.exists(ltp_ltd_path):
            ltp, ltd = MemristorModel.load_ltp_ltd_csv(ltp_ltd_path, ltp_count=config.LTP_COUNT)
            mem_model.set_color_curves(ltp, ltd)
            print('Loaded combined ltp_ltd.csv from data/')
        elif os.path.exists(ltp_rgb_path):
            ltp = MemristorModel.load_rgb_csv(ltp_rgb_path)
            ltd = None
            if os.path.exists(ltd_rgb_path):
                ltd = MemristorModel.load_rgb_csv(ltd_rgb_path)
            mem_model.set_color_curves(ltp, ltd)
            print('Loaded LTP/LTD RGB curves from CSV in data/')
        elif os.path.exists(ltp_npy):
            arr = np.load(ltp_npy, allow_pickle=True).item()
            ltp = arr.get('ltp') if isinstance(arr, dict) else arr
            ltd = None
            if os.path.exists(ltd_npy):
                arr2 = np.load(ltd_npy, allow_pickle=True).item()
                ltd = arr2.get('ltd') if isinstance(arr2, dict) else arr2
            mem_model.set_color_curves(ltp, ltd)
            print('Loaded LTP/LTD RGB curves from npy in data/')
        else:
            print('No LTP/LTD files found in data/; using synthesized curves')
    except Exception as e:
        print('Could not auto-load LTP/LTD curves:', e)

    # initialize memristor arrays for FC layers using current weights
    import config as cfg
    os.makedirs(cfg.EXPORT_DIR, exist_ok=True)
    prev_fc = None
    # create initial mapped Gp/Gn using map_weights_to_memristors (linear mapping)
    fc_weights = extract_fc_weights(model)
    mapped = map_weights_to_memristors(fc_weights, mem_model)
    # mapped is list of (Gp, Gn) per FC layer
    Gp_layers = [m[0] for m in mapped]
    Gn_layers = [m[1] for m in mapped]

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} acc={train_acc:.4f} val_acc={val_acc:.4f}")

        # after epoch snapshot FC weights and optionally apply hardware-in-loop writes
        fc_weights = extract_fc_weights(model)
        # prepare deltaW normalized per-layer
        deltaW_layers = []
        for idx, (W, b) in enumerate(fc_weights):
            # compute previous weights snapshot
            if prev_fc is None:
                prev = np.zeros_like(W)
            else:
                prev = prev_fc[idx][0]
            # weight delta
            dW = W - prev
            # normalize by max abs in this layer to [-1,1]
            norm = np.max(np.abs(W)) if W.size else 1.0
            dWn = dW / (norm + 1e-12)
            deltaW_layers.append(dWn)

        if cfg.HARDWARE_IN_LOOP:
            # apply to each FC layer using color mapping; we use color-agnostic here or pick 'r' as default
            for li in range(len(Gp_layers)):
                Gp, Gn = Gp_layers[li], Gn_layers[li]
                dWn = deltaW_layers[li]
                # apply mapping using configured mode
                Gp_new, Gn_new, pp, pn = mem_model.apply_weight_matrix_changes(Gp, Gn, dWn, color='r', mode=cfg.WEIGHT_UPDATE_MODE)
                Gp_layers[li] = Gp_new
                Gn_layers[li] = Gn_new
                # save pulses or stats if desired

        # save mapped conductances for inspection per epoch
        if cfg.EXPORT_EPOCH_NPY:
            for li, (Gp, Gn) in enumerate(zip(Gp_layers, Gn_layers)):
                np.save(os.path.join(cfg.EXPORT_DIR, f'epoch{epoch}_layer{li}_Gp.npy'), Gp)
                np.save(os.path.join(cfg.EXPORT_DIR, f'epoch{epoch}_layer{li}_Gn.npy'), Gn)

        prev_fc = fc_weights

    # final save
    torch.save(model.state_dict(), 'memristor_cnn_final.pth')

if __name__ == '__main__':
    main()
