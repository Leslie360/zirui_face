"""Run small-scale comparison between unidirectional and bidirectional update modes.

This script runs short training (few epochs) with `HARDWARE_IN_LOOP` enabled and
saves per-epoch mapped Gp/Gn arrays and accuracy logs for both modes.
"""
import os
import argparse
import numpy as np
import torch
import config
from model import MemristorCNN
from data_loader import get_cifar10_loaders
from memristor_model import MemristorModel
from train_memristor_cnn import train_one_epoch, evaluate, extract_fc_weights, map_weights_to_memristors

def run_mode(mode, epochs=2, batch_size=32, device='cpu'):
    # local override of config for quick run
    cfg = config
    cfg.HARDWARE_IN_LOOP = True
    cfg.WEIGHT_UPDATE_MODE = mode
    cfg.NUM_WORKERS = 0
    device = torch.device(device)

    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size, bits=4, data_dir=cfg.DATA_DIR, num_workers=0)
    model = MemristorCNN(num_classes=cfg.NUM_CLASSES).to(device)
    mem_model = MemristorModel(g_min=cfg.G_MIN, g_max=cfg.G_MAX)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)

    # auto-load device curves if present (same logic as train script)
    try:
        # support txt fallback
        ltp_ltd_path = os.path.join(cfg.DATA_DIR, cfg.LTP_LTD_CSV)
        if not os.path.exists(ltp_ltd_path):
            alt = os.path.join(cfg.DATA_DIR, 'ltp_ltd.txt')
            if os.path.exists(alt):
                ltp_ltd_path = alt
        ltp_rgb_path = os.path.join(cfg.DATA_DIR, cfg.LTP_RGB_CSV)
        ltp_npy = os.path.join(cfg.DATA_DIR, cfg.LTP_NPY)
        if ltp_ltd_path and os.path.exists(ltp_ltd_path):
            ltp, ltd = MemristorModel.load_ltp_ltd_csv(ltp_ltd_path, ltp_count=cfg.LTP_COUNT)
            mem_model.set_color_curves(ltp, ltd)
            print('Loaded combined ltp_ltd.csv for run_compare')
        elif os.path.exists(ltp_rgb_path):
            ltp = MemristorModel.load_rgb_csv(ltp_rgb_path)
            ltd = None
            mem_model.set_color_curves(ltp, ltd)
            print('Loaded ltp_rgb.csv for run_compare')
    except Exception as e:
        print('No device curves loaded for run_compare:', e)

    # initialize mapping
    fc_weights = extract_fc_weights(model)
    mapped = map_weights_to_memristors(fc_weights, mem_model)
    Gp_layers = [m[0] for m in mapped]
    Gn_layers = [m[1] for m in mapped]

    acc_log = {'train_acc': [], 'val_acc': []}
    os.makedirs(cfg.EXPORT_DIR, exist_ok=True)

    for epoch in range(1, epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        print(f'[{mode}] Epoch {epoch}: train_acc={train_acc:.4f} val_acc={val_acc:.4f}')
        acc_log['train_acc'].append(train_acc)
        acc_log['val_acc'].append(val_acc)

        # get new fc weights and apply hardware writes based on delta
        new_fc = extract_fc_weights(model)
        prev = fc_weights
        deltaW_layers = []
        for idx, (W, b) in enumerate(new_fc):
            prevW = prev[idx][0]
            dW = W - prevW
            norm = np.max(np.abs(W)) if W.size else 1.0
            deltaW_layers.append(dW / (norm + 1e-12))

        # apply to memristor arrays
        for li in range(len(Gp_layers)):
            Gp, Gn = Gp_layers[li], Gn_layers[li]
            dWn = deltaW_layers[li]
            Gp_new, Gn_new, pp, pn = mem_model.apply_weight_matrix_changes(Gp, Gn, dWn, color='r', mode=mode)
            Gp_layers[li] = Gp_new
            Gn_layers[li] = Gn_new
            # save snapshot
            np.save(os.path.join(cfg.EXPORT_DIR, f'compare_{mode}_epoch{epoch}_layer{li}_Gp.npy'), Gp_new)
            np.save(os.path.join(cfg.EXPORT_DIR, f'compare_{mode}_epoch{epoch}_layer{li}_Gn.npy'), Gn_new)

        fc_weights = new_fc

    # save accuracy log
    np.save(os.path.join(cfg.EXPORT_DIR, f'compare_{mode}_acc.npy'), acc_log)
    return acc_log

def main():
    parser = argparse.ArgumentParser(description='Run small compare between unidirectional and bidirectional mapping')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--modes', type=str, default='unidirectional,bidirectional',
                        help='comma-separated modes to run')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--color-mapping', type=str, default=None, choices=['round_robin','blocks', 'auto'],
                        help='override color mapping strategy')
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    results = {}

    # possibly override config color mapping
    if args.color_mapping is not None:
        config.COLOR_MAPPING = args.color_mapping

    for m in modes:
        print('Running mode', m)
        results[m] = run_mode(m, epochs=args.epochs, batch_size=args.batch_size, device=args.device)
    print('Compare finished. Results saved in', config.EXPORT_DIR)


if __name__ == '__main__':
    main()
