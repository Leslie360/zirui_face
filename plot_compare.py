import os
import numpy as np
import matplotlib.pyplot as plt
import config

EXPORT_DIR = config.EXPORT_DIR
PLOT_DIR = os.path.join(EXPORT_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

def load_acc(mode):
    p = os.path.join(EXPORT_DIR, f'compare_{mode}_acc.npy')
    if not os.path.exists(p):
        return None
    data = np.load(p, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        return dict(data)
    try:
        return data.tolist()
    except Exception:
        return data.item() if hasattr(data, 'item') else None

def plot_accuracy():
    modes = ['unidirectional', 'bidirectional']
    plt.figure(figsize=(8,5))
    for m in modes:
        acc = load_acc(m)
        if acc is None:
            continue
        train = acc.get('train_acc') if isinstance(acc, dict) else acc['train_acc']
        val = acc.get('val_acc') if isinstance(acc, dict) else acc['val_acc']
        epochs = range(1, 1 + len(train))
        plt.plot(epochs, train, marker='o', label=f'{m} train')
        plt.plot(epochs, val, marker='x', label=f'{m} val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training / Validation Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    out = os.path.join(PLOT_DIR, 'accuracy_compare.png')
    plt.savefig(out)
    plt.close()
    print('Saved', out)

def plot_conductance_histograms():
    # find saved Gp/Gn files
    files = [f for f in os.listdir(EXPORT_DIR) if f.endswith('.npy') and ('layer' in f)]
    # group by mode and epoch and layer
    groups = {}
    for f in files:
        parts = f.split('_')
        # expect like compare_{mode}_epoch{e}_layer{l}_Gp.npy
        try:
            mode = parts[1]
            epoch_part = [p for p in parts if p.startswith('epoch')][0]
            layer_part = [p for p in parts if p.startswith('layer')][0]
            epoch = int(epoch_part.replace('epoch',''))
            layer = int(layer_part.replace('layer',''))
        except Exception:
            continue
        key = (mode, epoch, layer)
        groups.setdefault(key, {})[f] = os.path.join(EXPORT_DIR, f)

    for (mode, epoch, layer), d in groups.items():
        gp_f = next((v for k,v in d.items() if k.endswith('_Gp.npy')), None)
        gn_f = next((v for k,v in d.items() if k.endswith('_Gn.npy')), None)
        if gp_f is None and gn_f is None:
            continue
        plt.figure(figsize=(8,4))
        if gp_f is not None:
            gp = np.load(gp_f)
            plt.hist(gp.flatten(), bins=60, alpha=0.6, label='Gp')
        if gn_f is not None:
            gn = np.load(gn_f)
            plt.hist(gn.flatten(), bins=60, alpha=0.6, label='Gn')
        plt.xlabel('Conductance (S)')
        plt.ylabel('Count')
        plt.title(f'{mode} epoch{epoch} layer{layer} conductance histogram')
        plt.legend()
        plt.grid(True)
        out = os.path.join(PLOT_DIR, f'{mode}_epoch{epoch}_layer{layer}_hist.png')
        plt.savefig(out)
        plt.close()
        print('Saved', out)

def main():
    plot_accuracy()
    plot_conductance_histograms()

if __name__ == '__main__':
    main()
