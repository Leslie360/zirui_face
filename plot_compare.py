import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from config import cfg

EXPORT_DIR = cfg.EXPORT_DIR
PLOT_DIR = os.path.join(EXPORT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# -------------------------------------------------
# utils
# -------------------------------------------------
def _load_hist(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"History file not found: {path}")
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == () and isinstance(obj.item(), dict):
        return obj.item()
    if isinstance(obj, dict):
        return obj
    raise ValueError(f"Unexpected history format from {path}")


def _parse_tag_from_filename(fn):
    """
    Parse info from:
    train_history_unidir__pre-4bit_ideal__bits4__ov0.0__p3__lr0.01__ep5.npy
    """
    base = os.path.basename(fn)
    info = {}

    patterns = {
        "mode": r"train_history_(unidir|bidir)",
        "pre": r"__pre-([^_]+)",
        "bits": r"__bits(\d+)",
        "ov": r"__ov([0-9\.]+)",
        "p": r"__p(\d+)",
        "lr": r"__lr([0-9\.eE-]+)",
        "ep": r"__ep(\d+)",
    }
    for k, pat in patterns.items():
        m = re.search(pat, base)
        if m:
            info[k] = m.group(1)

    return info


# -------------------------------------------------
# plotting
# -------------------------------------------------
def plot_fig7_bc(hist_u, hist_b, info_u=None, info_b=None, out_path=None):
    epochs_u = np.arange(1, len(hist_u["test_acc"]) + 1)
    epochs_b = np.arange(1, len(hist_b["test_acc"]) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs_u, hist_u["test_acc"], "-o", lw=2, ms=4, label="Unidirectional")
    plt.plot(epochs_b, hist_b["test_acc"], "-s", lw=2, ms=4, label="Bidirectional")

    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    title = "CIFAR-10 recognition (memristor CNN)"
    if info_u:
        desc = []
        if "pre" in info_u:
            desc.append(f"pre={info_u['pre']}")
        if "bits" in info_u:
            desc.append(f"{info_u['bits']}-bit")
        if "ov" in info_u:
            desc.append(f"overlap={info_u['ov']}")
        if "p" in info_u:
            desc.append(f"pulses≤{info_u['p']}")
        if desc:
            title += "\n(" + ", ".join(desc) + ")"

    plt.title(title)
    plt.tight_layout()

    if out_path is None:
        out_path = os.path.join(PLOT_DIR, "fig7_bc.png")

    plt.savefig(out_path, dpi=300)
    print("[OK] Saved Fig7(b,c) to:", out_path)


# -------------------------------------------------
# main
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--u", type=str, default=None, help="unidir history .npy")
    parser.add_argument("--b", type=str, default=None, help="bidir history .npy")
    parser.add_argument("--out", type=str, default=None, help="output png path")
    args = parser.parse_args()

    # default (backward compatible)
    u_path = args.u or os.path.join(EXPORT_DIR, "hist_unidir.npy")
    b_path = args.b or os.path.join(EXPORT_DIR, "hist_bidir.npy")

    hist_u = _load_hist(u_path)
    hist_b = _load_hist(b_path)

    info_u = _parse_tag_from_filename(u_path)
    info_b = _parse_tag_from_filename(b_path)

    plot_fig7_bc(hist_u, hist_b, info_u, info_b, args.out)


if __name__ == "__main__":
    main()
