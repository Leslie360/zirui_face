"""
Run comparison between unidirectional and bidirectional update modes
WITH train/test separated RGB preprocessing.

- Train: fixed ideal RGB separation (overlap=0)
- Test : sweep overlap in cfg.RGB_OVERLAP_LIST

For each overlap:
  run unidir + bidir
  save histories under exports/
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import cfg
from data_loader import get_cifar10_loaders
from model import get_model
from memristor_model import MemristorModel


# ------------------------------
# utils
# ------------------------------
def set_seed(seed: int):
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_tag(mode: str) -> str:
    """Make a filename tag that encodes key conditions."""
    # Train config
    pre_tr = getattr(cfg, "RGB_PRE_MODE_TRAIN", getattr(cfg, "RGB_PREPROCESS_MODE", "NA"))
    ov_tr = getattr(cfg, "RGB_OVERLAP_TRAIN", getattr(cfg, "RGB_OVERLAP", 0.0))

    # Test config
    pre_te = getattr(cfg, "RGB_PRE_MODE_TEST", getattr(cfg, "RGB_PREPROCESS_MODE", "NA"))
    ov_te = getattr(cfg, "RGB_OVERLAP_TEST", getattr(cfg, "RGB_OVERLAP", 0.0))

    bits = getattr(cfg, "BITS", getattr(cfg, "RGB_BITS", 4))
    pulses = getattr(cfg, "MAX_PULSES_PER_STEP", 1)
    lr = getattr(cfg, "LR", 0.01)
    ep = getattr(cfg, "EPOCHS", 0)

    return (f"{mode}"
            f"__tr-{pre_tr}__trOv{ov_tr}"
            f"__te-{pre_te}__teOv{ov_te}"
            f"__bits{bits}__p{pulses}__lr{lr}__ep{ep}")


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    correct, total = 0, 0
    running_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        running_loss += loss.item() * x.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1), running_loss / max(total, 1)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    correct, total = 0, 0
    running_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1), running_loss / max(total, 1)


# ------------------------------
# core run
# ------------------------------
def run_mode(mode: str, epochs: int, batch_size: int, device):
    # loaders
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size,
        data_dir=cfg.DATA_DIR,
        num_workers=cfg.NUM_WORKERS,
    )

    # model
    model = get_model(getattr(cfg, "MODEL_VARIANT", "base"), num_classes=cfg.NUM_CLASSES).to(device)

    # memristor model (load curves)
    mem_model = MemristorModel(g_min=cfg.G_MIN, g_max=cfg.G_MAX)
    mem_model.mode = mode  # "unidir" or "bidir"

    ltp_ltd_path = getattr(cfg, "LTP_LTD_PATH", None)
    if ltp_ltd_path is None or len(str(ltp_ltd_path).strip()) == 0:
        ltp_ltd_path = os.path.join(cfg.DATA_DIR, "ltp_ltd.txt")

    if os.path.exists(ltp_ltd_path):
        ltp, ltd = MemristorModel.load_ltp_ltd_csv(ltp_ltd_path, ltp_count=cfg.LTP_COUNT)

        # 如果数据为负但代表电导幅值，统一取绝对值
        for k in ltp:
            ltp[k] = np.abs(ltp[k])
        for k in ltd:
            ltd[k] = np.abs(ltd[k])

        mem_model.set_color_curves(ltp, ltd)
    else:
        print(f"[WARN] LTP/LTD file not found: {ltp_ltd_path}. Continue without curve file.")

    # attach memristor model to network if your model expects it
    # (你的 model.py 如果提供了类似 set_memristor_model 的接口就启用)
    if hasattr(model, "set_memristor_model"):
        model.set_memristor_model(mem_model)
    elif hasattr(model, "mem_model"):
        model.mem_model = mem_model

    criterion = nn.CrossEntropyLoss()

    opt_name = str(getattr(cfg, "OPTIMIZER", "sgd")).lower()
    wd = getattr(cfg, "WEIGHT_DECAY", 0.0)
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=wd)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=wd)
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.LR,
            momentum=0.9,
            weight_decay=wd,
        )

    sched_type = str(getattr(cfg, "LR_SCHED_TYPE", "none")).lower()
    scheduler = None
    if sched_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=getattr(cfg, "LR_SCHED_STEP", 30),
            gamma=getattr(cfg, "LR_SCHED_GAMMA", 0.1),
        )
    elif sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=getattr(cfg, "LR_SCHED_TMAX", cfg.EPOCHS),
        )

    hist = {
        "mode": mode,
        "train_pre": getattr(cfg, "RGB_PRE_MODE_TRAIN", getattr(cfg, "RGB_PREPROCESS_MODE", "NA")),
        "test_pre": getattr(cfg, "RGB_PRE_MODE_TEST", getattr(cfg, "RGB_PREPROCESS_MODE", "NA")),
        "train_overlap": float(getattr(cfg, "RGB_OVERLAP_TRAIN", getattr(cfg, "RGB_OVERLAP", 0.0))),
        "test_overlap": float(getattr(cfg, "RGB_OVERLAP_TEST", getattr(cfg, "RGB_OVERLAP", 0.0))),
        "epochs": int(epochs),
        "train_acc": [],
        "test_acc": [],
        "train_loss": [],
        "test_loss": [],
    }

    print(f"[MODE] UPDATE_MODE={mode}, MAX_PULSES_PER_STEP={getattr(cfg,'MAX_PULSES_PER_STEP',1)}")
    print(f"[TRAIN-PRE] {hist['train_pre']} ov={hist['train_overlap']} | "
          f"[TEST-PRE] {hist['test_pre']} ov={hist['test_overlap']} | bits={getattr(cfg,'BITS',4)}")

    for ep in range(1, epochs + 1):
        tr_acc, tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        te_acc, te_loss = evaluate(model, test_loader, criterion, device)

        hist["train_acc"].append(tr_acc)
        hist["test_acc"].append(te_acc)
        hist["train_loss"].append(tr_loss)
        hist["test_loss"].append(te_loss)

        if scheduler is not None:
            scheduler.step()

        print(f"Epoch {ep:03d}/{epochs} | train_acc={tr_acc:.4f} test_acc={te_acc:.4f} "
              f"train_loss={tr_loss:.4f} test_loss={te_loss:.4f}")

    return hist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--export_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)  # "cuda" / "cpu"
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if args.export_dir is not None:
        cfg.EXPORT_DIR = args.export_dir
    os.makedirs(cfg.EXPORT_DIR, exist_ok=True)

    if args.epochs is not None:
        cfg.EPOCHS = int(args.epochs)
    if args.batch_size is not None:
        cfg.BATCH_SIZE = int(args.batch_size)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # ------------------------------
    # FIX train pre-process (ideal)
    # ------------------------------
    cfg.RGB_PRE_MODE_TRAIN = "4bit_ideal"
    cfg.RGB_OVERLAP_TRAIN = 0.0

    # test pre-process (overlap sweep)
    cfg.RGB_PRE_MODE_TEST = "4bit_overlap"

    overlap_list = getattr(cfg, "RGB_OVERLAP_LIST", [0.0, 0.1, 0.3, 0.5])
    epochs = cfg.EPOCHS
    batch_size = cfg.BATCH_SIZE

    all_saved = []

    for ov in overlap_list:
        cfg.RGB_OVERLAP_TEST = float(ov)

        print("\n" + "=" * 90)
        print(f"[SWEEP-TEST] overlap={cfg.RGB_OVERLAP_TEST} | "
              f"train={cfg.RGB_PRE_MODE_TRAIN}(ov={cfg.RGB_OVERLAP_TRAIN}) "
              f"test={cfg.RGB_PRE_MODE_TEST}(ov={cfg.RGB_OVERLAP_TEST})")
        print("=" * 90)

        hist_unidir = run_mode("unidir", epochs=epochs, batch_size=batch_size, device=device)
        hist_bidir = run_mode("bidir", epochs=epochs, batch_size=batch_size, device=device)

        tag_u = make_tag("unidir")
        tag_b = make_tag("bidir")

        p_u = os.path.join(cfg.EXPORT_DIR, f"train_history_{tag_u}.npy")
        p_b = os.path.join(cfg.EXPORT_DIR, f"train_history_{tag_b}.npy")
        np.save(p_u, hist_unidir)
        np.save(p_b, hist_bidir)
        all_saved += [p_u, p_b]

        # easy filenames (avoid overwrite)
        ov_str = str(cfg.RGB_OVERLAP_TEST).replace(".", "p")
        p_u_easy = os.path.join(cfg.EXPORT_DIR, f"hist_unidir_testOv{ov_str}.npy")
        p_b_easy = os.path.join(cfg.EXPORT_DIR, f"hist_bidir_testOv{ov_str}.npy")
        np.save(p_u_easy, hist_unidir)
        np.save(p_b_easy, hist_bidir)
        all_saved += [p_u_easy, p_b_easy]

        print("\n[OK] Saved:")
        print(" ", p_u)
        print(" ", p_b)
        print(" ", p_u_easy)
        print(" ", p_b_easy)

    print("\n" + "=" * 90)
    print("[OK] Sweep finished. Saved files:")
    for p in all_saved:
        print(" ", p)
    print("=" * 90)


if __name__ == "__main__":
    main()
