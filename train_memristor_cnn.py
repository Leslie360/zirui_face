import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import cfg
from model import get_model
from data_loader import get_cifar10_loaders
from memristor_model import MemristorModel


# ============================================================
# 0) Utilities: access classifier Linear layers
# ============================================================
def get_fc_linears(model: nn.Module):
    if not hasattr(model, "classifier"):
        raise AttributeError("Model has no attribute 'classifier'. Please check your MemristorCNN definition.")
    lins = [m for m in model.classifier if isinstance(m, nn.Linear)]
    if len(lins) < 2:
        raise ValueError(f"Expected at least 2 Linear layers in model.classifier, got {len(lins)}")
    return lins[0], lins[1]


# ============================================================
# 1) Weight discreteness check (you asked for this)
# ============================================================
def weight_quant_check(model, layer="fc1", sample_size=200000):
    with torch.no_grad():
        fc1, fc2 = get_fc_linears(model)
        if layer == "fc1":
            W = fc1.weight.detach().cpu().numpy().ravel()
        elif layer == "fc2":
            W = fc2.weight.detach().cpu().numpy().ravel()
        else:
            raise ValueError("layer must be 'fc1' or 'fc2'")

    total = W.size
    if total <= sample_size:
        sample = W
    else:
        idx = np.random.choice(total, size=sample_size, replace=False)
        sample = W[idx]

    uniq = np.unique(sample).size
    uniq_3 = np.unique(np.round(sample, 3)).size
    uniq_4 = np.unique(np.round(sample, 4)).size
    uniq_5 = np.unique(np.round(sample, 5)).size

    print(
        f"[WEIGHT-CHECK] {layer}: total={total}, "
        f"unique(sample)={uniq}, unique@1e-3={uniq_3}, unique@1e-4={uniq_4}, unique@1e-5={uniq_5}, "
        f"min={sample.min():.4e}, max={sample.max():.4e}"
    )


# ============================================================
# 2) Build average discrete conductance states from your file
# ============================================================
def build_g_states_per_color(ltp_rgb: dict, ltd_rgb: dict):
    """Return dict of color->g_states (unique, sorted, abs)."""
    out = {}
    min_len = None
    for c in ["r", "g", "b"]:
        g = np.concatenate([ltp_rgb[c].ravel(), ltd_rgb[c].ravel()], axis=0).astype(float)
        g = np.abs(g)
        g = np.unique(g)
        g.sort()
        if g.size < 4:
            raise ValueError("g_states too small for color " + c)
        out[c] = g
        min_len = g.size if min_len is None else min(min_len, g.size)

    # truncate to min length to allow stacking
    for c in out:
        out[c] = out[c][:min_len].copy()
    return out


# ============================================================
# 3) Memristor FCL state + update
# ============================================================
class MemFCLState:
    def __init__(self, g_states_color: dict, max_pulses_per_step: int = 1, color_mapping: str = "round_robin"):
        # stack per-color states to allow fast indexing
        g_r, g_g, g_b = g_states_color["r"], g_states_color["g"], g_states_color["b"]
        min_len = min(g_r.size, g_g.size, g_b.size)
        self.N = int(min_len)
        self.g_stack = np.stack([
            g_r[:min_len],
            g_g[:min_len],
            g_b[:min_len],
        ], axis=0).astype(float)  # shape (3,N)
        self.g_min = float(self.g_stack.min())
        self.g_max = float(self.g_stack.max())
        self.g_range = max(self.g_max - self.g_min, 1e-30)
        self.g_step_scalar = (self.g_stack[:, -1] - self.g_stack[:, 0]) / max(self.N - 1, 1)
        self.max_pulses = int(max_pulses_per_step)
        self.color_mapping = color_mapping

        self.layers = {}
        self._pulse_sum = 0.0
        self._pulse_count = 0

    def reset_pulse_stats(self):
        self._pulse_sum = 0.0
        self._pulse_count = 0

    def pulse_stats(self):
        if self._pulse_count == 0:
            return 0.0
        return float(self._pulse_sum / self._pulse_count)

    def _color_indices(self, shape, strategy="round_robin"):
        rows, cols = shape
        color_idx = np.zeros(shape, dtype=np.int32)
        if strategy == "round_robin":
            for i in range(rows):
                for j in range(cols):
                    color_idx[i, j] = (i * cols + j) % 3
        elif strategy == "blocks":
            for j in range(cols):
                block = int(j * 3 / max(1, cols))
                color_idx[:, j] = min(block, 2)
        else:
            color_idx[:] = 0
        return color_idx

    def init_layer_from_float(self, name: str, W_float: np.ndarray, scale_factor: float = 1.0):
        W = W_float.astype(float)
        mid = (self.N - 1) // 2

        w_abs_p99 = float(np.percentile(np.abs(W), 99))
        if w_abs_p99 < 1e-12:
            w_abs_p99 = float(np.std(W) + 1e-6)
        S = (2.0 * w_abs_p99) / self.g_range
        S = S * max(scale_factor, 1e-12)
        S = max(S, 1e-12)

        color_idx = self._color_indices(W.shape, self.color_mapping)
        g_step = self.g_step_scalar[color_idx]

        deltaG = W / S
        steps = np.rint(deltaG / g_step).astype(np.int32)

        idx_p = np.full_like(steps, mid, dtype=np.int32)
        idx_n = np.full_like(steps, mid, dtype=np.int32)

        pos = steps > 0
        neg = steps < 0
        idx_p[pos] = idx_p[pos] + steps[pos]
        idx_n[neg] = idx_n[neg] + (-steps[neg])

        idx_p = np.clip(idx_p, 0, self.N - 1)
        idx_n = np.clip(idx_n, 0, self.N - 1)

        self.layers[name] = {
            "idx_p": idx_p,
            "idx_n": idx_n,
            "scale": float(S),
            "shape": W.shape,
            "color_idx": color_idx,
        }

    def _indices_to_weight(self, idx_p: np.ndarray, idx_n: np.ndarray, scale: float, color_idx: np.ndarray):
        Gp = self.g_stack[color_idx, idx_p]
        Gn = self.g_stack[color_idx, idx_n]
        return (Gp - Gn) * scale

    def writeback(self, name: str, W_tensor: torch.Tensor):
        info = self.layers[name]
        W = self._indices_to_weight(info["idx_p"], info["idx_n"], info["scale"], info["color_idx"])
        W_tensor.data.copy_(torch.from_numpy(W).to(W_tensor.device, dtype=W_tensor.dtype))

    def update(self, name: str, grad_W: np.ndarray, lr: float, mode: str):
        """
        Update idx_p/idx_n according to deltaW = -lr * grad.
        pulses computed from deltaG = |deltaW|/S and g_step.
        mode:
          - 'unidir': only allow idx += pulses (potentiation only) on either plus or minus side
          - 'bidir' : allow idx += pulses (pot) and idx -= pulses (dep) depending on need
        """
        info = self.layers[name]
        idx_p = info["idx_p"]
        idx_n = info["idx_n"]
        color_idx = info["color_idx"]
        S = float(info["scale"])

        deltaW = (-lr) * grad_W.astype(float)
        deltaG = np.abs(deltaW) / max(S, 1e-30)
        g_step = self.g_step_scalar[color_idx]
        pulses = np.rint(deltaG / np.maximum(g_step, 1e-30)).astype(np.int32)
        pulses = np.clip(pulses, 0, self.max_pulses)

        # pulse stats
        self._pulse_sum += float(np.mean(pulses))
        self._pulse_count += 1

        if mode not in ("unidir", "bidir"):
            raise ValueError("mode must be 'unidir' or 'bidir'")

        if mode == "unidir":
            # only potentiation allowed: idx += pulses
            # deltaW > 0 => increase G+ ; deltaW < 0 => increase G-
            pos = deltaW > 0
            neg = deltaW < 0
            idx_p[pos] = idx_p[pos] + pulses[pos]
            idx_n[neg] = idx_n[neg] + pulses[neg]

        else:
            # bidirectional:
            # deltaW > 0: prefer increase G+; if saturated then decrease G- (idx_n -=)
            # deltaW < 0: prefer increase G-; if saturated then decrease G+ (idx_p -=)
            pos = deltaW > 0
            neg = deltaW < 0

            # ---- positive updates: prefer idx_p += pulses, else idx_n -= pulses ----
            inc_p = pos & (idx_p < (self.N - 1))
            idx_p[inc_p] += pulses[inc_p]

            fallback_pos = pos & ~inc_p
            idx_n[fallback_pos] -= pulses[fallback_pos]

            # ---- negative updates: prefer idx_n += pulses, else idx_p -= pulses ----
            inc_n = neg & (idx_n < (self.N - 1))
            idx_n[inc_n] += pulses[inc_n]

            fallback_neg = neg & ~inc_n
            idx_p[fallback_neg] -= pulses[fallback_neg]

        info["idx_p"] = np.clip(idx_p, 0, self.N - 1)
        info["idx_n"] = np.clip(idx_n, 0, self.N - 1)


def build_optimizer_conv_and_bias(model: nn.Module, lr: float):
    """
    Optimizer updates:
      - all parameters except classifier Linear.weight (we hard-update those with memristor)
      - classifier Linear.bias can still be optimized (keeps baseline performance)
    """
    fc1, fc2 = get_fc_linears(model)
    fc_weight_ids = {id(fc1.weight), id(fc2.weight)}

    params = []
    for p in model.parameters():
        if id(p) in fc_weight_ids:
            continue
        params.append(p)

    opt = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=getattr(cfg, "WEIGHT_DECAY", 0.0))
    return opt


# ============================================================
# 4) Train / Eval
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device, fcl_state: MemFCLState, mode: str):
    model.train()
    correct, total = 0, 0
    running_loss = 0.0

    fc1, fc2 = get_fc_linears(model)

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # 1) update conv+others (excluding fc weights)
        optimizer.step()

        # 2) memristor update for fc weights
        if fcl_state is not None:
            # grad -> numpy
            g1 = fc1.weight.grad.detach().cpu().numpy()
            g2 = fc2.weight.grad.detach().cpu().numpy()

            fcl_state.update("fc1", g1, lr=optimizer.param_groups[0]["lr"], mode=mode)
            fcl_state.update("fc2", g2, lr=optimizer.param_groups[0]["lr"], mode=mode)

            # write back to torch weights (hard overwrite)
            fcl_state.writeback("fc1", fc1.weight)
            fcl_state.writeback("fc2", fc2.weight)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return acc, avg_loss


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    correct, total = 0, 0
    running_loss = 0.0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return acc, avg_loss


# ============================================================
# 5) Main
# ============================================================
def train(epochs=None, batch_size=None, lr=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = epochs if epochs is not None else cfg.EPOCHS
    batch_size = batch_size if batch_size is not None else cfg.BATCH_SIZE
    lr = lr if lr is not None else cfg.LR

    os.makedirs(cfg.EXPORT_DIR, exist_ok=True)

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size,
        num_workers=cfg.NUM_WORKERS,
        data_dir=cfg.DATA_DIR,
    )

    model = get_model(getattr(cfg, "MODEL_VARIANT", "base"), num_classes=cfg.NUM_CLASSES).to(device)

    # Load LTP/LTD curves
    mem_model = MemristorModel(g_min=cfg.G_MIN, g_max=cfg.G_MAX)

    if cfg.LTP_LTD_PATH and os.path.exists(cfg.LTP_LTD_PATH):
        ltp, ltd = MemristorModel.load_ltp_ltd_csv(cfg.LTP_LTD_PATH, ltp_count=cfg.LTP_COUNT)
        for k in ltp:
            ltp[k] = np.abs(ltp[k])
        for k in ltd:
            ltd[k] = np.abs(ltd[k])
        mem_model.set_color_curves(ltp, ltd)

        print("[LTP/LTD] loaded:", cfg.LTP_LTD_PATH)
        for c in ["b", "g", "r"]:
            ltp_min, ltp_max = float(ltp[c].min()), float(ltp[c].max())
            ltd_min, ltd_max = float(ltd[c].min()), float(ltd[c].max()) if ltd[c].size > 0 else (float("nan"), float("nan"))
            print(f"  {c}: LTP[{ltp_min:.3e}, {ltp_max:.3e}] | LTD[{ltd_min:.3e}, {ltd_max:.3e}]")
        print(f"  cfg.G_MIN={cfg.G_MIN:.3e}, cfg.G_MAX={cfg.G_MAX:.3e}")
    else:
        raise FileNotFoundError(f"LTP/LTD file not found: {cfg.LTP_LTD_PATH}")

    # Build discrete g_states per color
    g_states_color = build_g_states_per_color(ltp, ltd)
    min_len = min(g_states_color["r"].size, g_states_color["g"].size, g_states_color["b"].size)
    print(f"[G-STATES] per-color N={min_len}, r[{g_states_color['r'][0]:.3e},{g_states_color['r'][-1]:.3e}] g[{g_states_color['g'][0]:.3e},{g_states_color['g'][-1]:.3e}] b[{g_states_color['b'][0]:.3e},{g_states_color['b'][-1]:.3e}]")

    # mode-specific pulse cap & scale
    mode = getattr(cfg, "UPDATE_MODE", "unidir")
    max_pulses_mode = getattr(cfg, "BIDIR_MAX_PULSES", None) if mode == "bidir" else getattr(cfg, "UNIDIR_MAX_PULSES", None)
    max_pulses = max_pulses_mode if max_pulses_mode is not None else getattr(cfg, "MAX_PULSES_PER_STEP", 1)

    mem_scale_default = getattr(cfg, "MEM_SCALE_DEFAULT", 1.0)
    mem_scale_mode = getattr(cfg, "BIDIR_MEM_SCALE", None) if mode == "bidir" else getattr(cfg, "UNIDIR_MEM_SCALE", None)
    mem_scale = mem_scale_mode if mem_scale_mode is not None else mem_scale_default

    # Prepare memristor FCL state with color mapping
    fcl_state = MemFCLState(
        g_states_color=g_states_color,
        max_pulses_per_step=max_pulses,
        color_mapping=getattr(cfg, "COLOR_MAPPING", "round_robin"),
    )

    # init from current float weights
    fc1, fc2 = get_fc_linears(model)
    fcl_state.init_layer_from_float("fc1", fc1.weight.detach().cpu().numpy(), scale_factor=mem_scale)
    fcl_state.init_layer_from_float("fc2", fc2.weight.detach().cpu().numpy(), scale_factor=mem_scale)

    # write back once to enforce discrete initial state
    fcl_state.writeback("fc1", fc1.weight)
    fcl_state.writeback("fc2", fc2.weight)

    # optimizer: conv + biases (excluding fc weights)
    opt_name = str(getattr(cfg, "OPTIMIZER", "sgd")).lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad and id(p) not in {id(fc1.weight), id(fc2.weight)}], lr=lr, weight_decay=getattr(cfg, "WEIGHT_DECAY", 0.0))
    elif opt_name == "adam":
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad and id(p) not in {id(fc1.weight), id(fc2.weight)}], lr=lr, weight_decay=getattr(cfg, "WEIGHT_DECAY", 0.0))
    else:
        optimizer = build_optimizer_conv_and_bias(model, lr=lr)

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
            T_max=getattr(cfg, "LR_SCHED_TMAX", epochs),
        )
    label_smoothing = getattr(cfg, "LABEL_SMOOTHING", 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # choose mode here for this script
    print(f"[MODE] UPDATE_MODE={mode}, MAX_PULSES_PER_STEP={fcl_state.max_pulses}, MEM_SCALE={mem_scale}, LS={label_smoothing}")

    history = {"train_acc": [], "train_loss": [], "test_acc": [], "test_loss": []}

    for epoch in range(1, epochs + 1):
        fcl_state.reset_pulse_stats()

        train_acc, train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, fcl_state, mode=mode)
        test_acc, test_loss = evaluate(model, test_loader, criterion, device)

        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["test_acc"].append(test_acc)
        history["test_loss"].append(test_loss)

        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:
                pass

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_acc={train_acc:.4f} test_acc={test_acc:.4f} "
            f"train_loss={train_loss:.4f} test_loss={test_loss:.4f}"
        )

        # checks
        weight_quant_check(model, "fc1")
        weight_quant_check(model, "fc2")

        # state check
        for name in ["fc1", "fc2"]:
            info = fcl_state.layers[name]
            ip, inn = info["idx_p"], info["idx_n"]
            print(f"[STATE-CHECK] {name}: idx_p[{ip.min()},{ip.max()}] idx_n[{inn.min()},{inn.max()}] scale={info['scale']:.3e}")
        print(f"[PULSE-CHECK] avg pulses/step (mean over batches): {fcl_state.pulse_stats():.3f}")

    tag = f"{cfg.UPDATE_MODE}_{cfg.RGB_PREPROCESS_MODE}_ov{cfg.RGB_OVERLAP}_p{fcl_state.max_pulses}"
    np.save(os.path.join(cfg.EXPORT_DIR, f"train_history_{tag}.npy"), history)
    return history


if __name__ == "__main__":
    train()
