import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import cfg
from model import get_model
from data_loader import get_cifar10_loaders
from memristor_model import MemristorModel


import torch.nn.functional as F

# ============================================================
# 0a) Loss Functions
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both labels and mask')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / 0.07) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

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
    def __init__(self, g_states_color: dict, max_pulses_per_step: int = 1, color_mapping: str = "round_robin", device=None):
        self.device = device if device is not None else torch.device("cpu")
        # stack per-color states to allow fast indexing
        g_r, g_g, g_b = g_states_color["r"], g_states_color["g"], g_states_color["b"]
        min_len = min(g_r.size, g_g.size, g_b.size)
        self.N = int(min_len)
        
        # Convert to torch tensor on device
        g_stack_np = np.stack([
            g_r[:min_len],
            g_g[:min_len],
            g_b[:min_len],
        ], axis=0).astype(np.float32)  # shape (3,N)
        
        self.g_stack = torch.from_numpy(g_stack_np).to(self.device)
        
        self.g_min = float(self.g_stack.min())
        self.g_max = float(self.g_stack.max())
        self.g_range = max(self.g_max - self.g_min, 1e-30)
        
        # Calculate step scalar
        self.g_step_scalar = (self.g_stack[:, -1] - self.g_stack[:, 0]) / max(self.N - 1, 1)
        
        self.max_pulses = int(max_pulses_per_step)
        self.color_mapping = color_mapping

        # Variation parameters (default 0.0 means no variation)
        self.c2c_variation = 0.0  # Cycle-to-cycle variation (relative std dev)
        self.d2d_variation = 0.0  # Device-to-device variation (not implemented yet)
        self.stochastic_rounding = True # Use stochastic rounding for small updates

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
        # Create on device directly
        if strategy == "round_robin":
            # Using torch meshgrid or similar to create indices
            # (i * cols + j) % 3
            # i: (rows, 1), j: (1, cols)
            i = torch.arange(rows, device=self.device).view(-1, 1)
            j = torch.arange(cols, device=self.device).view(1, -1)
            color_idx = (i * cols + j) % 3
        elif strategy == "blocks":
            # block = int(j * 3 / max(1, cols))
            j = torch.arange(cols, device=self.device).view(1, -1)
            block = (j * 3 / max(1, cols)).long()
            block = torch.clamp(block, max=2)
            color_idx = block.expand(rows, cols)
        else:
            color_idx = torch.zeros(shape, dtype=torch.long, device=self.device)
        return color_idx.long()

    def init_layer_from_float(self, name: str, W_float: np.ndarray, scale_factor: float = 1.0):
        # W_float is numpy, convert to torch
        W = torch.from_numpy(W_float).float().to(self.device)
        mid = (self.N - 1) // 2

        w_abs = torch.abs(W)
        w_abs_p99 = torch.quantile(w_abs, 0.99).item()
        
        if w_abs_p99 < 1e-12:
            w_abs_p99 = float(torch.std(W) + 1e-6)
        S = (2.0 * w_abs_p99) / self.g_range
        S = S * max(scale_factor, 1e-12)
        S = max(S, 1e-12)

        color_idx = self._color_indices(W.shape, self.color_mapping)
        g_step = self.g_step_scalar[color_idx]

        deltaG = W / S
        steps = torch.round(deltaG / g_step).int()

        idx_p = torch.full_like(steps, mid)
        idx_n = torch.full_like(steps, mid)

        pos = steps > 0
        neg = steps < 0
        idx_p[pos] = idx_p[pos] + steps[pos]
        idx_n[neg] = idx_n[neg] + (-steps[neg])

        idx_p = torch.clamp(idx_p, 0, self.N - 1)
        idx_n = torch.clamp(idx_n, 0, self.N - 1)

        self.layers[name] = {
            "idx_p": idx_p,
            "idx_n": idx_n,
            "scale": float(S),
            "shape": W.shape,
            "color_idx": color_idx,
        }

    def _indices_to_weight(self, idx_p, idx_n, scale, color_idx):
        # All inputs are torch tensors
        Gp = self.g_stack[color_idx, idx_p]
        Gn = self.g_stack[color_idx, idx_n]
        return (Gp - Gn) * scale

    def writeback(self, name: str, W_tensor: torch.Tensor):
        info = self.layers[name]
        W = self._indices_to_weight(info["idx_p"], info["idx_n"], info["scale"], info["color_idx"])
        W_tensor.data.copy_(W)

    def update(self, name: str, grad_W: torch.Tensor, lr: float, mode: str):
        """
        Update idx_p/idx_n according to deltaW = -lr * grad.
        pulses computed from deltaG = |deltaW|/S and g_step.
        """
        info = self.layers[name]
        idx_p = info["idx_p"]
        idx_n = info["idx_n"]
        color_idx = info["color_idx"]
        S = float(info["scale"])

        # Ensure grad_W is on the correct device
        if grad_W.device != self.device:
            grad_W = grad_W.to(self.device)

        deltaW = (-lr) * grad_W
        deltaG = torch.abs(deltaW) / max(S, 1e-30)
        g_step = self.g_step_scalar[color_idx]
        
        # Calculate raw pulses needed
        raw_pulses = deltaG / torch.clamp(g_step, min=1e-30)
        
        # Add Cycle-to-Cycle Variation (noise on number of pulses or effective update?)
        if self.c2c_variation > 0:
            # Noise is proportional to the number of pulses
            noise = torch.randn_like(raw_pulses) * self.c2c_variation * torch.abs(raw_pulses)
            raw_pulses = raw_pulses + noise

        # Stochastic Rounding or Standard Rounding
        if self.stochastic_rounding:
            # floor + bernoulli(remainder)
            pulses_floor = torch.floor(raw_pulses)
            remainder = raw_pulses - pulses_floor
            mask = torch.rand_like(remainder) < remainder
            pulses = pulses_floor + mask.float()
            pulses = pulses.int()
        else:
            pulses = torch.round(raw_pulses).int()

        # Add Cycle-to-Cycle Variation (noise on number of pulses or effective update?)
        # Typically C2C affects the conductance change, but here we update by integer steps.
        # We can model it as noise on the 'raw_pulses' before rounding.
        # (Variation logic moved before rounding)

        pulses = torch.clamp(pulses, 0, self.max_pulses)

        # pulse stats
        self._pulse_sum += float(torch.mean(pulses.float()))
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

        # In-place update of info dict tensors
        info["idx_p"] = torch.clamp(idx_p, 0, self.N - 1)
        info["idx_n"] = torch.clamp(idx_n, 0, self.N - 1)


def build_optimizer_conv_and_bias(model: nn.Module, lr: float):
    """
    Optimizer updates:
      - all parameters except classifier Linear.weight (we hard-update those with memristor)
      - classifier Linear.bias can still be optimized (keeps baseline performance)
    """
    fc1, fc2 = get_fc_linears(model)
    fc_weight_ids = {id(fc1.weight), id(fc2.weight)}
    
    opt_type = getattr(cfg, "OPTIMIZER", "sgd").lower()

    params = []
    for p in model.parameters():
        if id(p) in fc_weight_ids:
            continue
        params.append(p)

    if opt_type == "adamw":
        opt = optim.AdamW(params, lr=lr, weight_decay=getattr(cfg, "WEIGHT_DECAY", 1e-4))
    elif opt_type == "adam":
        opt = optim.Adam(params, lr=lr, weight_decay=getattr(cfg, "WEIGHT_DECAY", 1e-4))
    else:
        # SGD default
        opt = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=getattr(cfg, "WEIGHT_DECAY", 5e-4))
    return opt


# ============================================================
# 4) Train / Eval
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device, fcl_state: MemFCLState, mode: str):
    model.train()
    correct, total = 0, 0
    running_loss = 0.0

    # Initialize custom losses
    supcon_criterion = SupConLoss().to(device)
    focal_criterion = FocalLoss().to(device)

    fc1, fc2 = get_fc_linears(model)

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        
        # Modified to get features for SupConLoss
        if isinstance(model, get_model("strong", num_classes=10).__class__): # Hacky check or just try/except
             try:
                 outputs, features = model(inputs, return_features=True)
                 loss_cls = focal_criterion(outputs, targets)
                 
                 # Normalize features for SupCon
                 features = F.normalize(features, dim=1)
                 features_sup = features.unsqueeze(1) # SupCon expects [batch, n_views, dim]
                 loss_supcon = supcon_criterion(features_sup, targets)
                 
                 # Increased SupCon weight to encourage better feature separation
                 loss = loss_cls + 0.5 * loss_supcon
             except TypeError:
                 # Fallback for base model which might not support return_features
                 outputs = model(inputs)
                 loss = criterion(outputs, targets)
        else:
             outputs = model(inputs)
             loss = criterion(outputs, targets)

        loss.backward()

        # Check for NaN loss
        if torch.isnan(loss):
            print("Warning: Loss is NaN. Skipping step.")
            optimizer.zero_grad()
            continue

        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 1) update conv+others (excluding fc weights)
        optimizer.step()

        # 2) memristor update for fc weights
        if fcl_state is not None:
            # grad -> torch tensor (keep on device)
            g1 = fc1.weight.grad
            g2 = fc2.weight.grad
            
            # Gradients are already clipped by clip_grad_norm_ above
            if g1 is not None:
                fcl_state.update("fc1", g1, lr=optimizer.param_groups[0]["lr"], mode=mode)
            if g2 is not None:
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
    
    # Weighted Loss to handle Class 3 (Cat) and 5 (Dog) confusion
    # Classes: 0:Plane, 1:Car, 2:Bird, 3:Cat, 4:Deer, 5:Dog, 6:Frog, 7:Horse, 8:Ship, 9:Truck
    # We increase weights for 3 and 5.
    class_weights = torch.ones(10).to(device)
    class_weights[3] = 1.5
    class_weights[5] = 1.5
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

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
