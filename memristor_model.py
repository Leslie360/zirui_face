import numpy as np
import os

from config import cfg  # ✅ 统一 cfg 实例风格


class MemristorModel:
    """Simple memristor conductance updater using LTP/LTD curves.

    This module loads empirical LTP/LTD arrays (or synthetic ones) and
    provides methods to map a desired weight change to pulse counts and
    update positive/negative device conductances G+ and G- accordingly.
    """

    def __init__(self, ltp=None, ltd=None, g_min=1e-6, g_max=1e-4):
        # ltp/ltd should be arrays mapping pulse index -> conductance
        # If not provided, synthesize simple exponential curves
        self.g_min = g_min
        self.g_max = g_max

        if ltp is None:
            self.ltp = self._synth_curve(up=True)
        else:
            self.ltp = np.array(ltp, dtype=float)

        if ltd is None:
            self.ltd = self._synth_curve(up=False)
        else:
            self.ltd = np.array(ltd, dtype=float)

        # default mode (can be overridden by caller)
        self.mode = "bidirectional"

    def _synth_curve(self, up=True, n=256):
        x = np.arange(n)
        if up:
            # increasing saturation
            return self.g_min + (self.g_max - self.g_min) * (1 - np.exp(-x / (n / 10)))
        else:
            # decreasing
            return self.g_max - (self.g_max - self.g_min) * (1 - np.exp(-x / (n / 10)))

    @staticmethod
    def load_rgb_csv(path):
        """Load a CSV (or whitespace delimited) file containing three columns for
        blue/green/red conductance values. Returns a dict with keys 'r','g','b'
        and 1D numpy arrays.

        Accepts files with header names like 'blue,green,red' or plain 3-column numeric files.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        try:
            data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
            names = data.dtype.names
            if names is not None:
                # look for common names
                lower = [n.lower() for n in names]
                out = {}
                if "red" in lower or "r" in lower:
                    idx = lower.index("red") if "red" in lower else lower.index("r")
                    out["r"] = data[names[idx]].astype(float)
                if "green" in lower or "g" in lower:
                    idx = lower.index("green") if "green" in lower else lower.index("g")
                    out["g"] = data[names[idx]].astype(float)
                if "blue" in lower or "b" in lower:
                    idx = lower.index("blue") if "blue" in lower else lower.index("b")
                    out["b"] = data[names[idx]].astype(float)
                if len(out) == 3:
                    return out
        except Exception:
            pass

        # fallback: try whitespace or comma separated numeric file
        arr = np.loadtxt(path, delimiter=",")
        if arr.ndim == 1 and arr.size % 3 == 0:
            arr = arr.reshape(-1, 3)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return {
                "r": arr[:, 0].astype(float),
                "g": arr[:, 1].astype(float),
                "b": arr[:, 2].astype(float),
            }
        raise ValueError("Cannot parse RGB conductance CSV format: " + path)

    @staticmethod
    def load_ltp_ltd_csv(path, ltp_count=31):
        """Load CSV where header has color columns and the first ltp_count rows
        are LTP (pulse indices 1..ltp_count) and the remaining rows are LTD.

        Returns (ltp_rgb_dict, ltd_rgb_dict)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        # Try reading the header line with robust encoding fallback
        encodings = ["utf-8", "latin1", "gbk"]
        header = None
        data_lines = None
        for enc in encodings:
            try:
                with open(path, "r", encoding=enc, errors="replace") as f:
                    all_lines = f.readlines()
                if len(all_lines) < 2:
                    raise ValueError("ltp_ltd CSV must have header + data rows")
                header = all_lines[0].strip()
                data_lines = all_lines[1:]
                break
            except Exception:
                header = None
                data_lines = None
                continue
        if header is None or data_lines is None:
            raise ValueError("Could not read ltp_ltd CSV with supported encodings")

        # parse header to determine column order (expect some variant of b/g/r)
        import re

        cols = [c.strip().lower() for c in re.split(r"[, \t\s]+", header) if c.strip() != ""]
        idx_map = {}
        for i, name in enumerate(cols):
            if name in ("b", "blue"):
                idx_map["b"] = i
            elif name in ("g", "green"):
                idx_map["g"] = i
            elif name in ("r", "red"):
                idx_map["r"] = i

        # fallback: assume first three columns correspond to b,g,r
        if set(idx_map.keys()) != {"b", "g", "r"}:
            idx_map = {"b": 0, "g": 1, "r": 2}

        # load numeric data from data_lines robustly
        num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
        data = []
        for line in data_lines:
            line = line.strip()
            if line == "":
                continue
            nums_str = num_re.findall(line)
            if len(nums_str) < 3:
                parts = [p.strip() for p in line.split(",") if p.strip() != ""]
                nums_try = []
                for p in parts:
                    m = num_re.search(p)
                    if m:
                        nums_try.append(m.group(0))
                if len(nums_try) < 3:
                    continue
                nums_str = nums_try
            try:
                needed = len(cols)
                if len(nums_str) < needed:
                    continue
                nums = [float(x) for x in nums_str[:needed]]
            except Exception:
                continue
            data.append(nums)

        if len(data) == 0:
            raise ValueError("No numeric data found in ltp_ltd CSV")

        arr2 = np.array(data, dtype=float)
        if arr2.ndim == 1:
            arr2 = arr2.reshape(1, -1)
        if arr2.shape[1] < 3:
            raise ValueError("Expected at least 3 columns of numeric data")

        b = arr2[:, idx_map["b"]].astype(float)
        g = arr2[:, idx_map["g"]].astype(float)
        r = arr2[:, idx_map["r"]].astype(float)

        def split_vals(v):
            v = np.array(v).astype(float)
            if v.size <= ltp_count:
                return v, np.array([])
            return v[:ltp_count].copy(), v[ltp_count:].copy()

        ltp_r, ltd_r = split_vals(r)
        ltp_g, ltd_g = split_vals(g)
        ltp_b, ltd_b = split_vals(b)

        ltp = {"r": ltp_r, "g": ltp_g, "b": ltp_b}
        ltd = {"r": ltd_r, "g": ltd_g, "b": ltd_b}
        return ltp, ltd

    def set_color_curves(self, ltp_rgb: dict, ltd_rgb: dict = None):
        """Set per-color LTP and optional LTD curves."""
        self.color_ltp = {k: np.array(v, dtype=float) for k, v in ltp_rgb.items()}
        if ltd_rgb is not None:
            self.color_ltd = {k: np.array(v, dtype=float) for k, v in ltd_rgb.items()}
        else:
            self.color_ltd = None

    def get_color_curve(self, color: str, up=True):
        """Return the per-color curve array for 'r'/'g'/'b'. If not present,
        fall back to the generic curve."""
        color = color.lower()[0]
        if hasattr(self, "color_ltp") and self.color_ltp is not None and up:
            return self.color_ltp.get(color, self.ltp)
        if hasattr(self, "color_ltd") and self.color_ltd is not None and (not up):
            return self.color_ltd.get(color, self.ltd)
        return self.ltp if up else self.ltd

    def pulses_for_conductance_change(self, current_g, target_g, color="r", up=True):
        """Compute number of pulses to go from current_g to target_g using the per-color curve."""
        curve = self.get_color_curve(color, up=up)
        idx_cur = int((np.abs(curve - current_g)).argmin())
        target_g_clipped = float(np.clip(target_g, curve.min(), curve.max()))
        idx_target = int((np.abs(curve - target_g_clipped)).argmin())
        pulses = idx_target - idx_cur
        return int(pulses)

    def apply_weight_change(self, g_p, g_n, delta_w, color="r", mode="bidirectional"):
        """Apply a signed weight change delta_w by updating G+ and/or G-."""
        desired_delta_g = float(delta_w) * (self.g_max - self.g_min)

        if mode == "bidirectional":
            if delta_w > 0:
                target_g_p = float(np.clip(g_p + desired_delta_g, self.g_min, self.g_max))
                p = self.pulses_for_conductance_change(g_p, target_g_p, color=color, up=True)
                g_p_new = self.apply_pulses(g_p, p, polarity="pos")
                return g_p_new, g_n, p, 0
            elif delta_w < 0:
                target_g_n = float(np.clip(g_n + abs(desired_delta_g), self.g_min, self.g_max))
                p = self.pulses_for_conductance_change(g_n, target_g_n, color=color, up=True)
                g_n_new = self.apply_pulses(g_n, p, polarity="pos")
                return g_p, g_n_new, 0, p
            else:
                return g_p, g_n, 0, 0

        # unidirectional
        if delta_w > 0:
            target_g_p = float(np.clip(g_p + desired_delta_g, self.g_min, self.g_max))
            p = self.pulses_for_conductance_change(g_p, target_g_p, color=color, up=True)
            g_p_new = self.apply_pulses(g_p, p, polarity="pos")
            return g_p_new, g_n, p, 0
        elif delta_w < 0:
            target_g_p = float(np.clip(g_p - abs(desired_delta_g), self.g_min, self.g_max))
            p = self.pulses_for_conductance_change(g_p, target_g_p, color=color, up=False)
            g_p_new = self.apply_pulses(g_p, p, polarity="neg")
            return g_p_new, g_n, 0, p
        else:
            return g_p, g_n, 0, 0

    def apply_weight_matrix_changes(self, Gp, Gn, deltaW, color="r", mode="bidirectional"):
        """Apply a matrix of weight deltas (same shape as Gp/Gn) to devices."""
        Gp_new = Gp.copy()
        Gn_new = Gn.copy()
        p_p = np.zeros_like(Gp, dtype=int)
        p_n = np.zeros_like(Gn, dtype=int)

        rows, cols = deltaW.shape

        # ✅ 统一 cfg：从 cfg.COLOR_MAPPING 读取
        if color is None or color == "auto":
            strategy = getattr(cfg, "COLOR_MAPPING", "round_robin")

            for i in range(rows):
                for j in range(cols):
                    if strategy == "round_robin":
                        idx = (i * cols + j) % 3
                        col = ["r", "g", "b"][idx]
                    elif strategy == "blocks":
                        block = int(j * 3 / max(1, cols))
                        col = ["r", "g", "b"][min(block, 2)]
                    else:
                        col = "r"

                    dw = float(deltaW[i, j])
                    gp, gn = float(Gp_new[i, j]), float(Gn_new[i, j])
                    gp_new, gn_new, pp, pn = self.apply_weight_change(gp, gn, dw, color=col, mode=mode)
                    Gp_new[i, j] = gp_new
                    Gn_new[i, j] = gn_new
                    p_p[i, j] = pp
                    p_n[i, j] = pn

            return Gp_new, Gn_new, p_p, p_n

        # fixed color for all elements
        for i in range(rows):
            for j in range(cols):
                dw = float(deltaW[i, j])
                gp, gn = float(Gp_new[i, j]), float(Gn_new[i, j])
                gp_new, gn_new, pp, pn = self.apply_weight_change(gp, gn, dw, color=color, mode=mode)
                Gp_new[i, j] = gp_new
                Gn_new[i, j] = gn_new
                p_p[i, j] = pp
                p_n[i, j] = pn

        return Gp_new, Gn_new, p_p, p_n

    def pulses_for_delta(self, delta_g, polarity="pos"):
        """Estimate number of pulses required to change conductance by delta_g."""
        curve = self.ltp if polarity == "pos" else self.ltd
        diffs = np.diff(curve, prepend=curve[0])
        if delta_g == 0:
            return 0
        target = abs(delta_g)
        cum = np.cumsum(np.abs(diffs))
        idx = np.searchsorted(cum, target)
        return int(idx) if idx < len(cum) else len(cum)

    def apply_pulses(self, g, pulses, polarity="pos"):
        """Apply pulses to a conductance value and return new conductance."""
        curve = self.ltp if polarity == "pos" else self.ltd
        idx = (np.abs(curve - g)).argmin()
        new_idx = np.clip(idx + pulses, 0, len(curve) - 1)
        return float(curve[new_idx])

    def map_weight_to_conductances(self, w, g_ref=5e-5):
        """Map a signed weight to a pair (G+, G-) with difference W = G+ - G-."""
        half = max(self.g_min, min(self.g_max, g_ref))
        if w >= 0:
            g_p = np.clip(half + abs(w) * (self.g_max - half), self.g_min, self.g_max)
            g_n = half
        else:
            g_n = np.clip(half + abs(w) * (self.g_max - half), self.g_min, self.g_max)
            g_p = half
        return float(g_p), float(g_n)

    def conductances_to_weight(self, Gp, Gn):
        """从Gp和Gn计算权重（W = Gp - Gn）"""
        return Gp - Gn
