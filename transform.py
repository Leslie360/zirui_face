import argparse
import glob
import os
from typing import Any, Dict

import numpy as np


def save_array_txt(arr: np.ndarray, out_path: str) -> None:
    arr = np.asarray(arr)
    # flatten to 1D for consistency
    flat = arr.reshape(-1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savetxt(out_path, flat, fmt="%.8f")
    print(f"saved txt: {out_path} (shape={arr.shape})")


def save_array_csv(arr: np.ndarray, out_path: str, header: str = "value") -> None:
    arr = np.asarray(arr)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if arr.ndim == 1:
        data = np.column_stack((np.arange(1, arr.size + 1), arr))
        np.savetxt(out_path, data, fmt=["%d", "%.8f"], delimiter=",", header="epoch," + header, comments="")
    else:
        np.savetxt(out_path, arr, fmt="%.8f", delimiter=",", comments="")
    print(f"saved csv: {out_path} (shape={arr.shape})")


def try_combined_csv(series: Dict[str, np.ndarray], out_path: str) -> bool:
    """If all series are 1D and same length, save one CSV with columns."""
    if not series:
        return False
    lengths = {len(v.reshape(-1)) for v in series.values()}
    if len(lengths) != 1:
        return False
    length = lengths.pop()
    data = [np.arange(1, length + 1)] + [np.asarray(v).reshape(-1) for v in series.values()]
    stacked = np.column_stack(data)
    header = "epoch," + ",".join(series.keys())
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savetxt(out_path, stacked, fmt=["%d"] + ["%.8f"] * (stacked.shape[1] - 1), delimiter=",", header=header, comments="")
    print(f"saved csv: {out_path} (combined {len(series)} series)")
    return True


def convert_file(path: str, out_dir: str, csv_dir: str = None) -> None:
    base = os.path.splitext(os.path.basename(path))[0]
    try:
        obj: Any = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"skip {path}: load failed ({e})")
        return

    # handle dict-like payload (e.g., training history with lists)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1:
        try:
            obj = obj.item()
        except Exception:
            pass

    if isinstance(obj, dict):
        wrote = False
        collected = {}
        for k, v in obj.items():
            if isinstance(v, (np.ndarray, list, tuple)):
                arr = np.asarray(v)
                collected[k] = arr
                out_path = os.path.join(out_dir, f"{base}__{k}.txt")
                save_array_txt(arr, out_path)
                if csv_dir:
                    out_csv = os.path.join(csv_dir, f"{base}__{k}.csv")
                    save_array_csv(arr, out_csv, header=k)
                wrote = True
        if csv_dir and collected:
            combined_csv = os.path.join(csv_dir, f"{base}__combined.csv")
            try_combined_csv(collected, combined_csv)
        if not wrote:
            print(f"skip {path}: no array-like values in dict")
    elif isinstance(obj, (np.ndarray, list, tuple)):
        arr = np.asarray(obj)
        out_path = os.path.join(out_dir, f"{base}.txt")
        save_array_txt(arr, out_path)
        if csv_dir:
            out_csv = os.path.join(csv_dir, f"{base}.csv")
            save_array_csv(arr, out_csv, header="value")
    else:
        print(f"skip {path}: unsupported type {type(obj)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert npy files to txt (source kept)")
    parser.add_argument("--dir", default="result_final_0.01", help="Folder containing npy files")
    parser.add_argument("--pattern", default="*.npy", help="Glob pattern under dir")
    parser.add_argument("--out-dir", default="result_final_0.01/txt", help="Output folder for txt")
    parser.add_argument("--csv-dir", help="Optional output folder for csv")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not paths:
        print("no npy files matched")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    if args.csv_dir:
        os.makedirs(args.csv_dir, exist_ok=True)
    for p in paths:
        convert_file(p, args.out_dir, csv_dir=args.csv_dir)


if __name__ == "__main__":
    main()
