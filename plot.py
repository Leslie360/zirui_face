import argparse
import glob
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_histories(paths: List[str], key: str) -> Dict[str, np.ndarray]:
	"""Load *.npy histories from explicit paths and return tag->series for given key."""
	out = {}
	for path in sorted(paths):
		try:
			hist = np.load(path, allow_pickle=True).item()
		except Exception:
			continue
		if not isinstance(hist, dict) or key not in hist:
			continue
		tag = os.path.splitext(os.path.basename(path))[0]
		out[tag] = np.array(hist[key], dtype=float)
	return out


def plot_histories(data: Dict[str, np.ndarray], out_path: str, title: str, key: str) -> None:
	if not data:
		print("No histories found to plot.")
		return

	max_len = max(len(v) for v in data.values())
	epochs = np.arange(1, max_len + 1)

	plt.figure(figsize=(8, 5))
	for tag, series in data.items():
		x = epochs[: len(series)]
		plt.plot(x, series, label=tag)

	plt.xlabel("Epoch")
	plt.ylabel(key)
	plt.title(title)
	plt.grid(True, alpha=0.3)
	plt.legend()
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	plt.tight_layout()
	plt.savefig(out_path, dpi=150)
	print(f"saved plot to {out_path}")


def export_txt(data: Dict[str, np.ndarray], out_dir: str, key: str) -> None:
	os.makedirs(out_dir, exist_ok=True)
	for tag, series in data.items():
		out_path = os.path.join(out_dir, f"{tag}_{key}.txt")
		with open(out_path, "w", encoding="utf-8") as f:
			for idx, val in enumerate(series, start=1):
				f.write(f"{idx}\t{val}\n")
		print(f"saved txt to {out_path}")


def main():
	parser = argparse.ArgumentParser(description="Plot specified npy histories and optionally export txt")
	parser.add_argument("--dir", default="exports", help="Folder containing npy histories (used with pattern)")
	parser.add_argument("--pattern", default="*.npy", help="Glob pattern under dir when --files not set")
	parser.add_argument("--files", nargs="+", help="Explicit npy files (overrides dir+pattern)")
	parser.add_argument("--file-list", help="Text file containing npy paths (one per line)")
	parser.add_argument("--key", default="test_acc", help="History key to plot/export")
	parser.add_argument("--output", default=os.path.join("exports", "plots", "histories.png"), help="Output png path")
	parser.add_argument("--title", default="Histories", help="Plot title")
	parser.add_argument("--txt-dir", help="If set, also export txt files with the chosen key")
	args = parser.parse_args()

	paths: List[str] = []
	# add files from list file
	if args.file_list:
		try:
			with open(args.file_list, "r", encoding="utf-8") as f:
				for line in f:
					line = line.strip()
					if line:
						paths.append(line)
		except FileNotFoundError:
			print(f"file list not found: {args.file_list}")
	# add explicit files
	if args.files:
		paths.extend(args.files)
	# fallback to dir+pattern
	if not paths:
		paths = glob.glob(os.path.join(args.dir, args.pattern))
	# deduplicate
	paths = list(dict.fromkeys(paths))

	data = load_histories(paths, key=args.key)
	plot_histories(data, args.output, title=args.title, key=args.key)

	if args.txt_dir:
		export_txt(data, args.txt_dir, key=args.key)


if __name__ == "__main__":
	main()
