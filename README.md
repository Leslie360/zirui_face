Project: Memristor-based CIFAR-10 CNN Emulation

Contents:
- memristor_model.py: simple model to map weights to G+/G- and apply LTP/LTD pulses
- model.py: PyTorch CNN (3 conv + 3 pool + 2 FC)
- data_loader.py: CIFAR-10 loader with CH-P-like RGB separation and 4-bit quantization
- train_memristor_cnn.py: training loop and mapping of FC weights to memristor conductances
- requirements.txt: Python dependencies

Quick start:

1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Run training (will download CIFAR-10):

```bash
python train_memristor_cnn.py
```

Notes:
- This repository provides a software emulation to map trained FC layer weights
  to memristor conductance pairs G+ and G-. The mapping in `memristor_model.py` is
  intentionally simple and meant as a starting point — replace it with empirical
  LTP/LTD data arrays (your measured `ltp`, `ltd`) to more accurately simulate
  the hardware behavior.
- The CH-P pre-processing implemented in `data_loader.py` quantizes each RGB
  channel to 4-bit resolution to mimic the color separation described in the
  paper. Adjust `bits` to match your device capabilities.
