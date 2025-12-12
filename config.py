"""Central configuration for memristor-CNN project.

Put all tunable parameters here so you can quickly change settings for
notebook debugging or full-run on server.
"""
import os

# Data / training
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3
NUM_WORKERS = 0  # set >0 on Linux/servers for speed; keep 0 on Windows notebooks

# Model
NUM_CLASSES = 10

# Memristor mapping
G_MIN = 1e-12
G_MAX = 1e-6
G_REF = (G_MIN + G_MAX) / 2
# LTP/LTD file names (relative to DATA_DIR)
LTP_LTD_CSV = 'ltp_ltd.csv'  # combined format (preferred)
LTP_RGB_CSV = 'ltp_rgb.csv'
LTD_RGB_CSV = 'ltd_rgb.csv'
LTP_NPY = 'ltp_rgb.npy'
LTD_NPY = 'ltd_rgb.npy'

# LTP split rows (for combined file)
LTP_COUNT = 31

# Mapping strategy: 'linear' (simple scale) or 'curve' (use device curves to compute pulses)
MAPPING_STRATEGY = 'curve'

# Weight update mode: 'unidirectional' or 'bidirectional'
WEIGHT_UPDATE_MODE = 'bidirectional'

# Hardware-in-loop toggle: when True, training loop will simulate writing pulses
# to memristor arrays after optimizer updates and export conductance snapshots.
HARDWARE_IN_LOOP = True

# Export settings
EXPORT_DIR = os.path.join(os.path.dirname(__file__), 'mapped')
EXPORT_EPOCH_NPY = True
EXPORT_EPOCH_CSV = False

# Color mapping strategy for assigning R/G/B curves to weights:
# 'round_robin' : cycle through r,g,b across matrix elements
# 'blocks'      : split input dimension into 3 contiguous blocks assigned to r,g,b
COLOR_MAPPING = 'round_robin'

# Debug / quick-run options
DEBUG_SMALL_SUBSET = False  # if True, use only a small subset of CIFAR for fast tests
