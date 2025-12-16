# config.py
import os

class Config:
    # =============================
    # Paths
    # =============================
    # 建议用相对路径，运行目录在项目根目录时可直接生效
    DATA_DIR = "data"
    EXPORT_DIR = "exports"   # 统一导出目录（npy/plots 都放这里）

    # 你的电导曲线文件
    LTP_LTD_PATH = os.path.join(DATA_DIR, "ltp_ltd.txt")

    FIRST_CONV_CHANNELS = 32
    SECOND_CONV_CHANNELS = 64
    THIRD_CONV_CHANNELS = 128

    # =============================
    # Dataset / Train
    # =============================
    NUM_CLASSES = 10
    BATCH_SIZE = 64
    EPOCHS = 100
    LR = 0.01
    WEIGHT_DECAY = 0.0
    NUM_WORKERS = 0

    # CIFAR10 是否用单通道版本（如果你用的是 CHPDataset 三通道就 False）
    SINGLE_CHANNEL = False

    # 输入量化 bit（你 data_loader 里会用到）
    BITS = 4

    # =============================
    # RGB pre-process (CH-P style)
    # =============================
    # 可选: "float" / "4bit_ideal" / "4bit_overlap"
    RGB_PREPROCESS_MODE = "4bit_overlap"

    # overlap 强度：0.0 表示理想分离；0.5 表示 50% overlap（论文对照常用）
    RGB_OVERLAP = 0.25

    # 扫描用 overlap 列表（用于 run_compare 一键跑多条曲线）不需要循环就空集
    RGB_OVERLAP_LIST = [0.0, 0.1, 0.3, 0.5]   # 先按论文最关键三档


    # 是否在 dataloader 启动时打印一条预处理配置（方便你核对）
    RGB_PREPROCESS_VERBOSE = True

    # =============================
    # Memristor / Conductance settings
    # （根据你 data/ltp_ltd.txt 统计出来的量级，按 abs(G) 适配）
    # =============================
    # 推荐值：略小于全局 min(|G|)，略大于全局 max(|G|)
    G_MIN = 8.0e-11
    G_MAX = 2.2e-09

    # 前 LTP_COUNT 行作为 LTP，剩余作为 LTD（你的文件 59 行数据时：31+28）
    LTP_COUNT = 31

    #是否启用 FCL 的器件离散更新（False 就回到 float baseline）
    USE_MEMRISTOR_FCL = True

    #单向/双向（run_compare 里由 mode 传入覆盖）
    #"unidir"| "bidir"
    UPDATE_MODE = "bidir"              

    # （默认 1；后续可改 3）
    MAX_PULSES_PER_STEP = 3

    # （你选的“平均曲线”）
    USE_AVG_CURVE = True
  
    # （固定 True；之后做 B 扩展再用）
    FCL_ONLY = True
    
    # 初始化策略：优先 match_float（把初始 float W 映射到差分态）
    #"match_float"| "midpoint"
    MEM_INIT = "match_float"            

    # RGB 映射策略（当 color="auto" 时生效）
    # round_robin / blocks
    COLOR_MAPPING = "round_robin"

    # =============================
    # Optional snapshots / plots
    # =============================
    SAVE_SNAPSHOTS = True
    SNAPSHOT_EPOCHS = [1, 10, 20, 30]

    # 权重可视化层配置
    WEIGHT_LAYER = "fc1"
    WEIGHT_NEURON_INDEX = 0

    # =============================
    # RGB preprocessing (train/test split)
    # =============================
    # 训练：固定 ideal（不串扰）
    RGB_PRE_MODE_TRAIN = "4bit_ideal"
    RGB_OVERLAP_TRAIN = 0.0

    # 测试：用 overlap 扫描（串扰退化）
    RGB_PRE_MODE_TEST = "4bit_overlap"
    RGB_OVERLAP_TEST = 0.0

    # 扫描列表（只用于 test）
    RGB_OVERLAP_LIST = [0.0, 0.1, 0.3, 0.5]

    # 量化位宽（RGB 4-bit）
    BITS = 4



cfg = Config()
