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
    BATCH_SIZE = 128           # memristor 版本收敛更快的一档 batch size
    EPOCHS = 200               # Increased to ensure convergence to >90%
    LR = 0.001                 # Conservative LR for AdamW + Memristor
    WEIGHT_DECAY = 1e-4        # Adjusted decay
    NUM_WORKERS = 0

    # Label smoothing to curb overfit / oscillation
    LABEL_SMOOTHING = 0.05

    # CIFAR10 是否用单通道版本（如果你用的是 CHPDataset 三通道就 False）
    SINGLE_CHANNEL = False

    # 输入量化 bit（data_loader 会用到）
    BITS = 4

    # 优化器与调度
    OPTIMIZER = "adamw"          # Revert to AdamW for stability with Memristor updates
    LR_SCHED_TYPE = "cosine"     # step | cosine
    LR_SCHED_STEP = 30
    LR_SCHED_GAMMA = 0.1
    LR_SCHED_TMAX = 100

    # 模型变体：base / strong / resnet18（memristor FCL 需要 classifier，故默认 strong）
    MODEL_VARIANT = "strong"

    # 数据增强
    STRONG_AUG = True

    # =============================
    # RGB preprocessing (train/test split)
    # =============================
    # 训练：默认理想 4bit，无串扰
    RGB_PRE_MODE_TRAIN = "4bit_ideal"
    RGB_OVERLAP_TRAIN = 0.0

    # 测试：默认 4bit_overlap，可通过列表扫描多档串扰
    RGB_PRE_MODE_TEST = "4bit_overlap"
    RGB_OVERLAP_TEST = 0.0

    # 扫描列表（test 用）
    RGB_OVERLAP_LIST = [0,0.1,0.3,0.5]   # 论文关键档位

    # Overlap 模型增强：alpha>1 放大串扰，gamma!=1 加非线性
    OVERLAP_ALPHA = 2.0   # Increased to 2.0 to force degradation at overlap=0.5
    OVERLAP_GAMMA = 1.0

    # 兼容旧字段（部分脚本仍读取 RGB_PREPROCESS_MODE / RGB_OVERLAP）
    RGB_PREPROCESS_MODE = RGB_PRE_MODE_TEST
    RGB_OVERLAP = RGB_OVERLAP_TEST

    # 是否在 dataloader 启动时打印一条预处理配置（方便核对）
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
    # mode-specific pulse caps（可覆盖 MAX_PULSES_PER_STEP）
    BIDIR_MAX_PULSES = 3
    UNIDIR_MAX_PULSES = 5

    # （你选的“平均曲线”）
    USE_AVG_CURVE = True
  
    # （固定 True；之后做 B 扩展再用）
    FCL_ONLY = True
    
    # 初始化策略：优先 match_float（把初始 float W 映射到差分态）
    #"match_float"| "midpoint"
    MEM_INIT = "match_float"            

    # 量化映射缩放：S = S_base * MEM_SCALE_*，用来调节 mem 更新幅度
    MEM_SCALE_DEFAULT = 1.0
    BIDIR_MEM_SCALE = 1.0
    UNIDIR_MEM_SCALE = 0.7

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

cfg = Config()
