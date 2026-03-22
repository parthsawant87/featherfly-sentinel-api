# config.py — SENTINEL Central Configuration
# EDIT: Change BASE_DIR to your actual project path
# NEVER import paths from anywhere except this file

import os
from pathlib import Path

# ── ROOT ──────────────────────────────────────────
BASE_DIR       = "/content/drive/MyDrive/SENTINEL"  # ← CHANGE THIS
DATA_DIR       = os.path.join(BASE_DIR, "data")
DATASET_RAW    = os.path.join(DATA_DIR, "dataset_raw")
DATASET_SPLIT  = os.path.join(DATA_DIR, "dataset_split")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
EXPORT_DIR     = os.path.join(BASE_DIR, "exports")
LOG_DIR        = os.path.join(BASE_DIR, "logs")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")
WAFER_SCAN_DIR = os.path.join(BASE_DIR, "wafer_scans")
DB_PATH        = os.path.join(BASE_DIR, "sentinel_results.db")

# ── CLASSES ───────────────────────────────────────
# Order must match your folder names alphabetically
CLASS_NAMES = [
    "BRIDGE",    # 0 — critical
    "CLEAN",     # 1 — pass
    "CMP",       # 2 — medium
    "CRACK",     # 3 — critical
    "LER",       # 4 — medium
    "OPEN",      # 5 — critical
    "OTHERS",    # 6 — review
    "PARTICLE",  # 7 — high
    "VIA",       # 8 — critical
]
NUM_CLASSES  = len(CLASS_NAMES)
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

# ── IMAGE ─────────────────────────────────────────
IMG_SIZE = 224
IMG_MEAN = (0.485, 0.456, 0.406)  # ImageNet mean
IMG_STD  = (0.229, 0.224, 0.225)  # ImageNet std

# ── TRAINING ──────────────────────────────────────
BATCH_SIZE          = 32
NUM_WORKERS         = 4
PHASE1_EPOCHS       = 10   # head only (backbone frozen)
PHASE1_LR           = 1e-3
PHASE2_EPOCHS       = 20   # full fine-tune (all layers)
PHASE2_LR           = 1e-4
WEIGHT_DECAY        = 1e-4
LABEL_SMOOTHING     = 0.1
EARLY_STOP_PATIENCE = 5
MIXUP_ALPHA         = 0.2   # 0.0 to disable

# ── DATASET SPLIT ─────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

# ── MODEL ─────────────────────────────────────────
BACKBONE             = "mobilenet_v3_large"
PRETRAINED           = True
DROPOUT              = 0.3
CONFIDENCE_THRESHOLD = 0.60  # below this → flag as LOW CONF

# ── EXPORT ────────────────────────────────────────
FP32_MODEL_NAME    = "sentinel_fp32.pth"
TFLITE_MODEL_NAME  = "sentinel_int8.tflite"
ONNX_MODEL_NAME    = "sentinel_mobilenet.onnx"
CALIB_SAMPLES      = 200   # images used for INT8 calibration

# ── YOLO ──────────────────────────────────────────
YOLO_MODEL    = "yolov8s-cls.pt"  # downloads automatically
YOLO_EPOCHS   = 30
YOLO_IMGSZ    = 224
YOLO_BATCH    = 32
YOLO_LR0      = 0.01
YOLO_PATIENCE = 7
YOLO_PROJECT  = "yolo_runs"
YOLO_NAME     = "sentinel_yolov8s"

# ── WAFER MAP ─────────────────────────────────────
WAFER_GRID_ROWS    = 10
WAFER_GRID_COLS    = 10
WAFER_RADIUS_MM    = 75   # 150mm wafer
WAFER_MAP_JSON     = "wafer_map.json"

# ── API ───────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

def create_dirs():
    for d in [DATASET_SPLIT, CHECKPOINT_DIR, EXPORT_DIR,
               LOG_DIR, RESULTS_DIR, WAFER_SCAN_DIR]:
        os.makedirs(d, exist_ok=True)

if __name__ == "__main__":
    create_dirs()
    print("[config] ✓ All directories created")
    print(f"[config] Classes: {CLASS_NAMES}")
