# config.py
import os

# Paths (change these to match your setup)
DATA_ROOT = "/path/to/iam"  # root containing images + mapping file
IAM_MAPPING = os.path.join(DATA_ROOT, "ground_truth.txt")  # user-supplied map: "imgname\ttext"
MODEL_DIR = "./models"
LOG_DIR = "./logs"

# Training
BATCH_SIZE = 16
NUM_WORKERS = 4
LR = 1e-4
EPOCHS = 50
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# Image processing
IMG_HEIGHT = 64  # fixed height for CRNN
IMG_MAX_WIDTH = 1600  # max width (we will pad or scale down)
MEAN = [0.5]
STD = [0.5]

# Model
NUM_CHANNELS = 1  # grayscale images
NUM_CLASSES = None  # set later after building charset
BLANK_IDX = 0

# Model hyperparams
CNN_OUT_CHANNELS = 512
LSTM_HIDDEN = 256
LSTM_LAYERS = 2
ATTN_LAYERS = 2
ATTN_HEADS = 8
DROPOUT = 0.1

# Misc
SEED = 42
SAVE_EVERY = 1
