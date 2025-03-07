import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# Constants
IMG_SIZE = 224  # Increased image size for better feature extraction
BATCH_SIZE = 32
NUM_CLASSES = 7
NUM_EPOCHS = 30
LEARNING_RATE = 0.001

# Class mapping
CLASS_NAMES = {
    'banana_healthy_leaf': 0,
    'black_sigatoka': 1,
    'yellow_sigatoka': 2,
    'panama_disease': 3,
    'moko_disease': 4,
    'insect_pest': 5,
    'bract_mosaic_virus': 6
}

# Inverse mapping
IDX_TO_CLASS = {v: k for k, v in CLASS_NAMES.items()}

# Set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 