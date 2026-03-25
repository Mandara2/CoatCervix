# config/settings.py
import os
import torch

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_classes = 3
batch_size = 16
epochs = 50
lr = 1e-4
class_names = ["Type 1", "Type 2", "Type 3"]
SEED = 42

# model name
MODEL_NAME = "CoatCervix"

# Paths configuration
BASE_PATH = "Principal path"
DATA_PATH = os.path.join(BASE_PATH, "Cervical_clean")
RESULTS_PATH = os.path.join(BASE_PATH, "Coat", "new_coat", "results")

GRAPHICS_PATH = os.path.join(RESULTS_PATH, "graphics")
CSV_PATH = os.path.join(RESULTS_PATH, "results.csv")
SAVE_GRAPHICS_PATH = os.path.join(GRAPHICS_PATH, MODEL_NAME)

# Image configuration
target_size = (224, 224)
epsilon = 1e-8
