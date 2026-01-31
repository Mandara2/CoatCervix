# config/settings.py
import os
import torch

# Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hiperparámetros
num_classes = 3
batch_size = 16
epochs = 50
lr = 1e-4
class_names = ["Type 1", "Type 2", "Type 3"]
SEED = 42

# Nombre del modelo
MODEL_NAME = "NewCoatROC"

# Paths principales
BASE_PATH = "/data/home/rtabares/Cervical"
DATA_PATH = os.path.join(BASE_PATH, "Cervical_clean")
RESULTS_PATH = os.path.join(BASE_PATH, "Coat", "new_coat", "results")

GRAPHICS_PATH = os.path.join(RESULTS_PATH, "graphics")
CSV_PATH = os.path.join(RESULTS_PATH, "results.csv")
SAVE_GRAPHICS_PATH = os.path.join(GRAPHICS_PATH, MODEL_NAME)

# Imagen objetivo
target_size = (224, 224)
epsilon = 1e-8
