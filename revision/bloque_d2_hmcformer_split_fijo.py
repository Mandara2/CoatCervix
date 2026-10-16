"""
Bloque D2 de la revision PeerJ: una corrida de la replica de HMCFormer con el split fijo de CoatCervix.

Para la Tabla 6 (una corrida por modelo en el split fijo). Reutiliza HMCFormer_train_only.py (la copia
corregida del bloque D) sin cambiar el entrenamiento: misma arquitectura, epocas, optimizador,
augmentation y seleccion del mejor epoch. El split es el mismo que usa CoatCervix: train_test_split
estratificado, test_size=0.2, random_state=42, sobre las 541 de train (432 para entrenar, 109 para
validar). El mejor checkpoint se evalua en las 225 de test.

  cd ~/Cervical/revision_peerj/hmcformer
  CUDA_VISIBLE_DEVICES=<gpu> /data/rtabares/alzheimer/cervical_clean/bin/python -u bloque_d2_hmcformer_split_fijo.py
"""
import importlib.util
import os

import numpy as np
from sklearn.model_selection import train_test_split

RUTA = os.path.expanduser("~/Cervical/revision_peerj/hmcformer/HMCFormer_train_only.py")
spec = importlib.util.spec_from_file_location("hmc", RUTA)
hm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hm)

hm.MODEL_NAME = "HMCFormer_split_fijo"
hm.SAVE_GRAPHICS_PATH = os.path.join(hm.GRAPHICS_PATH, hm.MODEL_NAME)
hm.ensure_dir(hm.SAVE_GRAPHICS_PATH)

X, y = hm.load_npy_data(hm.X_TRAIN_NPY, hm.X_TEST_NPY, hm.Y_TRAIN_NPY, hm.Y_TEST_NPY)
etiquetas = y.argmax(1) if y.ndim > 1 else y
idx = np.arange(len(X))
tr, va = train_test_split(idx, test_size=0.2, stratify=etiquetas, random_state=42)
print(f"Split fijo: train {len(tr)} | val {len(va)} | test {len(hm.TEST_X)}", flush=True)

config = {
    "batch_size": hm.BATCH_SIZE,
    "lr": hm.LR,
    "epochs": hm.EPOCHS,
    "num_classes": hm.NUM_CLASSES,
    "transformer_name": "swin_small_patch4_window7_224",
    "cnn_name": "efficientnet_b0",
    "feat_dim": 512,
    "weight_decay": hm.WEIGHT_DECAY,
}
hm.train_fold(0, tr, va, X, y, config, 42)
print("Resultado en", os.path.join(hm.SAVE_GRAPHICS_PATH, "test_por_fold.csv"))
