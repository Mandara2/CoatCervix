"""
Bloque A3 de la revision PeerJ: metricas de validacion de los 5 folds de EfficientNet-B0.

Solo inferencia. Usa los pesos de cross/effcross.py (Efficient/Checkpoints_KFold/efficient_fold_k.pt),
reconstruye sus folds (StratifiedKFold, 5 folds, shuffle, semilla 42, sobre las 541 de train) y su
preprocesamiento (filtro unsharp leido del propio effcross.py y normalizacion de ImageNet, sin resize).
Evalua cada peso en su fold de validacion y en el test.
Control: las metricas en el test deben coincidir con Efficient/resultados_kfold_efficient.csv.

  CUDA_VISIBLE_DEVICES=<gpu> /data/rtabares/alzheimer/cervical_clean/bin/python -u bloque_a3_efficientnet_cv.py
"""
import ast
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from torchvision import models

BASE = "/data/home/rtabares/Cervical"
DATA = os.path.join(BASE, "Cervical_clean")
SCRIPT = os.path.join(BASE, "cross", "effcross.py")
CKPT = os.path.join(BASE, "Efficient", "Checkpoints_KFold")

# Toma la funcion apply_unsharp tal como esta en effcross.py, sin ejecutar el resto del script.
arbol = ast.parse(open(SCRIPT, encoding="utf-8").read())
funcion = next(n for n in arbol.body if isinstance(n, ast.FunctionDef) and n.name == "apply_unsharp")
espacio = {"np": np, "cv2": cv2}
exec(compile(ast.Module(body=[funcion], type_ignores=[]), SCRIPT, "exec"), espacio)
apply_unsharp = espacio["apply_unsharp"]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preparar(X):
    Xu = np.array([apply_unsharp(im) for im in X])
    return torch.tensor(((Xu - MEAN) / STD).astype(np.float32)).permute(0, 3, 1, 2)


@torch.no_grad()
def predecir(model, X, device, bs=16):
    return torch.cat([model(X[i:i + bs].to(device)).argmax(1).cpu() for i in range(0, len(X), bs)]).numpy()


def metricas(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return accuracy_score(y, p), pr, rc, f1


def main():
    device = torch.device("cuda")
    X_train = np.load(os.path.join(DATA, "X_train.npy")).astype(np.float32)
    y_train = np.load(os.path.join(DATA, "y_train.npy"))
    X_test = np.load(os.path.join(DATA, "X_test.npy")).astype(np.float32)
    y_test = np.load(os.path.join(DATA, "y_test.npy"))
    Xtr, Xte = preparar(X_train), preparar(X_test)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(Xtr.numpy(), y_train)

    val, test = [], []
    for k, (tr, va) in enumerate(folds, start=1):
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
        model.load_state_dict(torch.load(os.path.join(CKPT, f"efficient_fold_{k}.pt"), map_location="cpu"))
        model.to(device).eval()
        mv = metricas(y_train[va], predecir(model, Xtr[va], device))
        mt = metricas(y_test, predecir(model, Xte, device))
        val.append(mv)
        test.append(mt)
        print(f"fold {k} ({len(va)} val) | VAL acc {mv[0]:.4f} P {mv[1]:.4f} R {mv[2]:.4f} F1 {mv[3]:.4f} "
              f"| TEST acc {mt[0]:.4f} P {mt[1]:.4f} R {mt[2]:.4f} F1 {mt[3]:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()

    for nombre, r in (("VALIDACION", val), ("TEST", test)):
        m, s = np.mean(r, 0), np.std(r, 0)
        print(f"\n{nombre}: " + " | ".join(f"{n} {m[i]:.4f} +- {s[i]:.4f}"
                                          for i, n in enumerate(["acc", "precision", "recall", "f1"])))


if __name__ == "__main__":
    main()
