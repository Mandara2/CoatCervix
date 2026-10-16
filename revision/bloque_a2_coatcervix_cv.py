"""
Bloque A2 de la revision PeerJ: evalua en el test los 5 modelos de la validacion cruzada de CoatCervix.

Solo inferencia. Usa los pesos de la corrida del 2026-02-07 (CoatCV_Multiclas_20260207_*.pth, script
Coat/new_model_binary_cross_validation), que reproduce los mismos valores por fold que las corridas del
2026-02-05. Reconstruye cada fold (StratifiedKFold, semilla 42, sobre las 541 de train) y su
preprocesamiento (resize, media y sd del train del fold antes de CLAHE, CLAHE, normalizacion).
Control: la accuracy de validacion de cada checkpoint debe coincidir con la "Mejor Accuracy" del log.

  CUDA_VISIBLE_DEVICES=<gpu> /data/rtabares/alzheimer/cervical_clean/bin/python -u bloque_a2_coatcervix_cv.py
"""
import csv
import glob
import os
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

BASE = "/data/home/rtabares/Cervical"
DATA = os.path.join(BASE, "Cervical_clean")
CV_DIR = os.path.join(BASE, "Coat", "new_model_binary_cross_validation")
PESOS = sorted(glob.glob(os.path.join(BASE, "Coat", "new_coat_binary", "results", "graphics",
                                      "CoatCV_Multiclas", "CoatCV_Multiclas_20260207_*.pth")))
OUT = os.path.join(BASE, "revision_peerj", "bloque_a2")

sys.path.insert(0, CV_DIR)
from data.preprocessing import apply_clahe, resize_images  # noqa: E402
from config import settings  # noqa: E402
from models.custom_coat import CustomCoaT  # noqa: E402

TARGET_SIZE = settings.target_size


def cargar(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for k in ("model_state_dict", "state_dict", "model"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    if not isinstance(sd, dict):
        sd = sd.state_dict()
    return {k.replace("module.", "", 1): v for k, v in sd.items()}


def preparar(X, mean, std):
    Xc = np.array([apply_clahe(img) for img in resize_images(X, TARGET_SIZE)])
    return torch.from_numpy(((Xc - mean) / (std + 1e-8)).transpose(0, 3, 1, 2).astype(np.float32))


@torch.no_grad()
def predecir(model, X, device, bs=32):
    return torch.cat([torch.softmax(model(X[i:i + bs].to(device)), 1).cpu() for i in range(0, len(X), bs)]).numpy()


def main():
    if len(PESOS) != 5:
        raise SystemExit(f"Se esperaban 5 pesos y hay {len(PESOS)}: {PESOS}")
    device = torch.device("cuda")
    os.makedirs(os.path.join(OUT, "predicciones"), exist_ok=True)
    X_train = np.load(os.path.join(DATA, "X_train.npy"))
    y_train = np.load(os.path.join(DATA, "y_train.npy"))
    X_test = np.load(os.path.join(DATA, "X_test.npy"))
    y_test = np.load(os.path.join(DATA, "y_test.npy"))
    y_train = y_train.argmax(1) if y_train.ndim > 1 else y_train.astype(int)
    y_test = y_test.argmax(1) if y_test.ndim > 1 else y_test.astype(int)
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train))

    filas, res = [], []
    for k, (path, (tr, va)) in enumerate(zip(PESOS, folds), start=1):
        Xr = resize_images(X_train[tr], TARGET_SIZE)
        mean, std = Xr.mean(axis=(0, 1, 2)), Xr.std(axis=(0, 1, 2))
        model = CustomCoaT(pretrained=False)
        model.load_state_dict(cargar(path), strict=True)
        model.to(device).eval()
        val_acc = accuracy_score(y_train[va], predecir(model, preparar(X_train[va], mean, std), device).argmax(1))
        probs = predecir(model, preparar(X_test, mean, std), device)
        pred = probs.argmax(1)
        p, r, f, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
        acc = accuracy_score(y_test, pred)
        res.append((acc, p, r, f))
        filas.append(["CoatCervix", k, os.path.basename(path), f"{val_acc:.4f}", f"{acc:.4f}", f"{p:.4f}", f"{r:.4f}", f"{f:.4f}"])
        with open(os.path.join(OUT, "predicciones", f"CoatCervix_cv_fold{k}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["idx", "y_true", "y_pred", "p_type1", "p_type2", "p_type3"])
            for i in range(len(y_test)):
                w.writerow([i, y_test[i], pred[i]] + [f"{x:.6f}" for x in probs[i]])
        print(f"fold {k}: val_acc {val_acc:.4f} | test acc {acc:.4f} P {p:.4f} R {r:.4f} F1 {f:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()

    with open(os.path.join(OUT, "metricas_por_fold.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["modelo", "fold", "peso", "val_acc", "test_acc", "test_precision", "test_recall", "test_f1"])
        w.writerows(filas)
    m, s = np.mean(res, 0), np.std(res, 0)
    print("\nCoatCervix (5 modelos de la validacion cruzada) en el test:")
    print(" | ".join(f"{n} {m[i]:.4f} +- {s[i]:.4f}" for i, n in enumerate(["acc", "precision", "recall", "f1"])))


if __name__ == "__main__":
    main()
