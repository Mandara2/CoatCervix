"""
Bloque A de la revision PeerJ: evaluacion en el test de los baselines de la Tabla 7.

Solo inferencia: no entrena ni modifica ningun peso. Para cada arquitectura usa los
5 pesos de la validacion cruzada del 2026-02-01 (scripts models_cross_validation_*),
que nunca vieron el conjunto de prueba, y calcula:
  1. Metricas en el test (225 imagenes) por fold y su media +- desviacion.
  2. Predicciones imagen por imagen (para la prueba de significancia).
  3. Accuracy en el fold de validacion (control: debe coincidir con el log original).
  4. Parametros, GMACs, latencia y VRAM de cada arquitectura (batch 1, 224 x 224).

Correr con el ambiente de entrenamiento:
  CUDA_VISIBLE_DEVICES=<gpu libre> /data/rtabares/alzheimer/cervical_clean/bin/python bloque_a_baselines.py
"""
import csv
import glob
import os
import sys
import time

import numpy as np
import timm
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from torch.utils.flop_counter import FlopCounterMode

BASE = "/data/home/rtabares/Cervical"
DATA = os.path.join(BASE, "Cervical_clean")
CKPT = os.path.join(BASE, "Coat", "new_coat_binary", "results", "graphics")
CV_DIR = os.path.join(BASE, "Coat", "models_cross_validation_swim_base")
OUT = os.path.join(BASE, "revision_peerj", "bloque_a")
SEED = 42
N_FOLDS = 5
NUM_CLASSES = 3

# Mismo preprocesamiento que el entrenamiento de la validacion cruzada.
sys.path.insert(0, CV_DIR)
from data.preprocessing import apply_clahe, resize_images  # noqa: E402
from config import settings  # noqa: E402

TARGET_SIZE = settings.target_size
LADO = tuple(TARGET_SIZE)[:2] if isinstance(TARGET_SIZE, (tuple, list)) else (TARGET_SIZE, TARGET_SIZE)

# Arquitectura -> (carpeta de pesos, nombre en timm, parametros esperados del state_dict en millones)
ARCHS = {
    "ConvNeXt-Large": ("convnext_large", "convnext_large", 196.23),
    "Swin-Base": ("convnext_large", "swin_base_patch4_window7_224", 86.75),
    "VGG16": ("convnext_large", "vgg16", 134.27),
    "VGG19": ("convnext_large", "vgg19", 139.58),
    "DenseNet201": ("convnext_large", "densenet201", 18.33),
    "ResNet50": ("resnet50", "resnet50", 23.57),
}
# DenseNet201 se corrio dos veces; la que corresponde a la Tabla 7 (log de las 14:31) es la segunda.
DENSENET_DESDE = "20260201_1431"


def cargar_state_dict(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for k in ("model_state_dict", "state_dict", "model"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    if not isinstance(sd, dict):
        sd = sd.state_dict()
    return {k.replace("module.", "", 1): v for k, v in sd.items()}


def millones(sd):
    return sum(v.numel() for v in sd.values() if hasattr(v, "numel")) / 1e6


def pesos_por_arquitectura():
    """Asigna cada .pth a su arquitectura por el numero de parametros y lo ordena por hora (= fold)."""
    asignados = {a: [] for a in ARCHS}
    for carpeta in {c for c, _, _ in ARCHS.values()}:
        for path in sorted(glob.glob(os.path.join(CKPT, carpeta, "*.pth"))):
            n = millones(cargar_state_dict(path))
            for arch, (c, _, esperado) in ARCHS.items():
                if c == carpeta and abs(n - esperado) < 0.05:
                    if arch == "DenseNet201" and os.path.basename(path).split("_", 2)[-1][:13] < DENSENET_DESDE:
                        continue
                    asignados[arch].append(path)
    for arch, paths in asignados.items():
        if len(paths) != N_FOLDS:
            raise SystemExit(f"{arch}: se esperaban {N_FOLDS} pesos y hay {len(paths)}: {paths}")
    return asignados


def preparar(X, mean, std):
    Xr = resize_images(X, TARGET_SIZE)
    Xc = np.array([apply_clahe(img) for img in Xr])
    Xn = (Xc - mean) / (std + 1e-8)
    return torch.from_numpy(Xn.transpose(0, 3, 1, 2).astype(np.float32))


@torch.no_grad()
def predecir(model, X, device, bs=32):
    probs = []
    for i in range(0, len(X), bs):
        probs.append(torch.softmax(model(X[i:i + bs].to(device)), dim=1).cpu())
    return torch.cat(probs).numpy()


def metricas(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return accuracy_score(y, pred), p, r, f


@torch.no_grad()
def eficiencia(model, device, reps=100, warmup=20):
    x = torch.randn(1, 3, *LADO, device=device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    with FlopCounterMode(display=False) as fc:
        model(x)
    gflops = fc.get_total_flops() / 1e9
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    tiempos = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model(x)
        torch.cuda.synchronize()
        tiempos.append((time.perf_counter() - t0) * 1000)
    vram = torch.cuda.max_memory_allocated() / 2**20
    return params, gflops, gflops / 2, float(np.mean(tiempos)), float(np.std(tiempos)), vram


def main():
    device = torch.device("cuda")
    os.makedirs(os.path.join(OUT, "predicciones"), exist_ok=True)
    print("torch", torch.__version__, "| timm", timm.__version__, "| GPU", torch.cuda.get_device_name(0))

    X_train = np.load(os.path.join(DATA, "X_train.npy"))
    y_train = np.load(os.path.join(DATA, "y_train.npy"))
    X_test = np.load(os.path.join(DATA, "X_test.npy"))
    y_test = np.load(os.path.join(DATA, "y_test.npy"))
    y_train = y_train.argmax(1) if y_train.ndim > 1 else y_train.astype(int)
    y_test = y_test.argmax(1) if y_test.ndim > 1 else y_test.astype(int)
    print("train", X_train.shape, np.bincount(y_train), "| test", X_test.shape, np.bincount(y_test))

    # Folds iguales a los del entrenamiento (StratifiedKFold, shuffle, semilla 42, sobre las 541 de train).
    folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X_train, y_train))
    stats = []
    for tr, _ in folds:
        Xr = resize_images(X_train[tr], TARGET_SIZE)
        stats.append((Xr.mean(axis=(0, 1, 2)), Xr.std(axis=(0, 1, 2))))

    pesos = pesos_por_arquitectura()
    filas_fold, filas_resumen, filas_ef = [], [], []
    for arch, (_, timm_name, _) in ARCHS.items():
        res = []
        for k, path in enumerate(pesos[arch]):
            model = timm.create_model(timm_name, pretrained=False, num_classes=NUM_CLASSES)
            model.load_state_dict(cargar_state_dict(path), strict=True)
            model.to(device).eval()
            mean, std = stats[k]
            _, va = folds[k]
            val_acc = accuracy_score(y_train[va], predecir(model, preparar(X_train[va], mean, std), device).argmax(1))
            probs = predecir(model, preparar(X_test, mean, std), device)
            pred = probs.argmax(1)
            acc, p, r, f = metricas(y_test, pred)
            res.append((acc, p, r, f))
            filas_fold.append([arch, k + 1, os.path.basename(path), f"{val_acc:.4f}", f"{acc:.4f}", f"{p:.4f}", f"{r:.4f}", f"{f:.4f}"])
            with open(os.path.join(OUT, "predicciones", f"{arch}_fold{k + 1}.csv"), "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["idx", "y_true", "y_pred", "p_type1", "p_type2", "p_type3"])
                for i in range(len(y_test)):
                    w.writerow([i, y_test[i], pred[i]] + [f"{x:.6f}" for x in probs[i]])
            print(f"{arch} fold {k + 1}: val_acc {val_acc:.4f} | test acc {acc:.4f} P {p:.4f} R {r:.4f} F1 {f:.4f}")
            if k == 0:
                filas_ef.append([arch] + [f"{v:.2f}" for v in eficiencia(model, device)])
            del model
            torch.cuda.empty_cache()
        m, s = np.mean(res, 0), np.std(res, 0)
        filas_resumen.append([arch] + [f"{m[i]:.4f} +- {s[i]:.4f}" for i in range(4)])

    # Eficiencia de arquitecturas sin pesos de esta corrida (solo arquitectura).
    for nombre, timm_name in [("ConvNeXt-Base (lo que mide la Tabla 8)", "convnext_base"), ("EfficientNet-B0", "efficientnet_b0")]:
        model = timm.create_model(timm_name, pretrained=False, num_classes=NUM_CLASSES).to(device).eval()
        filas_ef.append([nombre] + [f"{v:.2f}" for v in eficiencia(model, device)])
        del model
    try:
        sys.path.insert(0, os.path.join(BASE, "Coat", "models_cross_validation_swim_base", "models"))
        from custom_coat import CustomCoaT
        model = CustomCoaT(pretrained=False).to(device).eval()
        filas_ef.append(["CoatCervix"] + [f"{v:.2f}" for v in eficiencia(model, device)])
    except Exception as e:  # el nombre de la clase puede variar entre copias del repositorio
        print("CoatCervix no se pudo medir:", e)

    def guardar(nombre, cabecera, filas):
        with open(os.path.join(OUT, nombre), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cabecera)
            w.writerows(filas)

    guardar("metricas_por_fold.csv", ["modelo", "fold", "peso", "val_acc", "test_acc", "test_precision", "test_recall", "test_f1"], filas_fold)
    guardar("resumen_test.csv", ["modelo", "accuracy", "precision", "recall", "f1"], filas_resumen)
    guardar("eficiencia.csv", ["modelo", "params_M", "GFLOPs", "GMACs", "latencia_ms_media", "latencia_ms_sd", "vram_MB"], filas_ef)

    print("\n===== RESUMEN EN EL TEST (media +- desviacion de 5 folds)")
    for fila in filas_resumen:
        print(" | ".join(fila))
    print("\n===== EFICIENCIA (batch 1)")
    print("modelo | params_M | GFLOPs | GMACs | latencia_ms | sd | vram_MB")
    for fila in filas_ef:
        print(" | ".join(fila))
    print("\nResultados en", OUT)


if __name__ == "__main__":
    main()
