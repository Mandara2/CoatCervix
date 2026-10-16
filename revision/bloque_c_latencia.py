"""
Bloque C de la revision PeerJ: latencia, VRAM, parametros y GMACs con un protocolo unico.

Solo arquitectura (pesos al azar): la latencia no depende de los valores de los pesos.
Protocolo: batch 1, 224 x 224, float32, GPU sin otros procesos, 50 inferencias de
calentamiento y 200 medidas por modelo; se reportan mediana, rango intercuartil, media y sd.

Correr con la GPU LIBRE (el script avisa si hay otros procesos en ella):
  CUDA_VISIBLE_DEVICES=<gpu libre> /data/rtabares/alzheimer/cervical_clean/bin/python -u bloque_c_latencia.py
"""
import csv
import os
import subprocess
import sys
import time

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

BASE = "/data/home/rtabares/Cervical"
OUT = os.path.join(BASE, "revision_peerj", "bloque_c")
WARMUP, REPS = 50, 200
NUM_CLASSES = 3


class HMCFormerLike(nn.Module):
    """Copia de Coat/Pappers/HMCFormer.py (sin los print)."""

    def __init__(self, transformer_name="swin_small_patch4_window7_224", cnn_name="efficientnet_b0",
                 feat_dim=512, num_classes=NUM_CLASSES):
        super().__init__()
        self.transformer = timm.create_model(transformer_name, pretrained=False, num_classes=0, global_pool="")
        emb_dim = getattr(self.transformer, "num_features", 768)
        self.pool_proj = nn.Linear(emb_dim, feat_dim)
        self.cnn = timm.create_model(cnn_name, pretrained=False, num_classes=0, global_pool="avg")
        self.cnn_proj = nn.Linear(self.cnn.num_features, feat_dim)
        fusion_dim = feat_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(fusion_dim // 2, fusion_dim // 4), nn.ReLU(inplace=True), nn.Dropout(0.3))
        self.classifier = nn.Linear(fusion_dim // 4, num_classes)

    def forward(self, tr_x, cnn_x):
        feat_tr = self.transformer.forward_features(tr_x)
        if feat_tr.ndim == 4:
            if feat_tr.shape[1] > feat_tr.shape[-1]:
                feat_tr = feat_tr.mean(dim=(2, 3))
            else:
                feat_tr = feat_tr.permute(0, 3, 1, 2).mean(dim=(2, 3))
        elif feat_tr.ndim == 3:
            feat_tr = feat_tr.mean(dim=1)
        feat_tr = self.pool_proj(feat_tr)
        feat_cnn = self.cnn_proj(self.cnn(cnn_x))
        return self.classifier(self.fusion(torch.cat([feat_tr, feat_cnn], dim=1)))


def coatcervix():
    sys.path.insert(0, os.path.join(BASE, "Coat", "models_cross_validation_swim_base"))
    from models.custom_coat import CustomCoaT
    return CustomCoaT(pretrained=False)


MODELOS = [
    ("CoatCervix", coatcervix, 1),
    ("ConvNeXt-Large", lambda: timm.create_model("convnext_large", pretrained=False, num_classes=NUM_CLASSES), 1),
    ("Swin-Base", lambda: timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=NUM_CLASSES), 1),
    ("VGG16", lambda: timm.create_model("vgg16", pretrained=False, num_classes=NUM_CLASSES), 1),
    ("VGG19", lambda: timm.create_model("vgg19", pretrained=False, num_classes=NUM_CLASSES), 1),
    ("DenseNet201", lambda: timm.create_model("densenet201", pretrained=False, num_classes=NUM_CLASSES), 1),
    ("ResNet50", lambda: timm.create_model("resnet50", pretrained=False, num_classes=NUM_CLASSES), 1),
    ("EfficientNet-B0", lambda: timm.create_model("efficientnet_b0", pretrained=False, num_classes=NUM_CLASSES), 1),
    ("HMCFormer (replica)", HMCFormerLike, 2),
]


def procesos_en_gpu():
    """PIDs de otros procesos en la GPU elegida (CUDA_VISIBLE_DEVICES), no en todo el servidor."""
    try:
        idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        uuid = subprocess.run(["nvidia-smi", "-i", idx, "--query-gpu=uuid", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip().splitlines()
        return [a.split(",")[1].strip() for a in apps
                if a.split(",")[0].strip() == uuid and a.split(",")[1].strip() != str(os.getpid())]
    except Exception:
        return []


@torch.no_grad()
def medir(nombre, fabrica, n_entradas, device):
    model = fabrica().to(device).eval()
    x = [torch.randn(1, 3, 224, 224, device=device) for _ in range(n_entradas)]
    params = sum(p.numel() for p in model.parameters()) / 1e6
    with FlopCounterMode(display=False) as fc:
        model(*x)
    gmacs = fc.get_total_flops() / 2e9
    for _ in range(WARMUP):
        model(*x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        model(*x)
        torch.cuda.synchronize()
        t.append((time.perf_counter() - t0) * 1000)
    t = np.array(t)
    q1, med, q3 = np.percentile(t, [25, 50, 75])
    vram = torch.cuda.max_memory_allocated() / 2**20
    del model
    torch.cuda.empty_cache()
    return [nombre, f"{params:.2f}", f"{gmacs:.2f}", f"{med:.2f}", f"{q1:.2f}", f"{q3:.2f}",
            f"{t.mean():.2f}", f"{t.std():.2f}", f"{vram:.1f}"]


def main():
    device = torch.device("cuda")
    os.makedirs(OUT, exist_ok=True)
    torch.backends.cudnn.benchmark = False
    print("torch", torch.__version__, "| timm", timm.__version__, "| GPU", torch.cuda.get_device_name(0))
    otros = procesos_en_gpu()
    if otros:
        print(f"AVISO: la GPU elegida tiene {len(otros)} procesos de otros usuarios (PIDs {otros}). La latencia no sera confiable; conviene elegir otra GPU.")
    else:
        print("GPU elegida sin otros procesos: la latencia es confiable.")
    filas = []
    for nombre, fabrica, n in MODELOS:
        fila = medir(nombre, fabrica, n, device)
        filas.append(fila)
        print(" | ".join(fila), flush=True)
    cab = ["modelo", "params_M", "GMACs", "latencia_mediana_ms", "q1_ms", "q3_ms", "media_ms", "sd_ms", "vram_MB"]
    with open(os.path.join(OUT, "latencia.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cab)
        w.writerows(filas)
    print("\n" + " | ".join(cab))
    for f in filas:
        print(" | ".join(f))
    print("\nResultados en", OUT)


if __name__ == "__main__":
    main()
