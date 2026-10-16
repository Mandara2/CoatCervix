"""
Revision PeerJ: redibuja las Figuras 4 (curvas ROC) y 5 (matriz de confusion) con las etiquetas
"Type 1/2/3" en ingles, en lugar de "Clase 0/1/2" y "Class 0/1/2" (hallazgo de la revision A, #35).
Autorizado por Rafa el 2026-10-15: inferencia con los pesos existentes solo para redibujar.

No cambia ninguna cifra. Antes de guardar comprueba que el modelo reproduce exactamente:
  - la matriz de la Figura 5: [[35, 19, 4], [1, 117, 6], [0, 12, 31]] (accuracy 0.8133);
  - los AUC de la Figura 4 redondeados a dos decimales: 0.96, 0.94 y 0.95.
Si algo no coincide, se detiene sin guardar figuras.

No modifica nada existente: solo importa el codigo de new_coat y lee los pesos finales.
Salidas en ~/Cervical/revision_peerj/figuras_roc_matriz/.

  cd ~/Cervical/revision_peerj
  CUDA_VISIBLE_DEVICES=<gpu libre> /data/rtabares/alzheimer/cervical_clean/bin/python -u \
      bloque_g_figuras_roc_matriz.py 2>&1 | tee figuras_roc_matriz.log
"""
import os
import random
import sys

import numpy as np
import torch

CODIGO = os.path.expanduser("~/Cervical/Coat/new_coat")
PESO = os.path.join(CODIGO, "results/graphics/CoatCervixInterpretability/CoatCervixInterpretability.pth")
SALIDA = os.path.expanduser("~/Cervical/revision_peerj/figuras_roc_matriz")
MATRIZ_ESPERADA = np.array([[35, 19, 4], [1, 117, 6], [0, 12, 31]])
AUC_ESPERADO = [0.96, 0.94, 0.95]
NOMBRES = ["Type 1", "Type 2", "Type 3"]
SEMILLA = 42

random.seed(SEMILLA)
np.random.seed(SEMILLA)
torch.manual_seed(SEMILLA)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, CODIGO)
from config import settings  # noqa: E402,F401
from data.loaders import load_data  # noqa: E402
from data.preprocessing import preprocess_all_datasets  # noqa: E402
from models.custom_coat import CustomCoaT  # noqa: E402
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Datos y modelo, con el mismo split y preprocesamiento del paper.
X_train, X_val, X_test, y_train, y_val, y_test = load_data()
_, _, X_test_n, _, _ = preprocess_all_datasets(X_train, X_val, X_test, target_size=settings.target_size)
y_test = y_test.argmax(1) if y_test.ndim > 1 else y_test.astype(int)

model = CustomCoaT(pretrained=False)
sd = torch.load(PESO, map_location="cpu")
if isinstance(sd, dict) and "model_state_dict" in sd:
    sd = sd["model_state_dict"]
if not isinstance(sd, dict):
    sd = sd.state_dict()
model.load_state_dict(sd, strict=True)
model.to(device).eval()

Xt = torch.from_numpy(np.ascontiguousarray(X_test_n.transpose(0, 3, 1, 2)).astype(np.float32))
with torch.no_grad():
    logits = torch.cat([model(Xt[i:i + 32].to(device)).cpu() for i in range(0, len(Xt), 32)])
prob = torch.softmax(logits, 1).numpy()
pred = prob.argmax(1)

# 2. Comprobaciones: deben salir exactamente las cifras de las figuras enviadas.
acc = accuracy_score(y_test, pred)
matriz = confusion_matrix(y_test, pred, labels=[0, 1, 2])
curvas, aucs = [], []
for c in range(3):
    fpr, tpr, _ = roc_curve((y_test == c).astype(int), prob[:, c])
    curvas.append((fpr, tpr))
    aucs.append(auc(fpr, tpr))
print(f"accuracy en el test: {acc:.4f} (esperado 0.8133)")
print("matriz de confusion:\n", matriz)
print("AUC por clase:", [round(a, 4) for a in aucs], "(esperado", AUC_ESPERADO, "al redondear)")
ok = (abs(acc - 0.8133) < 1e-4 and np.array_equal(matriz, MATRIZ_ESPERADA)
      and [round(a, 2) for a in aucs] == AUC_ESPERADO)
if not ok:
    raise SystemExit("Las cifras no coinciden con las figuras del paper: no se generan figuras.")
os.makedirs(SALIDA, exist_ok=True)

# 3. Figura 4: curvas ROC (una contra el resto).
colores = ["#1f77b4", "#ff7f0e", "#2ca02c"]
fig, ax = plt.subplots(figsize=(6, 5))
for c, (fpr, tpr) in enumerate(curvas):
    ax.plot(fpr, tpr, color=colores[c], lw=1.6, label=f"{NOMBRES[c]} (AUC = {aucs[c]:.2f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.legend(loc="lower right", frameon=True)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(SALIDA, f"figura4_roc_v2.{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

# 4. Figura 5: matriz de confusion en numeros absolutos.
fig, ax = plt.subplots(figsize=(5.6, 4.8))
im = ax.imshow(matriz, cmap="Blues")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(NOMBRES)
ax.set_yticklabels(NOMBRES)
ax.set_xlabel("Predicted type")
ax.set_ylabel("True type")
umbral = matriz.max() / 2
for i in range(3):
    for j in range(3):
        ax.text(j, i, str(matriz[i, j]), ha="center", va="center", fontsize=12,
                color="white" if matriz[i, j] > umbral else "black")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(SALIDA, f"figura5_matriz_v2.{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

print("Figuras guardadas en", SALIDA)
