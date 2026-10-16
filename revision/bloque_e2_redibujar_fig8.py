"""
Revision PeerJ: vuelve a dibujar la Figura 8 a partir de los mapas ya calculados por bloque_e_figuras_xai.py.

No recalcula nada ni usa el modelo: lee los .npy y figuras_xai.json de ~/Cervical/revision_peerj/figuras_xai/.
Solo cambia la visualizacion de SHAP, que en la primera version quedaba casi invisible porque la escala
la fijaban unos pocos pixeles extremos:
  - suavizado gaussiano (sigma 4 px) solo para mostrar;
  - escala simetrica al percentil 99 del valor absoluto (no al maximo);
  - solo se pintan los pixeles con |valor| > 20 % de esa escala, sobre la imagen en gris.
LIME se dibuja igual que antes (contornos de los superpixeles positivos).

  cd ~/Cervical/revision_peerj
  /data/rtabares/alzheimer/cervical_clean/bin/python bloque_e2_redibujar_fig8.py
"""
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

CODIGO = os.path.expanduser("~/Cervical/Coat/new_coat")
CARPETA = os.path.expanduser("~/Cervical/revision_peerj/figuras_xai")
NOMBRES = ["Type 1", "Type 2", "Type 3"]
SIGMA = 4
PERCENTIL = 99
UMBRAL = 0.2

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, CODIGO)
from config import settings  # noqa: E402
from data.loaders import load_data  # noqa: E402
from data.preprocessing import preprocess_all_datasets  # noqa: E402
from skimage.segmentation import mark_boundaries  # noqa: E402

info = json.load(open(os.path.join(CARPETA, "figuras_xai.json")))
indices = [im["indice"] for im in info["imagenes"]]

X_train, X_val, X_test, _, _, _ = load_data()
_, _, X_test_n, mean, std = preprocess_all_datasets(X_train, X_val, X_test, target_size=settings.target_size)
mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
std = np.asarray(std, dtype=np.float32).reshape(1, 1, -1)

fig, ejes = plt.subplots(3, 3, figsize=(12, 12))
for c, i in enumerate(indices):
    visible = np.clip(X_test_n[i] * std + mean, 0, 1)
    mascara = np.load(os.path.join(CARPETA, f"lime_mascara_{c + 1}.npy"))
    sh = gaussian_filter(np.load(os.path.join(CARPETA, f"shap_{c + 1}.npy")), sigma=SIGMA)
    escala = np.percentile(np.abs(sh), PERCENTIL) or 1.0
    sh_visible = np.ma.masked_where(np.abs(sh) < UMBRAL * escala, sh)

    ejes[c, 0].imshow(visible)
    ejes[c, 0].set_ylabel(NOMBRES[c])
    ejes[c, 1].imshow(mark_boundaries(visible, mascara, color=(1, 1, 0)))
    ejes[c, 2].imshow(visible.mean(2), cmap="gray")
    im = ejes[c, 2].imshow(sh_visible, cmap="bwr", vmin=-escala, vmax=escala, alpha=0.75)
    barra = fig.colorbar(im, ax=ejes[c, 2], fraction=0.046, pad=0.04)
    barra.set_ticks([-escala, 0, escala])
    barra.set_ticklabels(["negative", "0", "positive"])
for e, titulo in zip(ejes[0], ("Input", "LIME", "SHAP")):
    e.set_title(titulo)
for e in ejes.ravel():
    e.set_xticks([])
    e.set_yticks([])

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(CARPETA, f"figura8_lime_shap_v2.{ext}"), dpi=300, bbox_inches="tight")
print("Listo:", os.path.join(CARPETA, "figura8_lime_shap_v2.png"))
