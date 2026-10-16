"""
Revision PeerJ: redraws Figures 7 and 8 with the maps that were evaluated quantitatively.

The Grad-CAM (stage 3) and SHAP maps are read from xai_quant/results/final/maps/<idx>.npz, which are the
maps behind the faithfulness table, so the figures show exactly what was evaluated. LIME keeps the mask
of bloque_e_figuras_xai.py (same configuration and seed as xai_lime.py).

Before drawing, it compares those maps with the ones of the first version of the figures
(figuras_xai/*.npy) and prints the Pearson correlation for each image.

Does not recompute anything or use the model. Writes figura7_gradcam_etapa3_v2.* and
figura8_lime_shap_v3.* to ~/Cervical/revision_peerj/figuras_xai/.

  cd ~/Cervical/revision_peerj
  /data/rtabares/alzheimer/cervical_clean/bin/python bloque_e3_figuras_mapas_evaluados.py
"""
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

CODIGO = os.path.expanduser("~/Cervical/Coat/new_coat")
CARPETA = os.path.expanduser("~/Cervical/revision_peerj/figuras_xai")
MAPAS = os.path.expanduser("~/Cervical/revision_peerj/xai_quant/results/final/maps")
NOMBRES = ["Type 1", "Type 2", "Type 3"]
SIGMA, PERCENTIL, UMBRAL = 4, 99, 0.2

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, CODIGO)
from config import settings  # noqa: E402
from data.loaders import load_data  # noqa: E402
from data.preprocessing import preprocess_all_datasets  # noqa: E402
from pytorch_grad_cam.utils.image import show_cam_on_image  # noqa: E402
from skimage.segmentation import mark_boundaries  # noqa: E402

info = json.load(open(os.path.join(CARPETA, "figuras_xai.json")))
indices = [im["indice"] for im in info["imagenes"]]

X_train, X_val, X_test, _, _, _ = load_data()
_, _, X_test_n, mean, std = preprocess_all_datasets(X_train, X_val, X_test, target_size=settings.target_size)
mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
std = np.asarray(std, dtype=np.float32).reshape(1, 1, -1)

datos = []
print("Correlation between the evaluated maps (results/final) and the first version of the figures:")
for c, i in enumerate(indices):
    evaluados = np.load(os.path.join(MAPAS, f"{i:04d}.npz"))
    cam = evaluados["gradcam"].astype(np.float32)
    sh = evaluados["shap"].astype(np.float32)
    cam_v1 = np.load(os.path.join(CARPETA, f"gradcam_etapa3_{c + 1}.npy"))
    sh_v1 = np.load(os.path.join(CARPETA, f"shap_{c + 1}.npy"))
    print(f"  {NOMBRES[c]} (index {i}): Grad-CAM r = {np.corrcoef(cam.ravel(), cam_v1.ravel())[0, 1]:.3f} | "
          f"SHAP r = {np.corrcoef(sh.ravel(), sh_v1.ravel())[0, 1]:.3f}")
    datos.append({"visible": np.clip(X_test_n[i] * std + mean, 0, 1).astype(np.float32),
                  "cam": np.clip(cam, 0, 1), "shap": sh,
                  "lime": np.load(os.path.join(CARPETA, f"lime_mascara_{c + 1}.npy"))})


def guardar(fig, nombre):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(CARPETA, f"{nombre}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


fig, ejes = plt.subplots(2, 3, figsize=(12, 8))
for c, d in enumerate(datos):
    ejes[0, c].imshow(d["visible"])
    ejes[0, c].set_title(NOMBRES[c])
    ejes[1, c].imshow(show_cam_on_image(d["visible"], d["cam"], use_rgb=True))
    ejes[1, c].set_title("Grad-CAM (stage 3)")
for e in ejes.ravel():
    e.axis("off")
guardar(fig, "figura7_gradcam_etapa3_v2")

fig, ejes = plt.subplots(3, 3, figsize=(12, 12))
for c, d in enumerate(datos):
    sh = gaussian_filter(d["shap"], sigma=SIGMA)
    escala = np.percentile(np.abs(sh), PERCENTIL) or 1.0
    ejes[c, 0].imshow(d["visible"])
    ejes[c, 0].set_ylabel(NOMBRES[c])
    ejes[c, 1].imshow(mark_boundaries(d["visible"], d["lime"], color=(1, 1, 0)))
    ejes[c, 2].imshow(d["visible"].mean(2), cmap="gray")
    im = ejes[c, 2].imshow(np.ma.masked_where(np.abs(sh) < UMBRAL * escala, sh), cmap="bwr",
                           vmin=-escala, vmax=escala, alpha=0.75)
    barra = fig.colorbar(im, ax=ejes[c, 2], fraction=0.046, pad=0.04)
    barra.set_ticks([-escala, 0, escala])
    barra.set_ticklabels(["negative", "0", "positive"])
for e, titulo in zip(ejes[0], ("Input", "LIME", "SHAP")):
    e.set_title(titulo)
for e in ejes.ravel():
    e.set_xticks([])
    e.set_yticks([])
guardar(fig, "figura8_lime_shap_v3")
print("Done:", CARPETA)
