"""
Revision PeerJ: regenera las Figuras 7 y 8 con el modelo final (decision de Rafa, 2026-10-12).

  Figura 7: Grad-CAM en la salida de la etapa 3 (backbone.serial_blocks3.9), sin eigen_smooth.
            Reemplaza LayerCAM en backbone.crpe4.conv_list.2, que no paso la validacion cuantitativa.
  Figura 8: LIME (SLIC 100/10/1, 1000 muestras, como sharp_lime.py) y SHAP con GradientExplainer
            (expected gradients, 50 imagenes de fondo del train, 200 muestras), como en xai_quant.py.

No modifica nada existente: solo importa el codigo de new_coat y lee los pesos finales.
Todas las salidas van a ~/Cervical/revision_peerj/figuras_xai/.

Antes de generar las figuras comprueba que el modelo reproduce el test del paper
(accuracy 0.8133, matriz de la Figura 5). Si no coincide, se detiene.

Imagenes: por defecto, la primera imagen del test bien clasificada de cada tipo. Si se identifican
las imagenes de la Figura 7 enviada, conviene usarlas con --indices (una por tipo, en orden 1, 2, 3),
para que el revisor vea que cambio el metodo y no la seleccion.

  cd ~/Cervical/revision_peerj
  CUDA_VISIBLE_DEVICES=<gpu libre> nohup /data/rtabares/alzheimer/cervical_clean/bin/python -u \
      bloque_e_figuras_xai.py > figuras_xai.log 2>&1 &
  # con imagenes elegidas:  ... bloque_e_figuras_xai.py --indices 12 40 200 ...
"""
import argparse
import hashlib
import json
import os
import random
import sys

import numpy as np
import torch

CODIGO = os.path.expanduser("~/Cervical/Coat/new_coat")
PESO = os.path.join(CODIGO, "results/graphics/CoatCervixInterpretability/CoatCervixInterpretability.pth")
SALIDA = os.path.expanduser("~/Cervical/revision_peerj/figuras_xai")
CAPA_GRADCAM = "backbone.serial_blocks3.9"
MATRIZ_ESPERADA = np.array([[35, 19, 4], [1, 117, 6], [0, 12, 31]])
NOMBRES = ["Type 1", "Type 2", "Type 3"]
SEMILLA = 42

parser = argparse.ArgumentParser()
parser.add_argument("--indices", type=int, nargs=3, default=None,
                    help="indices del test (Type 1, Type 2, Type 3)")
parser.add_argument("--shap-fondo", type=int, default=50)
parser.add_argument("--shap-muestras", type=int, default=200)
parser.add_argument("--lime-muestras", type=int, default=1000)
parser.add_argument("--lime-superpixeles", type=int, default=5,
                    help="superpixeles positivos que se resaltan en la figura de LIME")
args = parser.parse_args()

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

import shap  # noqa: E402
from lime import lime_image  # noqa: E402
from lime.wrappers.scikit_image import SegmentationAlgorithm  # noqa: E402
from pytorch_grad_cam import GradCAM  # noqa: E402
from pytorch_grad_cam.utils.image import show_cam_on_image  # noqa: E402
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  # noqa: E402
from skimage.segmentation import mark_boundaries  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix  # noqa: E402

os.makedirs(SALIDA, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


# 1. Datos y modelo, con el mismo split y preprocesamiento del paper.
X_train, X_val, X_test, y_train, y_val, y_test = load_data()
X_train_n, _, X_test_n, mean, std = preprocess_all_datasets(
    X_train, X_val, X_test, target_size=settings.target_size)
y_test = y_test.argmax(1) if y_test.ndim > 1 else y_test.astype(int)
mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
std = np.asarray(std, dtype=np.float32).reshape(1, 1, -1)

model = CustomCoaT(pretrained=False)
sd = torch.load(PESO, map_location="cpu")
if isinstance(sd, dict) and "model_state_dict" in sd:
    sd = sd["model_state_dict"]
if not isinstance(sd, dict):
    sd = sd.state_dict()
model.load_state_dict(sd, strict=True)
model.to(device).eval()


def a_tensor(x_nhwc):
    return torch.from_numpy(np.ascontiguousarray(x_nhwc.transpose(0, 3, 1, 2)).astype(np.float32))


Xt = a_tensor(X_test_n)
with torch.no_grad():
    logits = torch.cat([model(Xt[i:i + 32].to(device)).cpu() for i in range(0, len(Xt), 32)])
prob = torch.softmax(logits, 1).numpy()
pred = prob.argmax(1)

# 2. Comprobacion: debe ser el modelo del paper.
acc = accuracy_score(y_test, pred)
matriz = confusion_matrix(y_test, pred, labels=[0, 1, 2])
print(f"accuracy en el test: {acc:.4f} (esperado 0.8133)")
print("matriz de confusion:\n", matriz)
if abs(acc - 0.8133) > 1e-4 or not np.array_equal(matriz, MATRIZ_ESPERADA):
    raise SystemExit("El modelo no reproduce el test del paper: no se generan figuras.")

# 3. Imagenes: una por tipo.
if args.indices is None:
    indices = [int(np.where((y_test == c) & (pred == c))[0][0]) for c in range(3)]
else:
    indices = list(args.indices)
for c, i in enumerate(indices):
    print(f"{NOMBRES[c]}: indice {i}, clase real {NOMBRES[y_test[i]]}, prediccion {NOMBRES[pred[i]]}, "
          f"probabilidades {np.round(prob[i], 3).tolist()}")
    if y_test[i] != c:
        print(f"  Aviso: la imagen {i} no es {NOMBRES[c]}.")


def a_visible(x_n):
    """Imagen normalizada (HWC) -> imagen CLAHE en [0, 1], la que ve el modelo antes de normalizar."""
    return np.clip(x_n * std + mean, 0, 1).astype(np.float32)


# 4. Grad-CAM en la salida de la etapa 3.
capa = model.get_submodule(CAPA_GRADCAM)


def tokens_a_mapa(t):
    # Salida de un bloque serial de CoaT: (B, 1 + H*W, C), con el token de clase primero.
    t = t[:, 1:, :]
    lado = int(round(t.shape[1] ** 0.5))
    return t.reshape(t.shape[0], lado, lado, t.shape[2]).permute(0, 3, 1, 2)


gradcam = GradCAM(model=model, target_layers=[capa], reshape_transform=tokens_a_mapa)

# 5. SHAP (GradientExplainer) con fondo fijo del train, explicando la clase predicha.
rng = np.random.RandomState(SEMILLA)
fondo = a_tensor(X_train_n[rng.choice(len(X_train_n), args.shap_fondo, replace=False)]).to(device)


class SalidaDeClase(torch.nn.Module):
    def __init__(self, red, clase):
        super().__init__()
        self.red, self.clase = red, clase

    def forward(self, x):
        return self.red(x)[:, self.clase:self.clase + 1]


def shap_mapa(i, clase):
    torch.manual_seed(SEMILLA)
    np.random.seed(SEMILLA)
    explicador = shap.GradientExplainer(SalidaDeClase(model, clase), fondo)
    valores = explicador.shap_values(Xt[i:i + 1].to(device), nsamples=args.shap_muestras)
    valores = np.asarray(valores[0] if isinstance(valores, list) else valores)
    valores = valores.reshape(3, Xt.shape[2], Xt.shape[3])  # (canales, H, W)
    return valores.sum(0)


# 6. LIME con los parametros de sharp_lime.py.
def clasificador(imagenes_hwc):
    x = (np.asarray(imagenes_hwc, dtype=np.float32) - mean) / std
    salida = []
    with torch.no_grad():
        for k in range(0, len(x), 32):
            salida.append(torch.softmax(model(a_tensor(x[k:k + 32]).to(device)), 1).cpu().numpy())
    return np.concatenate(salida)


segmentador = SegmentationAlgorithm("slic", n_segments=100, compactness=10, sigma=1,
                                    random_seed=SEMILLA)
explicador_lime = lime_image.LimeImageExplainer(random_state=SEMILLA)

resultados = []
for c, i in enumerate(indices):
    clase = int(pred[i])
    visible = a_visible(X_test_n[i])

    cam = gradcam(input_tensor=Xt[i:i + 1].to(device), targets=[ClassifierOutputTarget(clase)],
                  eigen_smooth=False)[0]
    sh = shap_mapa(i, clase)
    exp = explicador_lime.explain_instance(visible.astype(np.float64), clasificador, labels=(clase,),
                                           top_labels=None, hide_color=0,
                                           num_samples=args.lime_muestras,
                                           segmentation_fn=segmentador, random_seed=SEMILLA)
    img_lime, mascara = exp.get_image_and_mask(clase, positive_only=True,
                                               num_features=args.lime_superpixeles, hide_rest=False)

    np.save(os.path.join(SALIDA, f"gradcam_etapa3_{c + 1}.npy"), cam)
    np.save(os.path.join(SALIDA, f"shap_{c + 1}.npy"), sh)
    np.save(os.path.join(SALIDA, f"lime_mascara_{c + 1}.npy"), mascara)
    resultados.append({"visible": visible, "cam": cam, "shap": sh,
                       "lime": mark_boundaries(img_lime, mascara, color=(1, 1, 0)),
                       "indice": i, "real": int(y_test[i]), "pred": clase, "prob": prob[i].tolist()})
    print(f"{NOMBRES[c]} listo")


# 7. Figuras.
def guardar(fig, nombre):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(SALIDA, f"{nombre}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


fig, ejes = plt.subplots(2, 3, figsize=(12, 8))
for c, r in enumerate(resultados):
    ejes[0, c].imshow(r["visible"])
    ejes[0, c].set_title(NOMBRES[c])
    ejes[1, c].imshow(show_cam_on_image(r["visible"], r["cam"], use_rgb=True))
    ejes[1, c].set_title("Grad-CAM (stage 3)")
for e in ejes.ravel():
    e.axis("off")
guardar(fig, "figura7_gradcam_etapa3")

fig, ejes = plt.subplots(3, 3, figsize=(12, 12))
for c, r in enumerate(resultados):
    limite = np.abs(r["shap"]).max() or 1.0
    ejes[c, 0].imshow(r["visible"])
    ejes[c, 0].set_ylabel(NOMBRES[c])
    ejes[c, 1].imshow(r["lime"])
    ejes[c, 2].imshow(r["visible"].mean(2), cmap="gray", alpha=0.4)
    im = ejes[c, 2].imshow(r["shap"], cmap="bwr", vmin=-limite, vmax=limite, alpha=0.7)
    fig.colorbar(im, ax=ejes[c, 2], fraction=0.046, pad=0.04)
for e, titulo in zip(ejes[0], ("Input", "LIME", "SHAP")):
    e.set_title(titulo)
for e in ejes.ravel():
    e.set_xticks([])
    e.set_yticks([])
guardar(fig, "figura8_lime_shap")

json.dump({
    "peso": PESO, "md5_peso": md5(PESO), "accuracy_test": float(acc), "matriz": matriz.tolist(),
    "capa_gradcam": CAPA_GRADCAM, "eigen_smooth": False,
    "shap": {"metodo": "GradientExplainer", "fondo": args.shap_fondo, "muestras": args.shap_muestras,
             "salida": "logit de la clase predicha"},
    "lime": {"segmentacion": "slic", "n_segments": 100, "compactness": 10, "sigma": 1,
             "muestras": args.lime_muestras, "superpixeles_mostrados": args.lime_superpixeles,
             "hide_color": 0},
    "imagenes": [{k: r[k] for k in ("indice", "real", "pred", "prob")} for r in resultados],
    "versiones": {"torch": torch.__version__, "shap": shap.__version__},
}, open(os.path.join(SALIDA, "figuras_xai.json"), "w"), indent=2)
print("\nListo. Figuras en", SALIDA)
