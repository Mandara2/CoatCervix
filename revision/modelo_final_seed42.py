"""
Revision PeerJ: reentrena la semilla 42 de CoatCervix (la "mejor corrida" del paper) y guarda sus pesos.

Los pesos del modelo final no se guardaron: el experimento de semillas sobrescribia el archivo en cada
semilla. Como el entrenamiento es determinista (cudnn.deterministic, semilla 42, mismo split), volver a
correr la semilla 42 con el MISMO codigo (Coat/new_coat_seed) y el mismo ambiente debe reproducir el
modelo final. El script lo comprueba de forma independiente.

No modifica nada existente: redirige todas las salidas a ~/Cervical/revision_peerj/modelo_final/ y
verifica al final que el peso original de new_coat_seed no cambio.

Verificacion (todas deben cumplirse para declarar que es el modelo final):
  1. Metricas en el test: accuracy 0.8133, precision 0.8396, recall 0.7560, F1 macro 0.7810 (+- 0.0001).
  2. Matriz de confusion del test identica a la de la Figura 5: [[35,19,4],[1,117,6],[0,12,31]].

  cd ~/Cervical/revision_peerj
  CUDA_VISIBLE_DEVICES=<gpu> nohup /data/rtabares/alzheimer/cervical_clean/bin/python -u modelo_final_seed42.py > modelo_final.log 2>&1 &
"""
import hashlib
import inspect
import json
import os
import shutil
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

CODIGO = os.path.expanduser("~/Cervical/Coat/new_coat_seed")
SALIDA = os.path.expanduser("~/Cervical/revision_peerj/modelo_final")
ESPERADO = {"accuracy": 0.8133, "precision": 0.8396, "recall": 0.7560, "f1": 0.7810}
MATRIZ_ESPERADA = np.array([[35, 19, 4], [1, 117, 6], [0, 12, 31]])


def huella(path):
    if not os.path.exists(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


sys.path.insert(0, CODIGO)
from config import settings  # noqa: E402

# Protege el peso existente del experimento de semillas.
PESO_ORIGINAL = os.path.join(settings.SAVE_GRAPHICS_PATH, f"{settings.MODEL_NAME}.pth")
HUELLA_ORIGINAL = huella(PESO_ORIGINAL)
os.makedirs(SALIDA, exist_ok=True)
RESPALDO = os.path.join(SALIDA, "respaldo_NewCoatseed.pth")
if HUELLA_ORIGINAL and not os.path.exists(RESPALDO):
    shutil.copy2(PESO_ORIGINAL, RESPALDO)  # por si el codigo original ignorara la redireccion

# Redirige todas las salidas a la carpeta nueva.
settings.MODEL_NAME = "CoatCervix_final_seed42"
settings.RESULTS_PATH = SALIDA
settings.GRAPHICS_PATH = os.path.join(SALIDA, "graphics")
settings.CSV_PATH = os.path.join(SALIDA, "results.csv")
settings.SAVE_GRAPHICS_PATH = os.path.join(settings.GRAPHICS_PATH, settings.MODEL_NAME)
settings.SEED = 42
os.makedirs(settings.SAVE_GRAPHICS_PATH, exist_ok=True)
print("Salidas en:", settings.SAVE_GRAPHICS_PATH, flush=True)

from pipeline.pipeline_runner import run_pipeline  # noqa: E402
from data.loaders import load_data  # noqa: E402
from data.preprocessing import preprocess_all_datasets  # noqa: E402
from models.custom_coat import CustomCoaT  # noqa: E402

# 1. Entrenamiento con el mismo pipeline del experimento de semillas.
if "seed" in inspect.signature(run_pipeline).parameters:
    run_pipeline(seed=42)
else:
    run_pipeline()

peso = os.path.join(settings.SAVE_GRAPHICS_PATH, f"{settings.MODEL_NAME}.pth")
if not os.path.exists(peso):
    # El codigo original no respeto la redireccion: el peso quedo en su ruta original.
    if HUELLA_ORIGINAL and huella(PESO_ORIGINAL) != HUELLA_ORIGINAL:
        shutil.copy2(PESO_ORIGINAL, peso)
        shutil.copy2(RESPALDO, PESO_ORIGINAL)
        print("Aviso: el peso se guardo en la ruta original; se movio a la carpeta nueva y se restauro el original.")
    else:
        raise SystemExit(f"No se encontro el peso guardado en {peso}")

# 2. Verificacion independiente: mismo split y preprocesamiento, prediccion en el test.
X_train, X_val, X_test, y_train, y_val, y_test = load_data()
_, _, X_test_n, _, _ = preprocess_all_datasets(X_train, X_val, X_test, target_size=settings.target_size)
y_test = y_test.argmax(1) if y_test.ndim > 1 else y_test.astype(int)

model = CustomCoaT(pretrained=False)
sd = torch.load(peso, map_location="cpu", weights_only=False)
if isinstance(sd, dict) and "model_state_dict" in sd:
    sd = sd["model_state_dict"]
if not isinstance(sd, dict):
    sd = sd.state_dict()
model.load_state_dict(sd, strict=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()
X = torch.from_numpy(X_test_n.transpose(0, 3, 1, 2).astype(np.float32))
with torch.no_grad():
    pred = torch.cat([model(X[i:i + 32].to(device)).argmax(1).cpu() for i in range(0, len(X), 32)]).numpy()

p, r, f, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
obtenido = {"accuracy": accuracy_score(y_test, pred), "precision": p, "recall": r, "f1": f}
matriz = confusion_matrix(y_test, pred, labels=[0, 1, 2])

print("\n===== VERIFICACION DEL MODELO FINAL", flush=True)
ok = True
for k, v in ESPERADO.items():
    coincide = abs(obtenido[k] - v) < 1e-4 + 5e-5
    ok &= coincide
    print(f"{k:9s}: obtenido {obtenido[k]:.4f} | esperado {v:.4f} | {'OK' if coincide else 'NO COINCIDE'}")
print("matriz de confusion obtenida:\n", matriz)
iguales = np.array_equal(matriz, MATRIZ_ESPERADA)
ok &= iguales
print("matriz igual a la de la Figura 5:", "OK" if iguales else "NO COINCIDE")

huella_peso = huella(peso)
final = os.path.join(SALIDA, "coatcervix_final_seed42.pth")
shutil.copy2(peso, final)
json.dump({"verificado": bool(ok), "metricas": {k: float(v) for k, v in obtenido.items()},
           "matriz": matriz.tolist(), "md5": huella_peso, "peso": final},
          open(os.path.join(SALIDA, "verificacion.json"), "w"), indent=2)

intacto = huella(PESO_ORIGINAL) == HUELLA_ORIGINAL
if not intacto and os.path.exists(RESPALDO):
    shutil.copy2(RESPALDO, PESO_ORIGINAL)
    intacto = huella(PESO_ORIGINAL) == HUELLA_ORIGINAL
print("\npeso original de new_coat_seed sin cambios:", "OK" if intacto else "CAMBIO (revisar)")
print("peso guardado en:", final, "| md5:", huella_peso)
print("\nRESULTADO:", "ES EL MODELO FINAL (semilla 42 reproducida)" if ok else
      "NO reproduce el modelo final: no usar estos pesos para las figuras")
