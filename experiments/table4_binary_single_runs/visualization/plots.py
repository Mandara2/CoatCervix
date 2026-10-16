# plots.py
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
def save_roc_curve(y_true, y_pred_probs, class_names, save_path, epoch=None):
    """
    Guarda curva ROC para clasificación binaria (2 clases).
    """

    # Convertir a numpy
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)

    # --- Validaciones ---
    assert y_pred_probs.shape[1] == 2, \
        f"Para ROC binaria deben existir 2 columnas (prob clase 0 y clase 1). Recibido: {y_pred_probs.shape[1]}"

    # --- Usar solo la probabilidad de la clase positiva ---
    y_scores = y_pred_probs[:, 1]  # prob. clase 1

    # --- Calcular FPR, TPR ---
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # --- Plot ---
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Referencia")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC – Clasificación Binaria")
    plt.legend(loc="lower right")

    # Nombre del archivo
    file_name = "roc_curve.png" if epoch is None else f"roc_curve_{epoch}.png"
    file_path = os.path.join(save_path, file_name)

    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Curva ROC guardada en: {file_path}")

def save_loss_curve(train_losses, val_losses, save_path, epochs=None):
    """
    Guarda la curva de pérdida de entrenamiento y validación.
    
    Parámetros:
    - train_losses: lista de pérdidas de entrenamiento por epoch
    - val_losses: lista de pérdidas de validación por epoch
    - save_path: carpeta donde guardar la imagen
    - epochs: opcional, número total de epochs entrenados
    """
    plt.figure(figsize=(8,6))
    
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve" + (f" - Until Epoch {epochs}" if epochs is not None else ""))
    plt.legend()
    plt.grid(True)

    file_name = "loss_curve.png" if epochs is None else f"loss_curve_epoch_{epochs}.png"
    file_path = os.path.join(save_path, file_name)

    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Curva de pérdida guardada en: {file_path}")

