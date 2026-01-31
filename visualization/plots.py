# plots.py
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

def save_roc_curve(y_true, y_pred_probs, class_names, save_path, epoch=None):
    """
    Guarda la curva ROC para un conjunto de predicciones.
    
    Parámetros:
    - y_true: etiquetas verdaderas (n_samples,)
    - y_pred_probs: probabilidades predichas (n_samples, n_classes)
    - class_names: lista de nombres de clases
    - save_path: carpeta donde guardar la imagen
    - epoch: opcional, para diferenciar curvas por epoch
    """

    # --- Normalización de datos (evita el error) ---
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)

    # --- Validaciones para detectar errores temprano ---
    assert y_true.ndim == 1, f"y_true debe ser 1D, forma recibida: {y_true.shape}"
    assert y_pred_probs.ndim == 2, f"y_pred_probs debe ser 2D, forma: {y_pred_probs.shape}"
    assert y_pred_probs.shape[0] == y_true.shape[0], \
        f"Las filas no coinciden entre y_true ({y_true.shape[0]}) y y_pred_probs ({y_pred_probs.shape[0]})"

    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    plt.figure(figsize=(8,6))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:,i], y_pred_probs[:,i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0,1], [0,1], "k--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")

    # Nombre del archivo
    file_name = "roc_curve.png" if epoch is None else f"roc_curve_epoch_{epoch}.png"
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
