# plots.py
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

def save_roc_curve(y_true, y_pred_probs, class_names, save_path, epoch=None):
    """
    save roc curve for multi-class classification.
    
    Parameters:
    - y_true: true labels (n_samples,)
    - y_pred_probs: predicted probabilities (n_samples, n_classes)
    - class_names: list of class names
    - save_path: folder where to save the image
    - epoch: optional, epoch number to include in the file name
    """

    # --- data normalization ---
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)

    # --- early errors detection ---
    assert y_true.ndim == 1, f"y_true must be 1D, shape recibed: {y_true.shape}"
    assert y_pred_probs.ndim == 2, f"y_pred_probs must be 2D, shape: {y_pred_probs.shape}"
    assert y_pred_probs.shape[0] == y_true.shape[0], \
        f"the row count of y_true ({y_true.shape[0]}) and y_pred_probs ({y_pred_probs.shape[0]}) do not match"

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
    print(f"✅ Curva ROC saved in : {file_path}")


def save_loss_curve(train_losses, val_losses, save_path, epochs=None):
    """
    saved loss curve for training and validation.
    
    Parameters:
    - train_losses: list of training losses per epoch
    - val_losses: list of validation losses per epoch
    - save_path: folder where to save the image
    - epochs: optional, total number of epochs trained
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
    print(f"✅ Loss curve saved in: {file_path}")
