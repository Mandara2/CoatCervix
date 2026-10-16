# ==============================================================
# EfficientNet + Grad-CAM + Sharpen Filter
# Entrenamiento, evaluación y visualización en imágenes cervicales
# ==============================================================

# === IMPORTACIONES GENERALES ===
import os
import csv
import math
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt
import cv2
from pytorch_grad_cam import LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ==============================================================
# CONFIGURACIÓN GENERAL
# ==============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4

# Fijar semilla para reproducibilidad
torch.manual_seed(42)
np.random.seed(42)

# === Rutas de datos y resultados ===
X_TRAIN_PATH = "/data/home/rtabares/Cervical/Cervical_clean/X_train.npy"
Y_TRAIN_PATH = "/data/home/rtabares/Cervical/Cervical_clean/y_train.npy"
X_TEST_PATH  = "/data/home/rtabares/Cervical/Cervical_clean/X_test.npy"
Y_TEST_PATH  = "/data/home/rtabares/Cervical/Cervical_clean/y_test.npy"

SAVE_GRAPHICS_PATH = "/data/home/rtabares/Cervical/Efficient/GraficasEff"
CSV_PATH = "/data/home/rtabares/Cervical/Efficient/resultados.csv"
MODEL_NAME = "EfficientNet_Clahe"

# === Valores de normalización (ImageNet) ===
imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
imagenet_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ==============================================================
# FUNCIONES AUXILIARES
# ==============================================================

def save_confusion_and_csv(all_labels, all_preds, class_names, model_name, csv_path, save_image_dir=None):
    """Guarda métricas y matriz de confusión en CSV y PNG."""
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"Prueba: {model_name}, Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
        writer.writerow([])
        writer.writerow(["Matriz de confusión"])
        writer.writerow(["Clase"] + class_names)
        for i, row in enumerate(cm):
            writer.writerow([class_names[i]] + row.tolist())
        writer.writerow([])
        writer.writerow(["Métricas globales"])
        writer.writerow(["Accuracy", acc])
        writer.writerow(["Precision", prec])
        writer.writerow(["Recall", rec])
        writer.writerow(["F1-score", f1])
        writer.writerow([]); writer.writerow([])
    print(f"Resultados guardados en {csv_path}")

    # === Guardar imagen de la matriz de confusión ===
    if save_image_dir is None:
        save_image_dir = os.path.dirname(csv_path) or "."

    os.makedirs(save_image_dir, exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", values_format='d')
    ax.set_title(f"Matriz de Confusión - {model_name}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    img_name = f"matriz_confusion_{model_name}_{timestamp}.png"
    img_path = os.path.join(save_image_dir, img_name)
    plt.tight_layout()
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Copiar imagen a carpeta de gráficos global
    if save_image_dir != SAVE_GRAPHICS_PATH:
        os.makedirs(SAVE_GRAPHICS_PATH, exist_ok=True)
        img = cv2.imread(img_path)
        if img is not None:
            cv2.imwrite(os.path.join(SAVE_GRAPHICS_PATH, img_name), img)

    print(f"Matriz de confusión guardada en: {img_path}")
    return img_path


def _find_target_layer_and_transform(model):
    """Detecta la capa objetivo más adecuada para Grad-CAM."""
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d):
            return module, None, name
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            def reshape_transform(tensor):
                if tensor.ndim == 4:
                    return tensor
                if tensor.ndim == 3:
                    B, N, C = tensor.shape
                    h = int(math.sqrt(N))
                    result = tensor[:, 1:, :].reshape(B, h, h, C).permute(0, 3, 1, 2)
                    return result
                raise ValueError("Dimensión inesperada en tensor")
            return module, reshape_transform, name
    return list(model.named_modules())[-1][1], None, name


def generate_gradcam_auto(model, image_tensor, target_class, save_path,
                          class_names=None, model_name=None,
                          mean=None, std=None, device=None,
                          use_eigen_smooth=True, verbose=True):
    """Genera el mapa Grad-CAM automáticamente."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    image_tensor = image_tensor.to(device).float()

    target_layer, reshape_transform, layer_name = _find_target_layer_and_transform(model)
    if verbose:
        print(f"[GradCAM] Capa objetivo: {layer_name}")

    cam = LayerCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform)
    if mean is None or std is None:
        raise ValueError("Faltan mean y std para revertir la normalización.")

    img_np = image_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
    img_vis = np.clip((img_np * std) + mean, 0, 1)
    targets = [ClassifierOutputTarget(int(target_class))]

    grayscale_cam = cam(input_tensor=image_tensor, targets=targets, eigen_smooth=use_eigen_smooth)
    cam_map = grayscale_cam[0]
    H, W, _ = img_vis.shape
    cam_resized = cv2.resize(cam_map, (W, H))
    visualization = show_cam_on_image(img_vis, cam_resized, use_rgb=True)

    os.makedirs(save_path, exist_ok=True)
    fname = f"gradcam_{model_name or 'model'}_{class_names[int(target_class)] if class_names else target_class}.png"
    cv2.imwrite(os.path.join(save_path, fname), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"[GradCAM] Guardado en: {os.path.join(save_path, fname)}")


# ==============================================================
# PIPELINE PRINCIPAL
# ==============================================================
def run_images_pipeline_efficientnet():
    # === 1. CARGAR DATOS ===
    X_train = np.load(X_TRAIN_PATH)
    y_train = np.load(Y_TRAIN_PATH)
    X_test  = np.load(X_TEST_PATH)
    y_test  = np.load(Y_TEST_PATH)

    CLASS_NAMES = [f"Clase_{i}" for i in np.unique(y_train)]
    num_classes = len(CLASS_NAMES)

    # === 2. NORMALIZAR A [0,1] ===
    X_train = X_train.astype(np.float32)
    X_test  = X_test.astype(np.float32)

    def apply_clahe(img_array):

          img_255 = np.clip(img_array * 255, 0, 255).astype(np.uint8)
          lab = cv2.cvtColor(img_255, cv2.COLOR_RGB2LAB)
          l, a, b = cv2.split(lab)

          # Crear el objeto CLAHE (clipLimit controla cuánto se amplía el contraste)
          clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
          l_clahe = clahe.apply(l)

          lab_clahe = cv2.merge((l_clahe, a, b))
          img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

          return img_clahe.astype(np.float32) / 255.0

    # Aplicar CLAHE a todo el conjunto
    X_train = np.array([apply_clahe(im) for im in X_train])
    X_test  = np.array([apply_clahe(im) for im in X_test])

    # === 4. NORMALIZAR CON ESTADÍSTICAS IMAGENET ===
    def norm_img(arr):
        return (arr - imagenet_mean) / imagenet_std

    X_train = norm_img(X_train)
    X_test  = norm_img(X_test)

    # === 5. CONVERTIR A TENSORES ===
    X_train_tensor = torch.tensor(X_train).permute(0, 3, 1, 2)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor  = torch.tensor(X_test).permute(0, 3, 1, 2)
    y_test_tensor  = torch.tensor(y_test, dtype=torch.long)

    # ... resto del código igual ...


    # === CONVERSIÓN A TENSORES ===
    X_train_tensor = torch.tensor(X_train).permute(0, 3, 1, 2)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor  = torch.tensor(X_test).permute(0, 3, 1, 2)
    y_test_tensor  = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset  = TensorDataset(X_test_tensor, y_test_tensor)

    val_size = int(0.1 * len(train_dataset))
    train_dataset, val_dataset = random_split(train_dataset, [len(train_dataset)-val_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # === MODELO ===
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # === ENTRENAMIENTO ===
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == lbls).sum().item()
            total += lbls.size(0)

        train_loss = running_loss / total
        train_acc = correct / total * 100

        model.eval()
        val_labels, val_preds = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                outputs = model(imgs)
                val_labels.extend(lbls.cpu().numpy())
                val_preds.extend(outputs.argmax(1).cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds) * 100
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] Loss: {train_loss:.4f} Train: {train_acc:.2f}% Val: {val_acc:.2f}%")

    # === EVALUACIÓN ===
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            outputs = model(imgs)
            all_labels.extend(lbls.cpu().numpy())
            all_preds.extend(outputs.argmax(1).cpu().numpy())

    save_confusion_and_csv(all_labels, all_preds, CLASS_NAMES, MODEL_NAME, CSV_PATH, SAVE_GRAPHICS_PATH)

    # === GRAD-CAM ===
    sample_imgs, sample_lbls = next(iter(test_loader))
    generate_gradcam_auto(
        model,
        sample_imgs[0].unsqueeze(0),
        int(sample_lbls[0].item()),
        SAVE_GRAPHICS_PATH,
        class_names=CLASS_NAMES,
        model_name=MODEL_NAME,
        mean=imagenet_mean,
        std=imagenet_std,
        device=DEVICE
    )


# ==============================================================
# EJECUCIÓN
# ==============================================================
if __name__ == "__main__":
    run_images_pipeline_efficientnet()

