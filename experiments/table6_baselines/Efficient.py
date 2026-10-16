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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import cv2
from pytorch_grad_cam import LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ===============================
# Configuración general
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4

# Fijar semilla para reproducibilidad
torch.manual_seed(42)
np.random.seed(42)

# Rutas
X_TRAIN_PATH = "/data/home/rtabares/Cervical/Cervical_clean/X_train.npy"
Y_TRAIN_PATH = "/data/home/rtabares/Cervical/Cervical_clean/y_train.npy"
X_TEST_PATH  = "/data/home/rtabares/Cervical/Cervical_clean/X_test.npy"
Y_TEST_PATH  = "/data/home/rtabares/Cervical/Cervical_clean/y_test.npy"

SAVE_GRAPHICS_PATH = "/data/home/rtabares/Cervical/Cervical_clean/GradCAM"
CSV_PATH = "/data/home/rtabares/Cervical/Cervical_clean/Pruebas_limpias/resultados.csv"
MODEL_NAME = "EfficientNet_Rafa"

# === Valores de normalización (Imagenet) ===
imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
imagenet_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ===============================
# Funciones auxiliares
# ===============================

def save_confusion_and_csv(all_labels, all_preds, class_names, model_name, csv_path, save_image_dir=None):
    """
    Guarda métricas y matriz de confusión en CSV y adicionalmente guarda
    la matriz de confusión como imagen PNG en `save_image_dir` (si se pasa),
    y en la carpeta del csv por defecto.
    """
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    # Guardar CSV (misma lógica que antes)
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

    # ---------- Guardar la matriz de confusión como imagen ----------
    # Decide carpeta de guardado de la imagen
    if save_image_dir is None:
        # por defecto usa la carpeta del CSV
        save_image_dir = os.path.dirname(csv_path)
        if save_image_dir == "":
            save_image_dir = "."

    os.makedirs(save_image_dir, exist_ok=True)

    # Ploteo y guardado
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

    # También guardar una copia en SAVE_GRAPHICS_PATH si es diferente
    if save_image_dir != SAVE_GRAPHICS_PATH:
        os.makedirs(SAVE_GRAPHICS_PATH, exist_ok=True)
        img_copy_path = os.path.join(SAVE_GRAPHICS_PATH, img_name)
        # leer y reescribir para asegurar copia (cv2 usa BGR)
        img = cv2.imread(img_path)
        if img is not None:
            cv2.imwrite(img_copy_path, img)
            print(f"Matriz de confusión guardada en: {img_copy_path}")
        else:
            print(f"Advertencia: no se pudo leer la imagen guardada en {img_path} para copiarla.")

    print(f"Matriz de confusión (imagen) guardada en: {img_path}")
    return img_path


# === FUNCIÓN DE DETECCIÓN AUTOMÁTICA DE CAPA OBJETIVO ===
def _find_target_layer_and_transform(model):
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
                    h = int(math.sqrt(N)) if int(math.sqrt(N))**2 == N else int(math.sqrt(N - 1))
                    if h * h == N:
                        result = tensor.reshape(B, h, h, C).permute(0, 3, 1, 2)
                    else:
                        result = tensor[:, 1:, :].reshape(B, h, h, C).permute(0, 3, 1, 2)
                    return result
                raise ValueError(f"reshape_transform: tensor.ndim inesperado ({tensor.ndim})")
            return module, reshape_transform, name

    all_mods = list(model.named_modules())
    if len(all_mods) > 0:
        name, module = all_mods[-1]
        return module, None, name
    raise RuntimeError("No se encontró módulo adecuado para Grad-CAM.")


# === IMPLEMENTACIÓN GRAD-CAM ROBUSTA (sin modificar) ===
def generate_gradcam_auto(
    model, image_tensor, target_class, save_path,
    class_names=None, model_name=None,
    mean=None, std=None, device=None,
    use_eigen_smooth=True, verbose=True
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    image_tensor = image_tensor.to(device).float()

    target_layer, reshape_transform, layer_name = _find_target_layer_and_transform(model)
    if verbose:
        print(f"[GradCAM] Capa objetivo: {layer_name}, reshape_transform: {reshape_transform is not None}")

    cam = LayerCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform)

    if mean is None or std is None:
        raise ValueError("Debes pasar mean y std para revertir la normalización.")

    img_np = image_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
    mean_arr = np.array(mean).reshape(1, 1, -1).astype(np.float32)
    std_arr = np.array(std).reshape(1, 1, -1).astype(np.float32)
    img_vis = (img_np * std_arr) + mean_arr
    img_vis = np.clip(img_vis, 0.0, 1.0)

    targets = [ClassifierOutputTarget(int(target_class))]
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets, eigen_smooth=use_eigen_smooth)
    if grayscale_cam.ndim == 4 and grayscale_cam.shape[1] == 1:
        grayscale_cam = grayscale_cam[:, 0, :, :]

    cam_map = grayscale_cam[0]
    H, W, _ = img_vis.shape
    cam_map_resized = cv2.resize(cam_map, (W, H), interpolation=cv2.INTER_LINEAR)
    cam_map_resized = np.clip(cam_map_resized, 0.0, 1.0)

    visualization = show_cam_on_image(img_vis, cam_map_resized, use_rgb=True, colormap=cv2.COLORMAP_JET)
    if visualization.dtype != np.uint8:
        visualization = (visualization * 255).astype(np.uint8)

    os.makedirs(save_path, exist_ok=True)
    target_name = class_names[int(target_class)] if class_names is not None else f"class_{int(target_class)}"
    fname = f"gradcam_{model_name or 'model'}_{target_name}.png"
    file_path = os.path.join(save_path, fname)

    cv2.imwrite(file_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    if verbose:
        print(f"[GradCAM] Guardado en: {file_path}")

    return file_path


# ===============================
# Pipeline principal
# ===============================
def run_images_pipeline_efficientnet():
    # Cargar datos
    X_train = np.load(X_TRAIN_PATH)
    y_train = np.load(Y_TRAIN_PATH)
    X_test  = np.load(X_TEST_PATH)
    y_test  = np.load(Y_TEST_PATH)

    CLASS_NAMES = [f"Clase_{i}" for i in np.unique(y_train)]
    num_classes = len(CLASS_NAMES)

    # Normalizar a [0,1]
    X_train = X_train.astype(np.float32)
    X_test  = X_test.astype(np.float32)

    # Normalizar con ImageNet stats (coherente con pesos preentrenados)
    def norm_img(arr):
        return (arr - imagenet_mean) / imagenet_std

    X_train = norm_img(X_train)
    X_test  = norm_img(X_test)

    # Tensor datasets
    X_train_tensor = torch.tensor(X_train).permute(0, 3, 1, 2)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor  = torch.tensor(X_test).permute(0, 3, 1, 2)
    y_test_tensor  = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset  = TensorDataset(X_test_tensor, y_test_tensor)

    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    # Split reproducible (semilla ya fijada arriba)
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Modelo EfficientNet con pesos preentrenados de ImageNet
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Entrenamiento
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
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

    # Evaluación en test
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            outputs = model(imgs)
            all_labels.extend(lbls.cpu().numpy())
            all_preds.extend(outputs.argmax(1).cpu().numpy())

    # Guardar CSV y además la imagen de la matriz de confusión
    img_path = save_confusion_and_csv(all_labels, all_preds, CLASS_NAMES, MODEL_NAME, CSV_PATH, save_image_dir=SAVE_GRAPHICS_PATH)

    # Grad-CAM para una muestra
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


# ===============================
# Ejecutar
# ===============================
if __name__ == "__main__":
    run_images_pipeline_efficientnet()
