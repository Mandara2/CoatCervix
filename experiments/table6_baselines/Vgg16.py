# =========================================================
# 1. Importar librerías
# =========================================================
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import models, transforms
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import timm
import csv
from datetime import datetime
from pytorch_grad_cam import LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn as nn
import math

# =========================================================
# Configuración inicial
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3
batch_size = 16
epochs = 50
lr = 1e-4
class_names = ["Clase 0","Clase 1","Clase 2"] # Nombres de las clases


# Variable global con nombre del experimento/modelo
MODEL_NAME = "vgg16"

# Ruta base donde quieres guardar resultados
GRAPHICS_PATH = "/data/home/rtabares/Cervical/Cervical_clean/Pruebas_limpias/graficas"
CSV_PATH = "/data/home/rtabares/Cervical/Cervical_clean/Pruebas_limpias/resultados.csv"


# Carpeta final combinando ruta base + nombre del modelo
SAVE_GRAPHICS_PATH = os.path.join(GRAPHICS_PATH, MODEL_NAME)



# =========================================================
# Cargar datos .npy
# =========================================================

X_train = np.load("/data/home/rtabares/Cervical/Cervical_clean/X_train.npy")
X_test  = np.load("/data/home/rtabares/Cervical/Cervical_clean/X_test.npy")

y_train = np.load("/data/home/rtabares/Cervical/Cervical_clean/y_train.npy")
y_test  = np.load("/data/home/rtabares/Cervical/Cervical_clean/y_test.npy")


# =========================================================
# Redimensionar imágenes a 224x224 (o 448x448)
# =========================================================
target_size = (224, 224)  # Cambia a 448 si quieres
def resize_images(images, size):
    resized = []
    for img in images:
        # Si grayscale, expandir a 3 canales
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[-1] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_resized = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        resized.append(img_resized)
    return np.array(resized)

X_train = resize_images(X_train, target_size)
X_test  = resize_images(X_test, target_size)
X_val  = resize_images(X_test, target_size)



# =========================================================
# Slipt
# =========================================================


X_train_new, X_val, y_train_new, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)



# =========================================================
# Filtro Sharpen
# =========================================================
def apply_sharpen(images):
    kernel_sharp = np.array([[0,-1,0],
                             [-1,5,-1],
                             [0,-1,0]])
    sharpened = []
    for img in images:
        # Asegurarse que la imagen sea uint8 antes de aplicar el filtro
        if img.dtype != np.uint8:
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (img * 255).astype(np.uint8)
        # Si es grayscale, convertir a RGB
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[-1] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img_sharp = cv2.filter2D(img, -1, kernel_sharp)
        sharpened.append(img_sharp)
    return np.array(sharpened)

X_train = X_train_new
y_train = y_train_new

# Ahora todos los nombres coinciden con el pipeline original
X_train = apply_sharpen(X_train)
X_val   = apply_sharpen(X_val)
X_test  = apply_sharpen(X_test)




# =========================================================
# Normalización [0,1] - ImageNet
# =========================================================

# Paso 2: escalar a [0,1]
X_train = X_train.astype(np.float32) / 255.0
X_val   = X_val.astype(np.float32) / 255.0
X_test  = X_test.astype(np.float32) / 255.0

# Paso 3: calcular mean y std sobre el set ya escalado
mean = X_train.mean(axis=(0,1,2))
std  = X_train.std(axis=(0,1,2))

epsilon = 1e-8  # evitar división por cero

# Paso 4: normalizar con mean y std del train
X_train = (X_train - mean) / (std + epsilon)
X_val   = (X_val - mean) / (std + epsilon)
X_test  = (X_test - mean) / (std + epsilon)

# =========================================================
# Dataset en PyTorch
# =========================================================
class NPYDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(np.transpose(X, (0, 3, 1, 2)))  # [N,C,H,W]
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = NPYDataset(X_train, y_train)
val_dataset   = NPYDataset(X_val, y_val)
test_dataset  = NPYDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# =========================================================
# Modelo: Vgg16 ORIGINAL (PyTorch)
# =========================================================


model = timm.create_model(
    "vgg16", pretrained=True, num_classes=num_classes
)
model = model.to(device)


# =========================================================
# Optimización
# =========================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


# =========================================================
# Entrenamiento
# =========================================================
def train_model(model, train_loader, val_loader, epochs, class_names=None):
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        train_acc = 100 * correct / total

        # Accuracy simple sobre val_loader
        val_acc = evaluate_model(model, val_loader)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {running_loss/len(train_loader.dataset):.4f} "
              f"Train Acc: {train_acc:.2f}% "
              f"Val Acc: {val_acc:.2f}%")

        # --- Seguimiento completo sobre validation ---
        if class_names is not None:
            print("\nEvaluación detallada sobre validation:")
            full_evaluation(model, val_loader, class_names)




# =========================================================
# Evaluación
# =========================================================
def evaluate_model(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
    return 100 * correct / total


#Aquí está la lógica para guardar los resultados en un csv
def full_evaluation(model, loader, class_names, model_name=None, roc_path=None, save_csv=True):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            _, preds = outputs.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())  # 👈 aquí guardas probabilidades

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Métricas globales como antes...
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro")
    rec = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")

    # Matriz de confusión
    cm = confusion_matrix(all_labels, all_preds)

    print("\n--- RESULTADOS ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nMatriz de confusión (valores):")
    print(cm)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta real")
    plt.title("Matriz de confusión")

    os.makedirs(SAVE_GRAPHICS_PATH, exist_ok=True)
    cm_path = os.path.join(SAVE_GRAPHICS_PATH, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    # --- Guardar resultados en CSV ---
    if save_csv and model_name is not None:
        with open(CSV_PATH, mode="a", newline="") as f:
            writer = csv.writer(f)

            # Título de la prueba
            writer.writerow([f"Prueba: {model_name}, Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
            writer.writerow([])

            # Matriz de confusión
            writer.writerow(["Matriz de confusión"])
            writer.writerow(["Clase"] + class_names)
            for i, row in enumerate(cm):
                writer.writerow([class_names[i]] + row.tolist())
            writer.writerow([])

            # Métricas globales
            writer.writerow(["Métricas globales"])
            writer.writerow(["Accuracy", acc])
            writer.writerow(["Precision", prec])
            writer.writerow(["Recall", rec])
            writer.writerow(["F1-score", f1])
            writer.writerow([])

            # Ruta de la curva ROC si existe
            if roc_path is not None:
                writer.writerow(["Ruta curva ROC", roc_path])
                writer.writerow([])

            writer.writerow([])  # Espacio entre pruebas

        print(f"Resultados guardados en {CSV_PATH}")

    return all_labels, all_preds, all_probs




# =========================================================
# 10. Graficas
# =========================================================
def save_roc_curve(y_true, y_pred_probs, class_names, save_path, epoch=None):
    """
    y_true: etiquetas verdaderas (n_samples,)
    y_pred_probs: probabilidades predichas (n_samples, n_classes)
    class_names: lista de nombres de clases
    save_path: carpeta donde guardar la imagen
    epoch: opcional, para diferenciar curvas por epoch
    """
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
    plt.title("ROC Curve" + (f" - Epoch {epoch}" if epoch is not None else ""))
    plt.legend(loc="lower right")

    # Nombre del archivo
    file_name = "roc_curve.png" if epoch is None else f"roc_curve_epoch_{epoch}.png"
    file_path = os.path.join(save_path, file_name)
    plt.savefig(file_path)
    plt.close()
    print(f"Curva ROC guardada en: {file_path}")



def _find_target_layer_and_transform(model):
    # 1) Si es modelo tipo VGG (o tiene atributo features), preferir la última Conv2d en model.features
    if hasattr(model, "features"):
        # recorrer features de atrás hacia adelante para encontrar Conv2d
        for module in reversed(list(model.features)):
            if isinstance(module, nn.Conv2d):
                return module, None, "features::<last_conv_in_features>"
        # si no encontró, continuar con búsqueda general

    # 2) Búsqueda general: Conv2d
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d):
            return module, None, name

    # 3) Buscar LayerNorm/BatchNorm para modelos tipo ViT/CoaT
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            def reshape_transform(tensor):
                if tensor.ndim == 4:
                    return tensor
                if tensor.ndim == 3:
                    B, N, C = tensor.shape
                    h = int(math.sqrt(N))
                    if h * h == N:
                        result = tensor.reshape(B, h, h, C).permute(0, 3, 1, 2)
                        return result
                    else:
                        result = tensor[:, 1:, :].reshape(B, int(math.sqrt(N-1)), int(math.sqrt(N-1)), C)
                        return result.permute(0, 3, 1, 2)
                raise ValueError("reshape_transform: tensor.ndim inesperado: %d" % tensor.ndim)
            return module, reshape_transform, name

    # 4) Fallback: último módulo
    all_mods = list(model.named_modules())
    if len(all_mods) > 0:
        name, module = all_mods[-1]
        return module, None, name
    raise RuntimeError("No se encontró módulo en el modelo (esto no debería pasar).")

def generate_gradcam_new(
    model, image_tensor, target_class, save_path,
    class_names=None, model_name=None, mean=None, std=None,
    use_eigen_smooth=True, verbose=True
):
    # checks
    if image_tensor.dim() != 4 or image_tensor.size(0) != 1:
        raise ValueError("image_tensor debe tener shape (1, C, H, W)")

    image_tensor = image_tensor.to(device).float()

    # encontrar capa objetivo
    target_layer, reshape_transform, layer_name = _find_target_layer_and_transform(model)
    if verbose:
        print(f"[GradCAM] target layer: {layer_name}, reshape_transform: {reshape_transform is not None}")

    # crear LayerCAM con o sin reshape
    cam = LayerCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform) if reshape_transform is not None else LayerCAM(model=model, target_layers=[target_layer])

    # preparar imagen de visualización (desnormalizar)
    if mean is None or std is None:
        raise ValueError("Pasa mean y std para revertir la normalización (arrays de 3 elementos).")

    mean_arr = np.array(mean).reshape(1,1,-1).astype(np.float32)
    std_arr  = np.array(std).reshape(1,1,-1).astype(np.float32)

    img_np = image_tensor[0].cpu().permute(1,2,0).numpy()  # (H,W,C) normalizado
    img_vis = (img_np * std_arr) + mean_arr
    img_vis = np.clip(img_vis, 0.0, 1.0).astype(np.float32)

    # generar CAM
    targets = [ClassifierOutputTarget(int(target_class))]
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets, eigen_smooth=use_eigen_smooth)

    # normalizar dimensiones si LayerCAM devuelve (1,1,H,W)
    if grayscale_cam.ndim == 4 and grayscale_cam.shape[1] == 1:
        grayscale_cam = grayscale_cam[:, 0, :, :]
    cam_map = grayscale_cam[0]  # (Hc, Wc)

    # redimensionar al tamaño de la imagen
    H, W, _ = img_vis.shape
    cam_map_resized = cv2.resize(cam_map, (W, H), interpolation=cv2.INTER_LINEAR)
    cam_map_resized = np.clip(cam_map_resized, 0.0, 1.0)

    # show_cam_on_image espera img flotante [0,1] en RGB
    visualization = show_cam_on_image(img_vis, cam_map_resized, use_rgb=True, colormap=cv2.COLORMAP_JET)

    # guardar
    os.makedirs(save_path, exist_ok=True)
    target_name = (class_names[int(target_class)] if class_names is not None else f"class_{int(target_class)}")
    fname = f"gradcam_{model_name or 'model'}_{target_name}.png"
    file_path = os.path.join(save_path, fname)

    if visualization.dtype != np.uint8:
        visualization = (visualization * 255).astype(np.uint8)
    visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, visualization_bgr)
    if verbose:
        print(f"[GradCAM] guardado en: {file_path} (img shape: {img_vis.shape}, cam shape: {cam_map_resized.shape})")
    return file_path



# =========================================================
# 11. Ejecutar entrenamiento + evaluación
# =========================================================

# Entrenamiento con seguimiento completo de métricas en validation
train_model(model, train_loader, val_loader, epochs, class_names=class_names)

print("EVALUACIÓN FINAL")
# Evaluación final sobre test (esto ya devuelve labels, preds y probs si lo añadiste en full_evaluation)
all_labels, all_preds, all_probs = full_evaluation(
    model,
    test_loader,
    class_names,
    model_name=MODEL_NAME,
    roc_path=SAVE_GRAPHICS_PATH,
    save_csv=True
)

# Guardar curva ROC al final después de entrenamiento
save_roc_curve(all_labels, all_probs, class_names, SAVE_GRAPHICS_PATH)

# selecciona sample
sample_images, sample_labels = next(iter(test_loader))
sample_image = sample_images[0].unsqueeze(0)    # (1,C,H,W)
sample_label = int(sample_labels[0])

print("sample_image dtype, shape:", sample_image.dtype, sample_image.shape)
print("mean dtype/shape:", type(mean), np.array(mean).shape, np.array(mean))
print("std  dtype/shape:", type(std),  np.array(std).shape, np.array(std))

# forzar float32
sample_image = sample_image.float()

# llama a la función (verbose True)
generate_gradcam_new(
    model=model,
    image_tensor=sample_image,
    target_class=sample_label,
    save_path=SAVE_GRAPHICS_PATH,
    class_names=class_names,
    model_name=MODEL_NAME,
    mean=mean,
    std=std,
    use_eigen_smooth=True,
    verbose=True
)


