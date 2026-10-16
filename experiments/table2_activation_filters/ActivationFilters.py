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
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import os, math, numpy as np, cv2, torch
from pytorch_grad_cam import LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn as nn
from pytorch_grad_cam import LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn as nn
import pandas as pd
from copy import deepcopy
import gc
import random
from collections import OrderedDict

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
MODEL_NAME = "ActivationsFunsctions"

# Ruta base donde quieres guardar resultados
GRAPHICS_PATH = "/data/home/rtabares/Cervical/Cervical_clean/Pruebas_limpias/graficas"
CSV_PATH = "/data/home/rtabares/Cervical/Coat/Loss_functions/resultados.csv"


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

# 1️⃣ Crear validación desde el entrenamiento
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# 2️⃣ Resize
X_train_resized = resize_images(X_train_split, target_size)
X_val_resized   = resize_images(X_val_split, target_size)
X_test_resized  = resize_images(X_test, target_size)

# 3️⃣ Conversión a float32
X_train_resized = X_train_resized.astype(np.float32)
X_val_resized   = X_val_resized.astype(np.float32)
X_test_resized  = X_test_resized.astype(np.float32)

# 4️⃣ Calcular media y desviación solo del set de entrenamiento
mean = X_train_resized.mean(axis=(0, 1, 2))
std  = X_train_resized.std(axis=(0, 1, 2))
epsilon = 1e-8





def apply_sharpen(img_array):
    # Asegurar rango [0,255] y tipo uint8 para OpenCV
    img_255 = np.clip(img_array * 255, 0, 255).astype(np.uint8)

    # Convertir a RGB si la imagen es en escala de grises o tiene un solo canal
    if img_255.ndim == 2:
        img_255 = cv2.cvtColor(img_255, cv2.COLOR_GRAY2RGB)
    elif img_255.shape[-1] == 1:
        img_255 = cv2.cvtColor(img_255, cv2.COLOR_GRAY2RGB)

    # Kernel de sharpen (realce de bordes y contraste local)
    kernel_sharp = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    # Aplicar convolución
    img_sharp = cv2.filter2D(img_255, -1, kernel_sharp)

    # Normalizar de nuevo a [0,1] para mantener consistencia con el pipeline
    img_sharp = img_sharp.astype(np.float32) / 255.0

    return img_sharp



def apply_clahe(img_array):
    """Aplica CLAHE en el espacio LAB."""
    img_255 = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img_255, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    return img_clahe.astype(np.float32) / 255.0


def apply_gamma(img_array, gamma=1.5):
    """Aplica corrección gamma (realza brillo/contraste)."""
    img_255 = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    corrected = cv2.LUT(img_255, table)
    return corrected.astype(np.float32) / 255.0


def apply_unsharp(img_array, sigma=1.0, strength=1.5):
    """Enfoca bordes con Unsharp Mask."""
    img = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    unsharp = cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)
    return unsharp.astype(np.float32) / 255.0


def apply_adaptive_contrast(img_array, clip_limit=2.0, tile_size=(8, 8)):
    """Realce adaptativo de contraste local + suavizado bilateral."""
    img_255 = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    img_lab = cv2.cvtColor(img_255, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_enhanced = clahe.apply(l)
    merged = cv2.merge((l_enhanced, a, b))
    img_enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    img_final = cv2.bilateralFilter(img_enhanced, 9, 75, 75)
    return img_final.astype(np.float32) / 255.0


def apply_soft_edge_enhance(img_array, alpha=1.2, beta=0.7):
    """Realce de bordes suaves con Laplaciano + mezcla."""
    img_255 = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    lap = cv2.Laplacian(img_255, cv2.CV_64F, ksize=3)
    lap = np.clip(lap, -255, 255)
    enhanced = cv2.convertScaleAbs(img_255 + alpha * lap)
    blended = cv2.addWeighted(enhanced, beta, img_255, 1 - beta, 0)
    return blended.astype(np.float32) / 255.0


def apply_nlmeans_denoise(img_array, h=10, template=7, search=21):
    """Reducción de ruido no local preservando bordes."""
    img_255 = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    denoised = cv2.fastNlMeansDenoisingColored(img_255, None, h, h, template, search)
    return denoised.astype(np.float32) / 255.0


def apply_clahe_yuv(img_array):
    """Aplica CLAHE sobre el canal Y (luminancia) del espacio YUV."""
    img_255 = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    img_yuv = cv2.cvtColor(img_255, cv2.COLOR_RGB2YUV)
    y_channel = img_yuv[:, :, 0]
    y_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(y_channel)
    img_yuv[:, :, 0] = y_clahe
    img_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return img_rgb.astype(np.float32) / 255.0


def denoise_colpo(img_array):
    """Reducción de ruido combinada (NLMeans + blur)."""
    img_255 = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    img_denoised = cv2.fastNlMeansDenoisingColored(img_255, None, 10, 10, 7, 21)
    img_blurred = cv2.GaussianBlur(img_denoised, (3, 3), 0)
    return img_blurred.astype(np.float32) / 255.0



FILTERS = {
    "sharpen": apply_sharpen,
    "clahe": apply_clahe,
    "gamma": apply_gamma,
    "unsharp": apply_unsharp,
    "adaptive_contrast": apply_adaptive_contrast,
    "soft_edge_enhance": apply_soft_edge_enhance,
    "nlmeans_denoise": apply_nlmeans_denoise,
    "clahe_yuv": apply_clahe_yuv,
    "denoise_colpo": denoise_colpo
}






def preprocess_images(images, filter_func, mean, std):
    imgs = np.array([filter_func(img) for img in images])
    if imgs.max() > 1.5:
        imgs = imgs.astype(np.float32) / 255.0
    else:
        imgs = imgs.astype(np.float32)
    imgs = (imgs - mean) / (std + 1e-8)
    return imgs

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





# ------------ 1) Definir activaciones (7) ------------
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Swish(nn.Module):  # SiLU (PyTorch tiene nn.SiLU)
    def forward(self, x):
        return x * torch.sigmoid(x)
    
# === Nuevas funciones ===

class TanhExp(nn.Module):
    """TanhExp: tanh(exp(x))"""
    def forward(self, x):
        return x * torch.tanh(torch.exp(x))


class LiSHT(nn.Module):
    """LiSHT: Linear Scaled Hyperbolic Tangent"""
    def forward(self, x):
        return x * torch.tanh(x)


class Snake(nn.Module):
    """Snake activation: periodic, útil para señales continuas"""
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x):
        return x + (1.0 / self.alpha) * torch.sin(self.alpha * x) ** 2


class ESwish(nn.Module):
    """E-Swish: β * x * sigmoid(x)"""
    def __init__(self, beta=1.5):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return self.beta * x * torch.sigmoid(x)

# Factory helper to return nn.Module instance (safe for stateful a la APL)
def get_activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.01)
    if name == "gelu":
        return nn.GELU()
    if name in ["silu", "swish"]:
        return nn.SiLU()
    if name == "mish":
        return Mish()
    if name == "elu":
        return nn.ELU(alpha=1.0)
    if name == "selu":
        return nn.SELU()
    if name == "tanhexp":
        return TanhExp()
    if name == "prelu":
        return nn.PReLU()
    if name == "lisht":
        return LiSHT()
    if name == "snake":
        return Snake()
    if name in ["eswish", "e-swish"]:
        return ESwish()
    raise ValueError(f"Activación no soportada: {name}")




# ------------ 2) CustomCoaT que permite cambiar la activación ------------
class CustomCoaT(nn.Module):
    def __init__(self, activation_name="relu", num_classes=3, pretrained=True, dropout=0.4):
        super().__init__()
        # Backbone sin cabeza
        self.backbone = timm.create_model("coat_lite_medium", pretrained=pretrained, num_classes=0, img_size=224)
        in_feats = getattr(self.backbone, "num_features", None)
        if in_feats is None:
            # fallback: intentar inferir
            raise RuntimeError("No se pudo inferir 'num_features' del backbone. Verifica timm version.")
        act_layer = get_activation(activation_name)
        # cabeza simple: Linear -> Act -> Dropout -> Linear
        self.classifier = nn.Sequential(
            nn.Linear(in_feats, 512),
            act_layer,
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # backbone devuelve (B, C) cuando num_classes=0 en timm
        feat = self.backbone(x)
        out = self.classifier(feat)
        return out

# ------------ 3) Funciones utilitarias ------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_cuda():
    torch.cuda.empty_cache()
    gc.collect()

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
    """
    Encuentra una buena capa objetivo para Grad-CAM:
      - Preferencia por Conv2d (CNN).
      - Si no existe Conv2d, toma la última LayerNorm o BatchNorm
        y devuelve un reshape_transform que maneja (B, N, C) -> (B, C, H, W).
    Devuelve (layer, reshape_transform_or_None, layer_name).
    """
    # 1) Buscar Conv2d (mejor para superposición espacial directa)
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d):
            return module, None, name

    # 2) Buscar LayerNorm o BatchNorm (posible ViT/CoaT)
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            def reshape_transform(tensor):
                # tensor puede ser (B, C, H, W) o (B, N, C)
                if tensor.ndim == 4:
                    return tensor
                if tensor.ndim == 3:
                    # manejar token de clase: si N == H*W + 1 -> asumimos token en pos 0
                    B, N, C = tensor.shape
                    # si N is perfect square -> no class token
                    h = int(math.sqrt(N))
                    if h * h == N:
                        # no class token
                        result = tensor.reshape(B, h, h, C).permute(0, 3, 1, 2)
                        return result
                    else:
                        # quitamos token 0 (posible class token)
                        result = tensor[:, 1:, :].reshape(B, int(math.sqrt(N-1)), int(math.sqrt(N-1)), C)
                        result = result.permute(0, 3, 1, 2)
                        return result
                raise ValueError("reshape_transform: tensor.ndim inesperado: %d" % tensor.ndim)
            return module, reshape_transform, name

    # 3) Fallback: usar la última módulo registrado (último name)
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
    """
    Versión robusta de Grad-CAM que detecta capa objetivo automáticamente
    y maneja modelos CNN y transformer-like (CoaT, ViT).
    - image_tensor: tensor shape (1, C, H, W) en el formato que sale del DataLoader (ya normalizado).
    - mean/std: arrays (3,) que se usan para desnormalizar (valores en [0,1]).
    """

    # --- checks básicos ---
    if image_tensor.dim() != 4 or image_tensor.size(0) != 1:
        raise ValueError("image_tensor debe tener shape (1, C, H, W)")

    # asegurar float32 y en device
    image_tensor = image_tensor.to(device).float()

    # encontrar capa objetivo y reshape_transform
    target_layer, reshape_transform, layer_name = _find_target_layer_and_transform(model)
    if verbose:
        print(f"[GradCAM] target layer: {layer_name}, reshape_transform: {reshape_transform is not None}")

    # crear cam con o sin reshape
    if reshape_transform is not None:
        cam = LayerCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform)
    else:
        cam = LayerCAM(model=model, target_layers=[target_layer])

    # preparar imagen para visualización: revertir normalización a [0,1] float
    img_np = image_tensor[0].cpu().permute(1, 2, 0).numpy()  # (H,W,C) pero normalizado

    if (mean is None) or (std is None):
        # intentar usar valores globales si no se pasaron
        raise ValueError("Pasa mean y std para revertir la normalización (arrays de 3 elementos).")
    mean_arr = np.array(mean).reshape(1,1,-1).astype(np.float32)
    std_arr  = np.array(std).reshape(1,1,-1).astype(np.float32)

    img_vis = (img_np * std_arr) + mean_arr     # revertir (X - mean)/std -> X
    img_vis = np.clip(img_vis, 0.0, 1.0).astype(np.float32)

    # generar el cam (forma: (batch, Hc, Wc) o (batch, 1, Hc, Wc) depende)
    targets = [ClassifierOutputTarget(int(target_class))]
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets, eigen_smooth=use_eigen_smooth)

    # grayscale_cam puede ser (1, Hc, Wc) o (1,1,Hc,Wc) -> manejar
    if grayscale_cam.ndim == 4 and grayscale_cam.shape[1] == 1:
        grayscale_cam = grayscale_cam[:, 0, :, :]  # -> (1, Hc, Wc)

    # extraer 2D map
    cam_map = grayscale_cam[0]   # (Hc, Wc)

    # redimensionar CAM al tamaño de la imagen
    H, W, _ = img_vis.shape
    cam_map_resized = cv2.resize(cam_map, (W, H), interpolation=cv2.INTER_LINEAR)
    cam_map_resized = np.clip(cam_map_resized, 0.0, 1.0)

    # generar visualización RGB uint8 [0,255]
    visualization = show_cam_on_image(img_vis, cam_map_resized, use_rgb=True, colormap=cv2.COLORMAP_JET)

    # preparar guardado
    os.makedirs(save_path, exist_ok=True)
    target_name = (class_names[int(target_class)] if class_names is not None else f"class_{int(target_class)}")
    fname = f"gradcam_{model_name or 'model'}_{target_name}.png"
    file_path = os.path.join(save_path, fname)

    # visualization ya es uint8 (0..255). Guardar correctamente en BGR:
    if visualization.dtype != np.uint8:
        visualization = (visualization * 255).astype(np.uint8)
    visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, visualization_bgr)

    if verbose:
        print(f"[GradCAM] guardado en: {file_path}    (img shape: {img_vis.shape}, cam shape: {cam_map_resized.shape})")

    return file_path

# =========================================================
# 6. Entrenamiento
# =========================================================
def train_one_model_with_filter(
    activation_name, filter_name, filter_func,
    X_train_orig, y_train, X_val_orig, y_val,
    epochs, lr, batch_size,
    save_base_path, device, num_classes, class_names, mean, std, seed=42
):
    """
    Entrena un modelo con una función de activación y un filtro de preprocesamiento específico.
    Se asegura de que las imágenes vuelvan a su forma original en cada filtro.
    Guarda:
      - Métricas por epoch
      - Curva ROC
      - Grad-CAM
      - Matriz de confusión
      - CSV con métricas finales
    """
    set_seed(seed)

    # === Crear carpeta de guardado por activación y filtro ===
    save_dir = os.path.join(save_base_path, f"activation_{activation_name}", f"filter_{filter_name}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n🧪 Entrenando {activation_name} con filtro {filter_name}")


    X_train_filt = preprocess_images(X_train_orig, filter_func, mean, std)
    X_val_filt   = preprocess_images(X_val_orig,   filter_func, mean, std)


    # Dataset final
    train_dataset = NPYDataset(X_train_filt, y_train)
    val_dataset   = NPYDataset(X_val_filt, y_val)


    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    # === Inicializar modelo y entrenamiento ===
    model = CustomCoaT(activation_name, num_classes=num_classes, pretrained=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    best_labels, best_probs = None, None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        # =================== VALIDACIÓN ===================
        model.eval()
        labels_all, preds_all, probs_all = [], [], []
        with torch.no_grad():
            for imgs_v, labels_v in val_loader:
                imgs_v, labels_v = imgs_v.to(device), labels_v.to(device)
                outputs_v = model(imgs_v)
                probs = torch.softmax(outputs_v, dim=1)
                _, preds = torch.max(outputs_v, 1)

                labels_all.extend(labels_v.cpu().numpy())
                preds_all.extend(preds.cpu().numpy())
                probs_all.extend(probs.cpu().numpy())

        # --- Métricas ---
        f1 = f1_score(labels_all, preds_all, average="macro")
        acc = np.mean(np.array(preds_all) == np.array(labels_all)) * 100
        print(f"[{activation_name} | {filter_name}] Epoch {epoch}/{epochs} - Val Acc: {acc:.2f}% - F1: {f1:.4f}")

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_labels = labels_all
            best_probs = probs_all
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_{activation_name}_{filter_name}.pth"))

    # === Curva ROC ===
    if best_labels is not None and best_probs is not None:
        save_roc_curve(np.array(best_labels), np.array(best_probs), class_names, save_dir, f"{activation_name}_{filter_name}")

    # === Grad-CAM ===
    imgs_sample, labels_sample = next(iter(val_loader))
    img_one = imgs_sample[0].unsqueeze(0)
    label_one = labels_sample[0].item()

    generate_gradcam_new(
        model=model,
        image_tensor=img_one,
        target_class=label_one,
        save_path=save_dir,
        class_names=class_names,
        model_name=f"{activation_name}_{filter_name}_best",
        mean=mean,
        std=std
    )

    # === Matriz de confusión ===
    preds_final = np.argmax(best_probs, axis=1)
    cm = confusion_matrix(best_labels, preds_final)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta real")
    plt.title(f"Confusion Matrix - {activation_name} ({filter_name})")
    plt.savefig(os.path.join(save_dir, f"confusion_{activation_name}_{filter_name}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # === Guardar resultados en CSV ===
    csv_path = os.path.join(save_dir, f"resultados_{activation_name}_{filter_name}.csv")
    precision = precision_score(best_labels, preds_final, average="macro")
    recall = recall_score(best_labels, preds_final, average="macro")
    acc = accuracy_score(best_labels, preds_final)

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Activación", "Filtro", "Accuracy", "Precision", "Recall", "F1-score"])
        writer.writerow([activation_name, filter_name, f"{acc:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{best_val_f1:.4f}"])

    clean_cuda()
    return {"activation": activation_name, "filter": filter_name, "best_f1": best_val_f1}

def evaluate_saved_model(
    activation_name, filter_name, filter_func,
    X_test_orig, y_test,
    device, num_classes, class_names, mean, std,
    batch_size=16,
    base_path="/data/home/rtabares/Cervical/Coat/Loss_functions/"
):
    """
    Evalúa un modelo entrenado (best_model_activation_filter.pth) en el conjunto de prueba
    aplicando su filtro correspondiente.
    Guarda métricas, matriz de confusión, ROC y Grad-CAM.
    """

    target_size = (224, 224)  # o (448, 448) según el modelo que entrenaste

    # Redimensionar si no tiene el tamaño esperado
    if X_test_orig.shape[1:3] != target_size:
        X_test_orig = np.array([cv2.resize(img, target_size) for img in X_test_orig])

    # Aplicar filtro y normalización
    X_test_filt = preprocess_images(X_test_orig, filter_func, mean, std)


    # === 1️⃣ Preparar rutas ===
    save_dir = os.path.join(base_path, f"activation_{activation_name}", f"filter_{filter_name}")
    model_path = os.path.join(save_dir, f"best_model_{activation_name}_{filter_name}.pth")

    if not os.path.exists(model_path):
        print(f"⚠️ No se encontró modelo para {activation_name} + {filter_name}: {model_path}")
        return None

    print(f"\n=== 🧠 Evaluando modelo guardado: {activation_name} ({filter_name}) ===")
    print(f"📂 Cargando desde: {model_path}")

    # === 2️⃣ Cargar modelo ===
    model = CustomCoaT(activation_name, num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # === 3️⃣ Aplicar filtro al conjunto de prueba ===
    X_test_filt = preprocess_images(X_test_orig, filter_func, mean, std)
    test_dataset = NPYDataset(X_test_filt, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # === 4️⃣ Evaluación ===
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # === 5️⃣ Métricas ===
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro")
    rec = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")

    print(f"✅ Test ({activation_name} | {filter_name}) -> Acc: {acc:.4f}, Prec: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    # === 6️⃣ Matriz de confusión ===
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta real")
    plt.title(f"Matriz de confusión (Test) - {activation_name} ({filter_name})")
    cm_path = os.path.join(save_dir, f"confusion_matrix_TEST_{activation_name}_{filter_name}.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Matriz de confusión guardada en {cm_path}")

    # === 7️⃣ Curva ROC ===
    try:
        save_roc_curve(all_labels, all_probs, class_names, save_dir, f"{activation_name}_{filter_name}_TEST")
    except Exception as e:
        print(f"⚠️ No se pudo generar ROC para {activation_name} ({filter_name}): {e}")

    # === 8️⃣ Grad-CAM ===
    imgs_sample, labels_sample = next(iter(test_loader))
    img_one = imgs_sample[0].unsqueeze(0).to(device)
    label_one = labels_sample[0].item()
    try:
        generate_gradcam_new(
            model=model,
            image_tensor=img_one,
            target_class=label_one,
            save_path=save_dir,
            class_names=class_names,
            model_name=f"{activation_name}_{filter_name}_TEST_best",
            mean=mean,
            std=std
        )
    except Exception as e:
        print(f"⚠️ Error generando Grad-CAM para {activation_name} ({filter_name}): {e}")

    # === 9️⃣ Guardar CSV individual ===
    csv_path = os.path.join(save_dir, f"resultados_TEST_{activation_name}_{filter_name}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Activation", "Filter", "Accuracy", "Precision", "Recall", "F1"])
        writer.writerow([activation_name, filter_name, f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}"])
    print(f"📄 Resultados guardados en {csv_path}")

    clean_cuda()
    return {"activation": activation_name, "filter": filter_name, "acc": acc, "prec": prec, "rec": rec, "f1": f1}



# =========================================================
# 7. Experimentos múltiples
# =========================================================
def run_activation_experiments(activations, filters, **kwargs):
    """
    Ejecuta entrenamiento para todas las combinaciones de:
    - Función de activación
    - Filtro de preprocesamiento

    Cada combinación entrena un modelo independiente y guarda resultados.
    """
    results = []
    for act in activations:
        for filt_name, filt_func in filters.items():
            print(f"\n=== Entrenando {act} con filtro {filt_name} ===")
            res = train_one_model_with_filter(
                activation_name=act,
                filter_name=filt_name,
                filter_func=filt_func,
                **kwargs
            )
            results.append(res)

    df = pd.DataFrame(results)
    print("\n✅ Resumen general de todos los experimentos:")
    print(df)

    # Guardar CSV global
    csv_global = os.path.join(kwargs["save_base_path"], "resultados_globales_PTH3.csv")
    df.to_csv(csv_global, index=False)
    print(f"📊 Resultados globales guardados en: {csv_global}")

    return df



# =========================================================
# 8. Ejemplo de uso
# =========================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Funciones de activación ===
    activations = ["ReLU",
    "GELU",
    "SiLU",
    "Mish",
    "TanhExp",
    "PReLU",
    "LiSHT",
    "Snake",
    "E-Swish"]

    # === ENTRENAMIENTO PRINCIPAL ===
    summary = run_activation_experiments(
        activations=activations,
        filters=FILTERS,
        X_train_orig=X_train_resized,
        y_train=y_train_split,
        X_val_orig=X_val_resized,
        y_val=y_val_split,
        epochs=50,
        lr=1e-4,
        batch_size=batch_size,
        save_base_path="/data/home/rtabares/Cervical/Coat/Loss_functions/",
        device=device,
        num_classes=num_classes,
        class_names=class_names,
        mean=mean,
        std=std,
        seed=42
    )

    # === EVALUACIÓN POSTERIOR (modelos ya entrenados) ===
    try:
        results = []
        for act in activations:
            for filt_name, filt_func in FILTERS.items():
                res = evaluate_saved_model(
                    activation_name=act,
                    filter_name=filt_name,
                    filter_func=filt_func,
                    X_test_orig=X_test,
                    y_test=y_test,
                    device=device,
                    num_classes=num_classes,
                    class_names=class_names,
                    mean=mean,
                    std=std,
                    batch_size=batch_size,
                    base_path="/data/home/rtabares/Cervical/Coat/Loss_functions/"
                )
                if res:
                    results.append(res)



        if results:
            df = pd.DataFrame(results)
            resumen_path = "/data/home/rtabares/Cervical/Coat/Loss_functions/resultados_globales_PTH3.csv"
            df.to_csv(resumen_path, index=False)
            print(f"\n✅ Resumen global guardado en {resumen_path}")
            print(df)
    except NameError:
        print("⚠️ 'test_loader' no está definido, se omitió la evaluación final.")

