import os
import math
import random
import csv
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from copy import deepcopy

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc, roc_auc_score)
from sklearn.preprocessing import label_binarize

import timm
from torchvision import transforms
from PIL import Image

# Grad-CAM
from pytorch_grad_cam import LayerCAM, GradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ----------------------------
# ---------- CONFIG ----------
# ----------------------------
MODEL_NAME = "HMCFormer_train_only"
GRAPHICS_PATH = os.path.expanduser("~/Cervical/revision_peerj/hmcformer/graficas")
CSV_PATH = os.path.expanduser("~/Cervical/Coat/Pappers/resultados.csv")
SAVE_GRAPHICS_PATH = os.path.join(GRAPHICS_PATH, MODEL_NAME)

# Data .npy paths
X_TRAIN_NPY = os.path.expanduser("~/Cervical/Cervical_clean/X_train.npy")
X_TEST_NPY  = os.path.expanduser("~/Cervical/Cervical_clean/X_test.npy")
Y_TRAIN_NPY = os.path.expanduser("~/Cervical/Cervical_clean/y_train.npy")
Y_TEST_NPY  = os.path.expanduser("~/Cervical/Cervical_clean/y_test.npy")

# Training / CV
NUM_CLASSES = 3
N_SPLITS = 5
SEEDS = [42]
EPOCHS = 40
BATCH_SIZE = 16
LR = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_X, TEST_Y = None, None  # revision PeerJ: test independiente

# Image pipeline
RESIZE_TO = 260
CROP_SIZE = 224

# ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Create folders
os.makedirs(SAVE_GRAPHICS_PATH, exist_ok=True)
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

# ----------------------------
# ------ Utilities ----------
# ----------------------------
def set_seed(seed: int):
    """Fija semilla para reproducibilidad completa."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# ----------------------------
# ---- Preprocessing LT_CLAHE ----
# ----------------------------
def apply_clahe_channel(img_channel, clipLimit=2.0, tileGridSize=(8,8)):
    """Aplicar CLAHE a un canal. Devuelve float [0..1]."""
    if img_channel.dtype != np.uint8:
        tmp = np.clip(img_channel * 255.0, 0, 255).astype(np.uint8)
    else:
        tmp = img_channel
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    res = clahe.apply(tmp)
    return (res.astype(np.float32) / 255.0)

def lt_clahe_lab_ycrcb(image_rgb):
    """
    Enhancement LT_CLAHE descrito en el paper.
    Returns: (enhanced_rgb, enhanced_grayscale) ambos float [0,1]
    """
    img = image_rgb.astype(np.float32)
    if img.max() > 1.1:
        img = img / 255.0
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    # LAB
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    L = lab[..., 0]
    A = lab[..., 1]
    B = lab[..., 2]

    # YCrCb
    ycrcb = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YCrCb).astype(np.float32) / 255.0
    Y = ycrcb[..., 0]
    Cr = ycrcb[..., 1]
    Cb = ycrcb[..., 2]

    # Apply CLAHE
    L_eq = apply_clahe_channel(L)
    Y_eq = apply_clahe_channel(Y)
    A_eq = apply_clahe_channel(A)
    B_eq = apply_clahe_channel(B)
    Cr_eq = apply_clahe_channel(Cr)
    Cb_eq = apply_clahe_channel(Cb)

    # Reconstruir
    lab_enh = np.stack([L_eq, A_eq, B_eq], axis=-1)
    ycrcb_enh = np.stack([Y_eq, Cr_eq, Cb_eq], axis=-1)

    # Convertir a RGB
    lab_rgb = cv2.cvtColor((lab_enh * 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    ycrcb_rgb = cv2.cvtColor((ycrcb_enh * 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB).astype(np.float32) / 255.0

    # Fusión
    fused = (lab_rgb + ycrcb_rgb) / 2.0
    fused = np.clip(fused, 0.0, 1.0)

    # Grayscale mejorado
    gray_enh = cv2.cvtColor((fused * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    return fused, gray_enh

# ----------------------------
# ---- Augmentation pipelines ----
# ----------------------------
def get_train_transforms(resize_to=RESIZE_TO, crop_size=CROP_SIZE):
    """Transforms para entrenamiento con augmentation usando torchvision."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(resize_to),
        transforms.RandomCrop(crop_size),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.08, contrast=0.08),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist())
    ])

def get_val_transforms(resize_to=RESIZE_TO, crop_size=CROP_SIZE):
    """Transforms para validación/test sin augmentation."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(resize_to),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist())
    ])

# ----------------------------
# ---- Dataset con doble rama ----
# ----------------------------
class HMCFormerDataset(Dataset):
    """
    Dataset que retorna dos entradas:
    - transformer_input: RGB normalizado ImageNet (3,H,W)
    - cnn_input: grayscale mejorado como 3 canales duplicados (3,H,W)
    - label: int
    
    IMPORTANTE: Para evitar data leakage, el enhancement LT_CLAHE se aplica
    DESPUÉS del split y dentro del dataset, no antes.
    """
    def __init__(self, X, y, transform=None, apply_enhancement=True):
        """
        Args:
            X: numpy array (N,H,W,3) float32 [0..255] o [0..1]
            y: numpy array (N,) int labels
            transform: albumentations transform (mismo para ambas ramas espacialmente)
            apply_enhancement: si True, aplica LT_CLAHE para rama CNN
        """
        assert X.ndim == 4 and X.shape[-1] == 3, f"X debe ser (N,H,W,3), recibido {X.shape}"
        assert len(X) == len(y), "X e y deben tener mismo tamaño"
        
        self.X = X
        self.y = y
        self.transform = transform
        self.apply_enhancement = apply_enhancement

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx].copy()  # IMPORTANTE: copy para evitar modificar datos originales
        label = int(self.y[idx])

        # Normalizar a [0,1]
        img_f = img.astype(np.float32)
        if img_f.max() > 1.1:
            img_f = img_f / 255.0

        # Preparar imagen para transformer (RGB original)
        tr_img = (img_f * 255).astype(np.uint8)

        # Preparar imagen para CNN (grayscale mejorado)
        if self.apply_enhancement:
            fused_rgb, gray_enh = lt_clahe_lab_ycrcb(img_f)
            cnn_img = (gray_enh * 255).astype(np.uint8)
            # Convertir grayscale a 3 canales para compatibilidad con redes pre-entrenadas
            cnn_img_3c = cv2.cvtColor(cnn_img, cv2.COLOR_GRAY2RGB)
        else:
            gray = cv2.cvtColor((img_f*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            cnn_img_3c = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # Aplicar mismas transformaciones espaciales a ambas ramas
        if self.transform is not None:
            # CRÍTICO: fijar seed para aplicar mismas transformaciones espaciales
            seed = np.random.randint(2147483647)
            
            # Transform para rama transformer
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            tr_tensor = self.transform(tr_img)
            
            # Transform para rama CNN (misma seed = mismas transformaciones espaciales)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            cnn_tensor = self.transform(cnn_img_3c)
        else:
            tr_tensor = torch.from_numpy(np.transpose(tr_img.astype(np.float32)/255.0, (2,0,1))).float()
            cnn_tensor = torch.from_numpy(np.transpose(cnn_img_3c.astype(np.float32)/255.0, (2,0,1))).float()

        return {'tr': tr_tensor, 'cnn': cnn_tensor, 'label': label}

# ----------------------------
# ---- Arquitectura HMCFormer ----
# ----------------------------
class HMCFormerLike(nn.Module):
    def __init__(self,
                 transformer_name="swin_small_patch4_window7_224",
                 cnn_name="efficientnet_b0",
                 pretrained=True,
                 feat_dim=512,
                 num_classes=NUM_CLASSES,
                 img_size=224):
        super().__init__()

        print(f"\n[INIT] Creando modelo con transformer={transformer_name}, cnn={cnn_name}")

        # Transformer branch
        self.transformer_name = transformer_name
        self.transformer = timm.create_model(
            transformer_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )

        emb_dim = getattr(self.transformer, "num_features", 768)
        print(f"[INIT] Transformer embedding dim: {emb_dim}")
        self.pool_proj = nn.Linear(emb_dim, feat_dim)

        # CNN branch
        self.cnn = timm.create_model(
            cnn_name, pretrained=pretrained, num_classes=0, global_pool='avg'
        )
        cnn_dim = self.cnn.num_features
        print(f"[INIT] CNN feature dim: {cnn_dim}")
        self.cnn_proj = nn.Linear(cnn_dim, feat_dim)

        # Fusion
        fusion_dim = feat_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(fusion_dim // 4, num_classes)

    def forward(self, tr_x, cnn_x):
        """
        Args:
            tr_x: tensor (B,3,H,W) - entrada transformer
            cnn_x: tensor (B,3,H,W) - entrada CNN
        Returns:
            logits: tensor (B, num_classes)
        """
        # Transformer forward
        feat_tr = self.transformer.forward_features(tr_x)
        
        # Manejar diferentes formatos de salida
        if feat_tr.ndim == 4:  # (B,H,W,C) o (B,C,H,W)
            if feat_tr.shape[1] > feat_tr.shape[-1]:  # likely (B,C,H,W)
                feat_tr = feat_tr.mean(dim=(2, 3))
            else:  # likely (B,H,W,C)
                feat_tr = feat_tr.permute(0, 3, 1, 2).mean(dim=(2, 3))
        elif feat_tr.ndim == 3:  # (B,N,C) - tokens
            feat_tr = feat_tr.mean(dim=1)
        elif feat_tr.ndim != 2:
            raise ValueError(f"Unexpected transformer output shape: {feat_tr.shape}")
        
        feat_tr = self.pool_proj(feat_tr)

        # CNN forward
        feat_cnn = self.cnn(cnn_x)
        feat_cnn = self.cnn_proj(feat_cnn)

        # Fusion
        fused = torch.cat([feat_tr, feat_cnn], dim=1)
        out = self.fusion(fused)
        logits = self.classifier(out)

        return logits

# ----------------------------
# ---- Métricas clínicas ----
# ----------------------------
def compute_clinical_metrics(y_true, y_pred, average="macro"):
    """Calcula métricas clínicas detalladas."""
    cm = confusion_matrix(y_true, y_pred)
    per_class = {}
    n_classes = cm.shape[0]
    
    for i in range(n_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FN - FP
        
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        
        per_class[i] = {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "TP": int(TP), "FP": int(FP),
            "FN": int(FN), "TN": int(TN)
        }
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "per_class": per_class,
        "confusion": cm
    }

# ----------------------------
# ---- Evaluación completa ----
# ----------------------------
def full_evaluation(model, loader, class_names, device=DEVICE, save_path=None, 
                   model_name=None, save_csv=True):
    """
    Evalúa modelo en loader y guarda métricas y visualizaciones.
    
    Returns:
        tuple: (labels, preds, probs)
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    
    with torch.no_grad():
        for batch in loader:
            tr = batch['tr'].to(device)
            cnn = batch['cnn'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(tr, cnn)
            probs = F.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Calcular métricas
    metrics = compute_clinical_metrics(all_labels, all_preds, average="macro")
    
    print("\n--- RESULTADOS ---")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    print("\nMatriz de confusión:")
    print(metrics['confusion'])

    # Guardar visualizaciones si se especifica path
    if save_path is not None:
        ensure_dir(save_path)
        
        # Confusion matrix
        cm = metrics['confusion']
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicción")
        plt.ylabel("Etiqueta real")
        plt.title("Matriz de confusión")
        cm_path = os.path.join(save_path, "confusion_matrix.png")
        plt.savefig(cm_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Confusion matrix guardada en: {cm_path}")

        # ROC curves
        n_classes = len(class_names)
        y_bin = label_binarize(all_labels, classes=list(range(n_classes)))
        
        plt.figure(figsize=(8, 6))
        for i in range(n_classes):
            if y_bin[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.2f})")
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend(loc="lower right")
        roc_path = os.path.join(save_path, "roc_curves.png")
        plt.savefig(roc_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"ROC curves guardadas en: {roc_path}")

    # Guardar en CSV si se requiere
    if save_csv and model_name is not None:
        ensure_dir(os.path.dirname(CSV_PATH))
        flat = {
            "model": model_name,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "accuracy": metrics['accuracy'],
            "precision_macro": metrics['precision'],
            "recall_macro": metrics['recall'],
            "f1_macro": metrics['f1']
        }
        
        for k, v in metrics['per_class'].items():
            flat[f"class_{k}_sensitivity"] = v['sensitivity']
            flat[f"class_{k}_specificity"] = v['specificity']
            flat[f"class_{k}_TP"] = v['TP']
            flat[f"class_{k}_FP"] = v['FP']
            flat[f"class_{k}_FN"] = v['FN']
            flat[f"class_{k}_TN"] = v['TN']
        
        df = pd.DataFrame([flat])
        df.to_csv(CSV_PATH, mode='a', header=not os.path.exists(CSV_PATH), index=False)
        print(f"Resultados guardados en {CSV_PATH}")

    return all_labels, all_preds, all_probs

# ----------------------------
# ---- Training/Validation ----
# ----------------------------
def train_one_epoch(model, loader, optimizer, criterion, device=DEVICE):
    """Entrena una época."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch in loader:
        tr = batch['tr'].to(device)
        cnn = batch['cnn'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(tr, cnn)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    
    avg_loss = running_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc

def validate_epoch(model, loader, criterion, device=DEVICE):
    """Valida una época."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            tr = batch['tr'].to(device)
            cnn = batch['cnn'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(tr, cnn)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * labels.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    
    avg_loss = running_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc

# ----------------------------
# ---- Entrenamiento de un fold ----
# ----------------------------
def train_fold(fold_id, train_idx, val_idx, X, y, config, seed):
    """
    Entrena un fold completo.
    
    IMPORTANTE: Los índices train_idx y val_idx son mutuamente excluyentes
    garantizando que no haya data leakage.
    """
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"FOLD {fold_id} | SEED {seed}")
    print(f"{'='*60}")
    print(f"Train samples: {len(train_idx)} | Val samples: {len(val_idx)}")
    
    # Verificar que no hay overlap (sanity check)
    assert len(set(train_idx) & set(val_idx)) == 0, "¡DATA LEAKAGE! Train y val se solapan"
    
    # Crear subsets de datos (CRÍTICO: no mezclar train/val)
    train_X = X[train_idx]
    train_y = y[train_idx]
    val_X = X[val_idx]
    val_y = y[val_idx]
    
    # Verificar distribución de clases
    print(f"\nDistribución train: {Counter(train_y)}")
    print(f"Distribución val: {Counter(val_y)}")
    
    # Crear transforms
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    # Crear datasets (enhancement se aplica AQUÍ, no antes del split)
    train_ds = HMCFormerDataset(train_X, train_y, transform=train_transform, apply_enhancement=True)
    val_ds = HMCFormerDataset(val_X, val_y, transform=val_transform, apply_enhancement=True)
    
    # Crear dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False  # mantener todas las muestras
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    # Crear modelo
    model = HMCFormerLike(
        transformer_name=config.get('transformer_name', "swin_small_patch4_window7_224"),
        cnn_name=config.get('cnn_name', "efficientnet_b0"),
        pretrained=True,
        feat_dim=config.get('feat_dim', 512),
        num_classes=config['num_classes'],
        img_size=CROP_SIZE
    ).to(DEVICE)
    
    # Optimizer y scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config.get('weight_decay', WEIGHT_DECAY)
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )
    
    # Tracking
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = -1
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    # Paths para checkpoints
    fold_save_path = os.path.join(SAVE_GRAPHICS_PATH, f"fold_{fold_id}_seed_{seed}")
    ckpt_path = os.path.join(fold_save_path, "checkpoints")
    ensure_dir(ckpt_path)
    
    # Entrenamiento
    print(f"\nIniciando entrenamiento...")
    for epoch in range(1, config['epochs'] + 1):
        
        # Estrategia de congelamiento (primeras 10 épocas)
        if epoch <= 10:
            # Congelar backbones, entrenar solo fusion y classifier
            for name, param in model.named_parameters():
                if 'fusion' in name or 'classifier' in name or 'pool_proj' in name or 'cnn_proj' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            # Descongelar todo
            for param in model.parameters():
                param.requires_grad = True
        
        # Train y validación
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device=DEVICE)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device=DEVICE)
        scheduler.step(val_loss)
        
        # Guardar historial
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch:3d}/{config['epochs']}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'fold_id': fold_id,
                'seed': seed
            }, os.path.join(ckpt_path, "best_checkpoint.pth"))
            
            print(f"✅ Mejor modelo guardado (epoch {epoch}, val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%)")
    
    # Guardar curvas de entrenamiento
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Loss Curves - Fold {fold_id}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'Accuracy Curves - Fold {fold_id}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    curves_path = os.path.join(fold_save_path, "training_curves.png")
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Curvas de entrenamiento guardadas en: {curves_path}")
    
    # Cargar mejor modelo para evaluación final
    best_ckpt = torch.load(os.path.join(ckpt_path, "best_checkpoint.pth"))
    model.load_state_dict(best_ckpt['model_state_dict'])
    
    # Evaluación final del fold
    print(f"\n{'='*60}")
    print(f"EVALUACIÓN FINAL - FOLD {fold_id}")
    print(f"{'='*60}")
    
    labels, preds, probs = full_evaluation(
        model,
        val_loader,
        class_names=[f"Clase_{i}" for i in range(NUM_CLASSES)],
        device=DEVICE,
        save_path=fold_save_path,
        model_name=f"{MODEL_NAME}_fold{fold_id}_seed{seed}",
        save_csv=True
    )
    
    # revision PeerJ: evaluar el mejor checkpoint del fold en el test independiente
    test_ds = HMCFormerDataset(TEST_X, TEST_Y, transform=val_transform, apply_enhancement=True)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False,
                             num_workers=4, pin_memory=True, drop_last=False)
    test_dir = os.path.join(fold_save_path, 'test')
    ensure_dir(test_dir)
    t_labels, t_preds, t_probs = full_evaluation(
        model, test_loader, class_names=[f'Clase_{i}' for i in range(NUM_CLASSES)],
        device=DEVICE, save_path=test_dir,
        model_name=f'{MODEL_NAME}_fold{fold_id}_seed{seed}_TEST', save_csv=True)
    t_m = compute_clinical_metrics(t_labels, t_preds, average='macro')
    resumen = os.path.join(SAVE_GRAPHICS_PATH, 'test_por_fold.csv')
    nuevo = not os.path.exists(resumen)
    with open(resumen, 'a') as fh:
        if nuevo:
            fh.write('seed,fold,best_epoch,val_acc,test_acc,test_precision,test_recall,test_f1\n')
        fh.write(f"{seed},{fold_id},{best_epoch},{best_val_acc:.2f},{t_m['accuracy']:.4f},"
                 f"{t_m['precision']:.4f},{t_m['recall']:.4f},{t_m['f1']:.4f}\n")
    pd.DataFrame({'y_true': np.asarray(t_labels), 'y_pred': np.asarray(t_preds)}).to_csv(
        os.path.join(test_dir, 'predicciones_test.csv'), index_label='idx')
    print(f"[TEST] fold {fold_id} seed {seed}: acc {t_m['accuracy']:.4f} P {t_m['precision']:.4f} "
          f"R {t_m['recall']:.4f} F1 {t_m['f1']:.4f}", flush=True)

    # Guardar historial como CSV
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(fold_save_path, "history.csv"), index=False)
    
    return {
        'fold_id': fold_id,
        'seed': seed,
        'model': model,
        'history': history,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'val_labels': labels,
        'val_preds': preds,
        'val_probs': probs,
        'checkpoint_path': os.path.join(ckpt_path, "best_checkpoint.pth")
    }

# ----------------------------
# ---- Cross-Validation Principal ----
# ----------------------------
def run_kfold_cv(X, y, n_splits=N_SPLITS, seeds=SEEDS, config=None):
    """
    Ejecuta k-fold cross-validation con múltiples seeds.
    
    ESTÁNDARES CIENTÍFICOS IMPLEMENTADOS:
    1. StratifiedKFold para mantener distribución de clases
    2. Múltiples seeds para robustez estadística
    3. Sin data leakage (splits antes de cualquier procesamiento)
    4. Tracking completo de métricas
    5. Reproducibilidad total
    
    Args:
        X: numpy array (N, H, W, 3) - imágenes originales
        y: numpy array (N,) - etiquetas
        n_splits: número de folds
        seeds: lista de semillas para repeticiones
        config: diccionario de configuración
    
    Returns:
        tuple: (all_results, summary_df)
    """
    if config is None:
        config = {
            'batch_size': BATCH_SIZE,
            'lr': LR,
            'epochs': EPOCHS,
            'num_classes': NUM_CLASSES,
            'transformer_name': 'swin_small_patch4_window7_224',
            'cnn_name': 'efficientnet_b0',
            'feat_dim': 512,
            'weight_decay': WEIGHT_DECAY
        }
    
    print("\n" + "="*80)
    print("INICIANDO CROSS-VALIDATION")
    print("="*80)
    print(f"Datos: X={X.shape}, y={y.shape}")
    print(f"Distribución de clases: {Counter(y)}")
    print(f"N_splits: {n_splits}")
    print(f"Seeds: {seeds}")
    print(f"Device: {DEVICE}")
    print("="*80 + "\n")
    
    all_results = []
    fold_metrics_list = []
    
    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'#'*80}")
        print(f"SEED {seed_idx + 1}/{len(seeds)}: {seed}")
        print(f"{'#'*80}\n")
        
        set_seed(seed)
        
        # Crear StratifiedKFold (CRÍTICO: garantiza estratificación)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        
        for fold_id, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            
            # Entrenar fold
            result = train_fold(fold_id, train_idx, val_idx, X, y, config, seed)
            all_results.append(result)
            
            # Calcular métricas y agregar a lista
            metrics = compute_clinical_metrics(
                result['val_labels'],
                result['val_preds'],
                average="macro"
            )
            
            metrics_flat = {
                'seed': seed,
                'fold': fold_id,
                'best_epoch': result['best_epoch'],
                'best_val_loss': result['best_val_loss'],
                'best_val_acc': result['best_val_acc'],
                'accuracy': metrics['accuracy'],
                'precision_macro': metrics['precision'],
                'recall_macro': metrics['recall'],
                'f1_macro': metrics['f1']
            }
            
            # Agregar métricas por clase
            for cls_id, cls_metrics in metrics['per_class'].items():
                metrics_flat[f'class_{cls_id}_sensitivity'] = cls_metrics['sensitivity']
                metrics_flat[f'class_{cls_id}_specificity'] = cls_metrics['specificity']
            
            fold_metrics_list.append(metrics_flat)
            
            print(f"\nFold {fold_id} completado - Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}\n")
    
    # Crear DataFrame con resultados
    summary_df = pd.DataFrame(fold_metrics_list)
    
    # Guardar resumen completo
    summary_path = os.path.join(SAVE_GRAPHICS_PATH, f"{MODEL_NAME}_cv_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{'='*80}")
    print(f"Resumen de CV guardado en: {summary_path}")
    print(f"{'='*80}\n")
    
    # Calcular estadísticas agregadas
    print("\n" + "="*80)
    print("ESTADÍSTICAS AGREGADAS (TODOS LOS FOLDS Y SEEDS)")
    print("="*80)
    
    metrics_cols = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    for metric in metrics_cols:
        mean_val = summary_df[metric].mean()
        std_val = summary_df[metric].std()
        print(f"{metric:20s}: {mean_val:.4f} ± {std_val:.4f}")
    
    print("="*80 + "\n")
    
    # Guardar estadísticas agregadas
    agg_stats = {
        'model': MODEL_NAME,
        'n_splits': n_splits,
        'n_seeds': len(seeds),
        'total_folds': len(summary_df),
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    for metric in metrics_cols:
        agg_stats[f'{metric}_mean'] = summary_df[metric].mean()
        agg_stats[f'{metric}_std'] = summary_df[metric].std()
        agg_stats[f'{metric}_min'] = summary_df[metric].min()
        agg_stats[f'{metric}_max'] = summary_df[metric].max()
    
    agg_df = pd.DataFrame([agg_stats])
    agg_path = os.path.join(SAVE_GRAPHICS_PATH, f"{MODEL_NAME}_aggregated_stats.csv")
    agg_df.to_csv(agg_path, index=False)
    print(f"Estadísticas agregadas guardadas en: {agg_path}\n")
    
    # Crear visualización de resultados por fold
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, metric in enumerate(metrics_cols):
        ax = axes[idx // 2, idx % 2]
        
        for seed in seeds:
            seed_data = summary_df[summary_df['seed'] == seed]
            ax.plot(seed_data['fold'], seed_data[metric], 
                   marker='o', label=f'Seed {seed}', alpha=0.7)
        
        ax.axhline(y=summary_df[metric].mean(), color='r', 
                  linestyle='--', label=f'Mean: {summary_df[metric].mean():.4f}')
        ax.set_xlabel('Fold')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} por Fold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    viz_path = os.path.join(SAVE_GRAPHICS_PATH, f"{MODEL_NAME}_cv_results.png")
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualización de resultados guardada en: {viz_path}\n")
    
    return all_results, summary_df

# ----------------------------
# ---- Carga de datos ----
# ----------------------------
def load_npy_data(X_train_path, X_test_path, y_train_path, y_test_path):
    """
    Carga datos desde archivos .npy.
    
    IMPORTANTE: Concatena train y test para hacer CV sobre todo el dataset.
    Esto es correcto si el test set original fue un holdout separado previamente.
    Si prefieres mantener test separado, no concatenes.
    """
    print("Cargando datos...")
    X_train = np.load(X_train_path)
    X_test = np.load(X_test_path)
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)
    
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Concatenar para CV (modificar si quieres mantener test separado)
    # revision PeerJ: la validacion cruzada usa solo el train; el test queda aparte
    global TEST_X, TEST_Y
    TEST_X, TEST_Y = X_test, y_test
    X_all, y_all = X_train, y_train
    
    print(f"Dataset completo: X={X_all.shape}, y={y_all.shape}")
    print(f"Distribución de clases: {Counter(y_all)}\n")
    
    return X_all, y_all

# ----------------------------
# ---- Main ----
# ----------------------------
if __name__ == "__main__":
    print("\n" + "="*80)
    print("HMCFORMER - CROSS VALIDATION PIPELINE")
    print("="*80 + "\n")
    
    # Cargar datos
    X_all, y_all = load_npy_data(X_TRAIN_NPY, X_TEST_NPY, Y_TRAIN_NPY, Y_TEST_NPY)
    
    # Configuración
    config = {
        'batch_size': BATCH_SIZE,
        'lr': LR,
        'epochs': EPOCHS,
        'num_classes': NUM_CLASSES,
        'transformer_name': "swin_small_patch4_window7_224",
        'cnn_name': "efficientnet_b0",
        'feat_dim': 512,
        'weight_decay': WEIGHT_DECAY
    }
    
    print(f"Configuración:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()
    
    # Ejecutar CV
    all_results, summary_df = run_kfold_cv(
        X_all, y_all,
        n_splits=N_SPLITS,
        seeds=SEEDS,
        config=config
    )
    
    print("\n" + "="*80)
    print("CROSS-VALIDATION COMPLETADO EXITOSAMENTE")
    print("="*80)
    print(f"\nTotal de folds entrenados: {len(all_results)}")
    print(f"Resultados guardados en: {SAVE_GRAPHICS_PATH}")
    print("\nArchivos generados:")
    print(f"  - {MODEL_NAME}_cv_summary.csv (métricas por fold)")
    print(f"  - {MODEL_NAME}_aggregated_stats.csv (estadísticas agregadas)")
    print(f"  - {MODEL_NAME}_cv_results.png (visualización)")
    print(f"  - Carpetas individuales por fold con checkpoints y gráficas")
    print("\n" + "="*80 + "\n")
