import cv2
import numpy as np


def resize_images(images, size):
    """
    Redimensiona una lista o array de imágenes a un tamaño específico.
    Asume que todas las imágenes ya tienen 3 canales (RGB).
    """
    resized = [cv2.resize(img, size, interpolation=cv2.INTER_LINEAR) for img in images]
    return np.array(resized, dtype=np.float32)


def apply_clahe(img_array):

    """
    Aplica el filtro CLAHE a una imagen RGB (en rango 0–1 o 0–255).
    Retorna imagen en rango [0,1].
    """

    img_255 = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img_255, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    return img_clahe.astype(np.float32) / 255.0

def preprocess_all_datasets(X_train, X_val, X_test, target_size):
    """
    Redimensiona, aplica CLAHE y normaliza X_train, X_val y X_test en un solo paso.
    La normalización se hace usando mean y std calculados solo sobre X_train filtrado.

    Parámetros
    ----------
    X_train, X_val, X_test : np.ndarray
        Conjuntos de imágenes originales (N, H, W, C)
    target_size : tuple[int, int]
        Tamaño deseado para redimensionar las imágenes

    Retorna
    -------
    tuple : (X_train_norm, X_val_norm, X_test_norm)
        Conjuntos normalizados listos para DataLoader
    """

    # --- Redimensionar todos los conjuntos ---
    X_train_resized = resize_images(X_train, target_size)
    X_val_resized   = resize_images(X_val, target_size)
    X_test_resized  = resize_images(X_test, target_size)

    # --- Calcular mean y std solo de X_train filtrado ---
    mean = X_train_resized.mean(axis=(0,1,2))
    std  = X_train_resized.std(axis=(0,1,2))

    # --- Aplicar filtro CLAHE ---
    X_train_filt = np.array([apply_clahe(img) for img in X_train_resized])
    X_val_filt   = np.array([apply_clahe(img) for img in X_val_resized])
    X_test_filt  = np.array([apply_clahe(img) for img in X_test_resized])

    
    # --- Normalizar todos los conjuntos ---
    X_train_norm = (X_train_filt - mean) / (std + 1e-8)
    X_val_norm   = (X_val_filt - mean) / (std + 1e-8)
    X_test_norm  = (X_test_filt  - mean) / (std + 1e-8)

    return X_train_norm, X_val_norm, X_test_norm, mean, std