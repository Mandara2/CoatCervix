# pipeline_runner.py
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
import os

from config import settings
from data.loaders import load_data
from training.train import train_model
from training.evaluate import evaluate_saved_model
from data.preprocessing import preprocess_all_datasets
from data.dataloaders import create_dataloaders


def run_pipeline():
    """
    Ejecuta el pipeline completo de entrenamiento y evaluación del modelo.
    """
    print("🚀 Iniciando pipeline...")

    # ================================
    # 1️⃣ CARGA DE DATOS
    # ================================
    print("📥 Cargando datos...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    print(f"✅ Datos cargados: {X_train.shape[0]} train | {X_test.shape[0]} test | {X_val.shape[0]} val")

    # ================================
    # 2️⃣ PREPROCESAMIENTO
    # ================================
    print("🧪 Redimensionando y preprocesando imágenes...")

    X_train, X_val, X_test, mean, std = preprocess_all_datasets(
        X_train, X_val, X_test, target_size=settings.target_size
    )

    # ================================
    # 3️⃣ CREAR DATALOADERS
    # ================================
    train_loader, val_loader, test_loader = create_dataloaders(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    batch_size=settings.batch_size,
    seed=settings.SEED
)

    print("🏋️ Entrenando modelo...")
    results = train_model(train_loader, val_loader, seed=settings.SEED)
    model = results["model"]
    print(f"✅ Entrenamiento finalizado | "
      f"Mejor F1: {results['best_val_f1']:.4f} | "
      f"Mejor Accuracy: {results['best_val_acc']:.2f}% | "
      f"Última Train Loss: {results['history_train_loss'][-1]:.4f} | "
      f"Última Val Loss: {results['history_val_loss'][-1]:.4f}")


    print("📊 Evaluando modelo...")
    
    weights_path = os.path.join(settings.SAVE_GRAPHICS_PATH, f"{settings.MODEL_NAME}.pth")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_output_dir = os.path.join(settings.SAVE_GRAPHICS_PATH, f"{settings.MODEL_NAME}_{timestamp}")
    os.makedirs(model_output_dir, exist_ok=True)

    evaluate_saved_model(weights_path, test_loader, mean, std)
    print(f"✅ Evaluación completada. Resultados guardados en: {model_output_dir}")

    print("🎯 Pipeline completado exitosamente.")

if __name__ == "__main__":
    run_pipeline()
