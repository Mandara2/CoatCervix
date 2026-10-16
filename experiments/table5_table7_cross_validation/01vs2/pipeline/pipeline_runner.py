# pipeline_runner.py
import numpy as np
import os

from config import settings
from data.loaders import load_data
from data.preprocessing import preprocess_all_datasets
from data.label_mapper import parse_label_mode, filter_and_map_labels
from training.cross_validation import run_cross_validation


def run_pipeline():
    """
    Ejecuta el pipeline completo de cross-validation para clasificación binaria.
    
    Flujo:
    1. Carga datos (X_train_full, X_test) - SIN split train/val
    2. Filtra clases según LABEL_MODE (3 clases → 2 clases)
    3. Preprocesa datos (resize + CLAHE + normalización)
    4. Ejecuta cross-validation de 5 folds (StratifiedKFold crea los splits)
    """
    print("="*70)
    print("🚀 PIPELINE DE CROSS-VALIDATION")
    print("="*70 + "\n")

    # ================================
    # 1️⃣ CARGA DE DATOS
    # ================================
    print("[1/4] Cargando datos...")
    X_train_full, X_test, y_train_full, y_test = load_data()
    
    print(f"  ✅ Datos cargados:")
    print(f"     - Train (completo): {X_train_full.shape[0]} muestras")
    print(f"     - Test: {X_test.shape[0]} muestras (reservado para evaluación final)")
    print(f"     - Distribución Train: {dict(zip(*np.unique(y_train_full, return_counts=True)))}\n")

    # ================================
    # 2️⃣ FILTRAR CLASES
    # ================================
    mode = parse_label_mode(settings.LABEL_MODE)
    print(f"[2/4] Filtrando clases según modo: {mode}")
    
    # Filtrar train
    X_filtered, y_binary, class_names, num_classes = filter_and_map_labels(
        mode=mode,
        X=X_train_full,
        y=y_train_full
    )
    
    # Nota: X_test se deja intacto por ahora, CV solo usa train/val
    
    print(f"  ✅ Clases filtradas:")
    print(f"     - Train: {X_filtered.shape[0]} muestras")
    print(f"     - Num clases: {num_classes}")
    print(f"     - Nombres: {class_names}")
    print(f"     - Distribución Train: {dict(zip(*np.unique(y_binary, return_counts=True)))}\n")

    # ================================
    # 3️⃣ CROSS-VALIDATION
    # ================================
    print("[3/4] Ejecutando cross-validation...")
    print("  ⚠️  Preprocesamiento (resize + CLAHE + normalización) se hará dentro de cada fold\n")
    
    results = run_cross_validation(
        X_full=X_filtered,  # ← Train sin preprocesar
        y_full=y_binary,
        class_names=class_names,
        n_folds=5,
        seed=settings.SEED
    )

    # ================================
    # 📊 RESUMEN FINAL
    # ================================
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETADO")
    print("="*70)
    print(f"\n📊 Resultados finales ({mode}):")
    print(f"   Accuracy:  {results['mean_acc']:.4f} ± {results['std_acc']:.4f}")
    print(f"   Precision: {results['mean_prec']:.4f} ± {results['std_prec']:.4f}")
    print(f"   Recall:    {results['mean_rec']:.4f} ± {results['std_rec']:.4f}")
    print(f"   F1-Score:  {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
    
    if results['mean_auc'] is not None:
        print(f"   AUC-ROC:   {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
    
    print(f"\n📁 Resultados guardados en:")
    print(f"   {os.path.join(settings.RESULTS_PATH, 'cross_validation', mode)}\n")

if __name__ == "__main__":
    run_pipeline()

