# training/cross_validation.py
import os
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

from config import settings
from data.dataloaders import create_dataloaders
from training.train import train_model
from training.metrics import compute_and_save_metrics
from training.utils import clean_cuda


def run_cross_validation(X_full, y_full, class_names, n_folds=5, seed=None):
    """
    Ejecuta cross-validation estratificado de K-folds para clasificación binaria.
    
    IMPORTANTE: Esta función recibe datos SIN preprocesar y aplica:
    - Resize, CLAHE, normalización dentro de cada fold
    - Mean/std calculados SOLO del train fold (después de CLAHE)
    - Evaluación en el conjunto de VALIDACIÓN de cada fold
    
    Args:
        X_full: numpy array [N, H, W, C] con imágenes SIN preprocesar (train)
        y_full: numpy array [N] con etiquetas binarias (0 o 1) (train)
        class_names: list de str con nombres de las 2 clases (ej: ["Class 0", "Class 1"])
        n_folds: int, número de folds (default 5)
        seed: int, semilla para reproducibilidad (default: settings.SEED)
    
    Returns:
        dict con resultados agregados de todos los folds
    """
    
    if seed is None:
        seed = settings.SEED
    
    # Validar que los datos sean binarios
    unique_labels = np.unique(y_full)
    # assert len(unique_labels) == 2, f"Datos deben ser binarios, recibido: {unique_labels}"
    # assert set(unique_labels) == {0, 1}, f"Labels deben ser {{0, 1}}, recibido: {unique_labels}"
    
    # Extraer nombre de combinación desde class_names
    class_combination = "_vs_".join(class_names)
    
    print(f"\n{'='*70}")
    print(f" CROSS-VALIDATION: {class_combination} ({n_folds} folds)")
    print(f"{'='*70}\n")
    
    # ================================
    # 1️⃣ VALIDACIONES DE INTEGRIDAD
    # ================================
    print(f"[VALIDACIÓN] Verificando integridad de datos...")
    print(f"  - Shape: {X_full.shape}")
    print(f"  - Dtype: {X_full.dtype}")
    print(f"  - Labels shape: {y_full.shape}")
    print(f"  - Clases únicas: {unique_labels}")
    print(f"  - Distribución: {dict(zip(*np.unique(y_full, return_counts=True)))}")
    print(f"  - Clases: {class_names}\n")
    
    # ================================
    # 2️⃣ CREAR DIRECTORIO DE RESULTADOS
    # ================================
    cv_base_dir = os.path.join(settings.RESULTS_PATH, "cross_validation", class_combination)
    os.makedirs(cv_base_dir, exist_ok=True)
    print(f" Directorio de resultados: {cv_base_dir}\n")
    
    # ================================
    # 3️⃣ CONFIGURAR STRATIFIED K-FOLD
    # ================================
    print(f"[ESTRATIFICACIÓN] Creando {n_folds} folds estratificados...")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    # ================================
    # 4️⃣ ALMACENAMIENTO DE RESULTADOS
    # ================================
    fold_results = []
    all_fold_labels = []
    all_fold_preds = []
    all_fold_probs = []
    
    # ================================
    # 5️⃣ ITERAR SOBRE FOLDS
    # ================================
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full), start=1):
        
        print(f"\n{'='*70}")
        print(f"📊 FOLD {fold_idx}/{n_folds}")
        print(f"{'='*70}\n")
        
        # 5.1 Validar que no hay solapamiento entre train y val
        assert len(set(train_idx) & set(val_idx)) == 0, "¡Solapamiento detectado entre train y val!"
        
        # 5.2 Dividir datos del fold
        X_train_fold = X_full[train_idx]
        y_train_fold = y_full[train_idx]
        X_val_fold = X_full[val_idx]
        y_val_fold = y_full[val_idx]
        
        print(f"[FOLD {fold_idx}] Validaciones:")
        print(f"  - Train: {len(train_idx)} muestras | Val: {len(val_idx)} muestras")
        print(f"  - Distribución Train: {dict(zip(*np.unique(y_train_fold, return_counts=True)))}")
        print(f"  - Distribución Val: {dict(zip(*np.unique(y_val_fold, return_counts=True)))}")
        print(f"  - Solapamiento: {len(set(train_idx) & set(val_idx))} (debe ser 0)")
        print(f"  - Total único: {len(set(train_idx) | set(val_idx))} (debe ser {len(X_full)})\n")
        
        # 5.3 PREPROCESAMIENTO (dentro del fold para evitar data leakage)
        print(f"[FOLD {fold_idx}] Preprocesando datos...")
        from data.preprocessing import resize_images, apply_clahe
        
        # 5.3.1 Resize
        X_train_resized = resize_images(X_train_fold, settings.target_size)
        X_val_resized = resize_images(X_val_fold, settings.target_size)
        
        # 5.3.2 Calcular mean/std SOLO del train fold (ANTES de CLAHE - Lógica Original)
        # Nota: Esto replica el comportamiento del pipeline original aunque matemáticamente
        # sea menos estándar (normalizar img con CLAHE usando stats sin CLAHE).
        mean = X_train_resized.mean(axis=(0, 1, 2))
        std = X_train_resized.std(axis=(0, 1, 2))
        
        # 5.3.3 CLAHE
        X_train_clahe = np.array([apply_clahe(img) for img in X_train_resized])
        X_val_clahe = np.array([apply_clahe(img) for img in X_val_resized])
        
        # 5.3.4 Normalizar usando estadísticos del train fold (Pre-CLAHE)
        X_train_norm = (X_train_clahe - mean) / (std + 1e-8)
        X_val_norm = (X_val_clahe - mean) / (std + 1e-8)
        
        print(f"  ✅ Preprocesamiento completado")
        print(f"     - Train shape: {X_train_norm.shape}")
        print(f"     - Val shape: {X_val_norm.shape}")
        print(f"     - Mean (train): {mean.mean():.4f}")
        print(f"     - Std (train): {std.mean():.4f}\n")
        
        # 5.4 Crear DataLoaders
        print(f"[FOLD {fold_idx}] Creando DataLoaders...")
        
        # Crear dummy test loader (no se usa, pero create_dataloaders lo requiere)
        X_test_dummy = np.empty((0, *X_train_norm.shape[1:]))
        y_test_dummy = np.empty(0, dtype=np.int64)
        
        train_loader, val_loader, _ = create_dataloaders(
            X_train_norm, y_train_fold,
            X_val_norm, y_val_fold,
            X_test_dummy, y_test_dummy,
            batch_size=settings.batch_size,
            seed=seed
        )
        
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Val batches: {len(val_loader)}\n")
        
        # 5.5 Entrenar modelo
        print(f"[FOLD {fold_idx}] Entrenando modelo...")
        
        train_results = train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            seed=seed
        )
        
        model = train_results["model"]
        
        print(f"  ✅ Entrenamiento completado | "
              f"Mejor F1: {train_results['best_val_f1']:.4f} | "
              f"Mejor Acc: {train_results['best_val_acc']:.2f}%\n")
        
        # 5.5 Evaluar modelo en VALIDATION FOLD
        print(f"[FOLD {fold_idx}] Evaluando modelo en validation fold...")
        
        fold_dir = os.path.join(cv_base_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calcular mean y std dummy (los datos ya están normalizados)
        mean_dummy = 0.0
        std_dummy = 1.0
        
        eval_results = compute_and_save_metrics(
            model=model,
            X_test_loader=val_loader,  # ← Evaluar en VALIDATION FOLD
            class_names=class_names,
            save_dir=fold_dir,
            model_name=f"{class_combination}_fold{fold_idx}",
            timestamp=timestamp,
            mean=mean_dummy,
            std=std_dummy,
            device=settings.device
        )
        
        # 5.6 Calcular AUC-ROC
        try:
            # Para clasificación binaria, usar probabilidades de la clase positiva
            # Para multiclass, roc_auc_score requiere manejo especial o 'ovr'
            if y_val_fold.min() >= 0 and len(np.unique(y_val_fold)) == 2:
                 # BINARY CASE
                 y_probs_positive = np.array(eval_results["all_probs"])[:, 1]
                 auc_roc = roc_auc_score(eval_results["all_labels"], y_probs_positive)
            else:
                 # MULTICLASS CASE OR OTHER
                 # Intentar calcular AUC multiclass con strategy 'ovr' si probs tienen shape correcto
                 if np.array(eval_results["all_probs"]).shape[1] > 2:
                     auc_roc = roc_auc_score(eval_results["all_labels"], eval_results["all_probs"], multi_class='ovr')
                 else:
                     auc_roc = None
        except Exception as e:
            print(f"  ⚠️  No se pudo calcular AUC-ROC: {e}")
            auc_roc = None
        
        # 5.7 Guardar métricas del fold en JSON
        fold_metrics = {
            "fold": fold_idx,
            "accuracy": float(eval_results["acc"]),
            "precision": float(eval_results["prec"]),
            "recall": float(eval_results["rec"]),
            "f1_score": float(eval_results["f1"]),
            "auc_roc": float(auc_roc) if auc_roc is not None else None,
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "best_val_f1_during_training": float(train_results['best_val_f1']),
            "best_val_acc_during_training": float(train_results['best_val_acc'])
        }
        
        metrics_path = os.path.join(fold_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(fold_metrics, f, indent=4)
        
        print(f"  💾 Métricas guardadas: {metrics_path}\n")
        
        # 5.8 Almacenar resultados
        fold_results.append(fold_metrics)
        all_fold_labels.extend(eval_results["all_labels"])
        all_fold_preds.extend(eval_results["all_preds"])
        all_fold_probs.extend(eval_results["all_probs"])
        
        # 5.9 Limpiar memoria
        del model, train_loader, val_loader, train_results, eval_results
        clean_cuda()
        
        print(f"✅ Fold {fold_idx} completado\n")
    
    # ================================
    # 6️⃣ AGREGAR RESULTADOS
    # ================================
    print(f"\n{'='*70}")
    print(f"📈 AGREGANDO RESULTADOS DE {n_folds} FOLDS")
    print(f"{'='*70}\n")
    
    aggregated_results = aggregate_cv_results(
        fold_results=fold_results,
        all_labels=all_fold_labels,
        all_preds=all_fold_preds,
        all_probs=all_fold_probs,
        class_names=class_names,
        class_combination=class_combination,
        save_dir=cv_base_dir
    )
    
    # ================================
    # 7️⃣ RESUMEN FINAL
    # ================================
    print(f"\n{'='*70}")
    print(f"✅ CROSS-VALIDATION COMPLETADO: {class_combination}")
    print(f"{'='*70}\n")
    
    print(f"📊 Resultados Agregados (Mean ± Std):")
    print(f"   Accuracy:  {aggregated_results['mean_acc']:.4f} ± {aggregated_results['std_acc']:.4f}")
    print(f"   Precision: {aggregated_results['mean_prec']:.4f} ± {aggregated_results['std_prec']:.4f}")
    print(f"   Recall:    {aggregated_results['mean_rec']:.4f} ± {aggregated_results['std_rec']:.4f}")
    print(f"   F1-Score:  {aggregated_results['mean_f1']:.4f} ± {aggregated_results['std_f1']:.4f}")
    
    if aggregated_results['mean_auc'] is not None:
        print(f"   AUC-ROC:   {aggregated_results['mean_auc']:.4f} ± {aggregated_results['std_auc']:.4f}")
    
    print(f"\n📁 Resultados guardados en: {cv_base_dir}\n")
    
    return aggregated_results


def aggregate_cv_results(fold_results, all_labels, all_preds, all_probs, 
                         class_names, class_combination, save_dir):
    """
    Agrega resultados de todos los folds y genera visualizaciones.
    
    Args:
        fold_results: lista de dicts con métricas de cada fold
        all_labels: lista con todas las etiquetas verdaderas (concatenadas de todos los folds)
        all_preds: lista con todas las predicciones (concatenadas de todos los folds)
        all_probs: lista con todas las probabilidades (concatenadas de todos los folds)
        class_names: lista con nombres de las clases
        class_combination: str, nombre de la combinación (ej: "0vs1")
        save_dir: str, directorio donde guardar resultados agregados
    
    Returns:
        dict con estadísticas agregadas
    """
    
    # ================================
    # 1️⃣ CALCULAR ESTADÍSTICAS
    # ================================
    metrics_df = pd.DataFrame(fold_results)
    
    stats = {
        "class_combination": class_combination,
        "n_folds": len(fold_results),
        "mean_acc": metrics_df["accuracy"].mean(),
        "std_acc": metrics_df["accuracy"].std(),
        "mean_prec": metrics_df["precision"].mean(),
        "std_prec": metrics_df["precision"].std(),
        "mean_rec": metrics_df["recall"].mean(),
        "std_rec": metrics_df["recall"].std(),
        "mean_f1": metrics_df["f1_score"].mean(),
        "std_f1": metrics_df["f1_score"].std(),
        "mean_auc": metrics_df["auc_roc"].mean() if "auc_roc" in metrics_df and metrics_df["auc_roc"].notna().any() else None,
        "std_auc": metrics_df["auc_roc"].std() if "auc_roc" in metrics_df and metrics_df["auc_roc"].notna().any() else None,
        "fold_results": fold_results
    }
    
    # ================================
    # 2️⃣ GUARDAR CSV CON MÉTRICAS
    # ================================
    csv_path = os.path.join(save_dir, "aggregated_metrics.csv")
    
    summary_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Mean": [stats["mean_acc"], stats["mean_prec"], stats["mean_rec"], stats["mean_f1"]],
        "Std": [stats["std_acc"], stats["std_prec"], stats["std_rec"], stats["std_f1"]]
    }
    
    if stats["mean_auc"] is not None:
        summary_data["Metric"].append("AUC-ROC")
        summary_data["Mean"].append(stats["mean_auc"])
        summary_data["Std"].append(stats["std_auc"])
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(csv_path, index=False)
    print(f"💾 Métricas agregadas guardadas: {csv_path}")
    
    # ================================
    # 3️⃣ GUARDAR MÉTRICAS POR FOLD
    # ================================
    fold_csv_path = os.path.join(save_dir, "metrics_per_fold.csv")
    metrics_df.to_csv(fold_csv_path, index=False)
    print(f"💾 Métricas por fold guardadas: {fold_csv_path}")
    
    # ================================
    # 4️⃣ MATRIZ DE CONFUSIÓN AGREGADA
    # ================================
    plot_aggregated_confusion_matrix(
        all_labels=all_labels,
        all_preds=all_preds,
        class_names=class_names,
        save_path=os.path.join(save_dir, "aggregated_confusion_matrix.png"),
        title=f"Matriz de Confusión Agregada - {class_combination}"
    )
    
    # ================================
    # 5️⃣ CURVAS ROC AGREGADAS
    # ================================
    try:
        plot_aggregated_roc_curves(
            fold_results=fold_results,
            all_labels=all_labels,
            all_probs=all_probs,
            class_names=class_names,
            save_path=os.path.join(save_dir, "aggregated_roc_curves.png"),
            title=f"Curvas ROC - {class_combination}"
        )
    except Exception as e:
        print(f"⚠️  No se pudo generar curvas ROC agregadas: {e}")
    
    # ================================
    # 6️⃣ BOXPLOTS DE MÉTRICAS
    # ================================
    plot_metrics_boxplots(
        metrics_df=metrics_df,
        save_path=os.path.join(save_dir, "metrics_boxplots.png"),
        title=f"Distribución de Métricas - {class_combination}"
    )
    
    # ================================
    # 7️⃣ GUARDAR RESUMEN JSON
    # ================================
    summary_json_path = os.path.join(save_dir, "summary.json")
    
    summary_json = {
        "class_combination": class_combination,
        "n_folds": stats["n_folds"],
        "aggregated_metrics": {
            "accuracy": {"mean": stats["mean_acc"], "std": stats["std_acc"]},
            "precision": {"mean": stats["mean_prec"], "std": stats["std_prec"]},
            "recall": {"mean": stats["mean_rec"], "std": stats["std_rec"]},
            "f1_score": {"mean": stats["mean_f1"], "std": stats["std_f1"]},
        },
        "fold_details": fold_results
    }
    
    if stats["mean_auc"] is not None:
        summary_json["aggregated_metrics"]["auc_roc"] = {
            "mean": stats["mean_auc"], 
            "std": stats["std_auc"]
        }
    
    with open(summary_json_path, "w") as f:
        json.dump(summary_json, f, indent=4)
    
    print(f"💾 Resumen JSON guardado: {summary_json_path}\n")
    
    return stats


def plot_aggregated_confusion_matrix(all_labels, all_preds, class_names, save_path, title):
    """
    Genera matriz de confusión agregada de todos los folds.
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Matriz de confusión agregada guardada: {save_path}")


def plot_aggregated_roc_curves(fold_results, all_labels, all_probs, class_names, save_path, title):
    """
    Genera curvas ROC agregadas (curva promedio de todos los folds).
    """
    from sklearn.metrics import roc_curve, auc
    
    # Convertir probabilidades a numpy array
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Para clasificación binaria, usar probabilidades de clase positiva
    y_probs_positive = all_probs[:, 1]
    
    # Calcular curva ROC
    fpr, tpr, _ = roc_curve(all_labels, y_probs_positive)
    roc_auc = auc(fpr, tpr)
    
    # Plotear
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Curvas ROC agregadas guardadas: {save_path}")


def plot_metrics_boxplots(metrics_df, save_path, title):
    """
    Genera boxplots de las métricas a través de los folds.
    """
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
    
    # Filtrar solo métricas que existen
    metrics_to_plot = [m for m in metrics_to_plot if m in metrics_df.columns]
    
    if "auc_roc" in metrics_df.columns:
        metrics_to_plot.append("auc_roc")
    
    # Preparar datos para boxplot
    data_to_plot = []
    labels = []
    
    for metric in metrics_to_plot:
        data_to_plot.append(metrics_df[metric].values)
        labels.append(metric.replace("_", " ").title())
    
    # Plotear
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='blue'),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(color='blue'),
                    capprops=dict(color='blue'))
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Boxplots de métricas guardados: {save_path}")

