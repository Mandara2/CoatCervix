import numpy as np
import pandas as pd
from pipeline.pipeline_runner import run_pipeline
from config import settings
import os

def run_seed_experiment():
    seeds = [42, 123, 2024, 7, 99]  # Semillas para probar veracidad y robustez
    results_list = []

    print(f"🔬 Iniciando experimento con {len(seeds)} semillas: {seeds}")

    for seed in seeds:
        print(f"\n{'='*40}")
        print(f"🔄 Ejecutando con semilla: {seed}")
        print(f"{'='*40}")
        
        try:
            # Ejecutar pipeline y obtener métricas
            metrics = run_pipeline(seed=seed)
            
            # Guardar resultados
            metrics["seed"] = seed
            results_list.append(metrics)
            
        except Exception as e:
            print(f"⚠️ Error ejecuntando semilla {seed}: {e}")

    # Convertir a DataFrame
    df = pd.DataFrame(results_list)

    if df.empty:
        print("❌ No se obtuvieron resultados.")
        return

    # Definir métricas de interés
    metrics_to_report = ["acc", "f1", "rec", "prec"]
    metrics_names = {
        "acc": "Accuracy",
        "f1": "F1 Score",
        "rec": "Recall",
        "prec": "Precision"
    }

    print("\n" + "="*50)
    print("📊 RESULTADOS FINALES MEJORADOS (Mean ± SD)")
    print("="*50)

    summary_stats = {}

    for metric in metrics_to_report:
        if metric in df.columns:
            # Escalar a porcentaje si es Accuracy o parece ser 0-1
            is_percentage = metric == "acc" or df[metric].mean() <= 1.0
            
            values = df[metric].values
            if is_percentage and metric == "acc": # Acc usually comes as 0-100 or 0-1, let's normalize to percentage representation if needed
                 # En evaluate.py acc se multiplica por 100? Revisemos evaluate.py output
                 # evaluate.py: acc = accuracy_score(...) -> 0.0 to 1.0
                 # BUT train.py multiplies by 100.
                 # Let's check evaluate.py again.
                 # evaluate.py line 41: acc = accuracy_score(...) -> Returns roughly 0.0 to 1.0
                 pass

            # Convert to percentage for display if it is 0-1 range
            # Accuracy from scikit-learn is 0-1.
            # Let's assume all are 0-1 except maybe if explicitly handled otherwise.
            
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            # Display as percentage for consistency if desired, or kept as decimal.
            # Usually strict academic papers use decimal or percentage. 
            # I will use percentage for readability as it is common.
            
            mean_pct = mean_val * 100
            std_pct = std_val * 100
            
            print(f"🔹 {metrics_names[metric]}: {mean_pct:.2f} ± {std_pct:.2f}")
            
            summary_stats[metric] = f"{mean_pct:.2f} ± {std_pct:.2f}"

    # Guardar resumen en CSV
    save_path = os.path.join(settings.RESULTS_PATH, "seed_experiment_summary.csv")
    os.makedirs(settings.RESULTS_PATH, exist_ok=True)
    
    df.to_csv(os.path.join(settings.RESULTS_PATH, "seed_experiment_details.csv"), index=False)
    
    # Save summary text
    with open(save_path, "w") as f:
        f.write("Metric,Mean +/- SD\n")
        for metric, stats in summary_stats.items():
            f.write(f"{metrics_names[metric]},{stats}\n")

    print(f"\n💾 Resultados detallados y resumen guardados en: {settings.RESULTS_PATH}")
    print("="*50)

if __name__ == "__main__":
    run_seed_experiment()

