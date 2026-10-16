# data/label_mapper.py
import numpy as np
import re

def parse_label_mode(mode: str):
    """
    Normaliza el modo de entrada.
    Soporta:
        - "0vs1", "1vs2", "0vs12", "01vs2", "012vs2"
        - "multiclass"
    """
    if mode is None:
        return "multiclass"

    s = str(mode).lower().strip().replace(" ", "")
    s = s.replace("+", "")
    s = s.replace("(", "").replace(")", "")

    if "vs" not in s and "v" in s:
        s = s.replace("v", "vs")

    if s != "multiclass" and not re.fullmatch(r"[0-9]+vs[0-9]+", s):
        raise ValueError(
            f"Modo inválido: {mode}. "
            f"Ejemplos válidos: '0vs1','1vs2','0vs12','01vs2','multiclass'"
        )

    return s


def build_label_map(mode: str, y: np.ndarray):
    """
    Construye el mapeo para modos como:
      - "0vs1"
      - "1vs2"
      - "0vs12"        (0 vs 1+2)
      - "01vs2"        (0+1 vs 2)
      - "multiclass"
    """

    mode = parse_label_mode(mode)
    y = np.asarray(y)

    print(f"\n🔎 [LabelMapper] Modo solicitado: {mode}")

    # --- MULTICLASS ---
    if mode == "multiclass":
        print("🟦 Modo multiclass — no se filtran clases.")
        print(f"📊 Total muestras: {len(y)}")
        unique, counts = np.unique(y, return_counts=True)
        print(f"📌 Distribución original: {dict(zip(unique, counts))}")

        return (
            np.arange(len(y)),
            y.copy(),
            ["Class 0","Class 1","Class 2"],
            3
        )

    # --- BINARY MODES (XvsY) ---
    left, right = mode.split("vs")

    def parse_set(s):
        return set(int(c) for c in s) if s else set()

    left_set  = parse_set(left)
    right_set = parse_set(right)

    if not left_set or not right_set:
        raise ValueError("Los conjuntos no pueden estar vacíos.")

    print(f"🟢 Grupo LEFT → clase 0: {sorted(left_set)}")
    print(f"🔴 Grupo RIGHT → clase 1: {sorted(right_set)}")

    allowed = left_set.union(right_set)
    mask = np.isin(y, list(allowed))

    print(f"📊 Total muestras antes del filtrado: {len(y)}")
    print(f"📊 Total muestras permitidas ({sorted(allowed)}): {mask.sum()}")

    if mask.sum() == 0:
        raise ValueError(f"No hay muestras disponibles para el modo {mode}")

    y_sub = y[mask].copy()
    y_new = np.zeros_like(y_sub)

    # Map RIGHT → 1
    for r in right_set:
        y_new[y_sub == r] = 1

    # Map LEFT → 0
    for l in left_set:
        y_new[y_sub == l] = 0

    # Print mapping summary
    print("🔄 Mapeo realizado:")
    for original_label in sorted(allowed):
        new_label = 0 if original_label in left_set else 1
        print(f"   - {original_label} → {new_label}")

    # Print distribution
    unique_new, counts_new = np.unique(y_new, return_counts=True)
    print(f"📌 Distribución remapeada: {dict(zip(unique_new, counts_new))}")

    # Class names
    left_name  = "+".join(str(k) for k in sorted(left_set))
    right_name = "+".join(str(k) for k in sorted(right_set))
    class_names = [f"Class {left_name}", f"Class {right_name}"]

    return np.where(mask)[0], y_new, class_names, 2



def filter_and_map_labels(mode, X, y):
    """
    Aplica build_label_map y filtra X,y.
    Con prints para confirmar resultado.
    """
    indices, y_new, class_names, num_classes = build_label_map(mode, y)

    print(f"📁 X original: {X.shape[0]} muestras")
    print(f"📁 X filtrado: {len(indices)} muestras")
    print(f"🏷️ Clases finales: {class_names}\n")

    return X[indices], y_new, class_names, num_classes
