# data/label_mapper.py
import numpy as np

def parse_label_mode(mode: str):
    """
    Normaliza el modo de entrada. Ejemplos válidos:
      "0vs1", "1vs2", "0vs2", "0vs12", "0v12", "0v(1+2)", "multiclass"
    Devuelve una representación estándar (lower).
    """
    if mode is None:
        return "multiclass"
    s = str(mode).lower().replace(" ", "")
    s = s.replace("v", "vs")  # aceptar "0v12"
    s = s.replace("+", "")
    s = s.replace("(", "").replace(")", "")
    if s in ("multiclass", ""):
        return "multiclass"
    return s

def build_label_map(mode: str, y: np.ndarray):
    """
    Recibe:
      - mode: string como "0vs1", "1vs2", "0vs12" (0 vs (1,2)), o "multiclass".
      - y: array original de etiquetas (num_samples,)
    Devuelve:
      - indices: mask o arreglo de índices de muestras que se usarán
      - y_new: etiquetas remapeadas (0..num_classes-1)
      - class_names: lista de nombres para las clases resultantes
      - num_classes: 1,2 o 3 (1 no esperado; si no hay clases, error)
    Reglas:
      - "0vs1" => keep only samples with y in {0,1}, map 0->0,1->1
      - "0vs12" => keep samples with y in {0,1,2}, map 0->0, (1,2)->1
      - "multiclass" => keep all, map unchanged (num_classes=3)
    """
    mode = parse_label_mode(mode)
    y = np.asarray(y)
    if mode == "multiclass":
        return np.arange(len(y)), y.copy(), ["Class 0","Class 1","Class 2"], 3

    # soportar patrones: "avuB"  e.g. "0vs1", "0vs12", "1vs2"
    if "vs" not in mode:
        raise ValueError(f"Mode inválido: {mode}. Usa '0vs1','1vs2','0vs12' o 'multiclass'.")

    left, right = mode.split("vs", 1)

    # parse sets: ejemplo left="0", right="12" -> right_set = {1,2}
    def parse_set(s):
        if s == "":
            return set()
        return set(int(c) for c in s)

    left_set = parse_set(left)
    right_set = parse_set(right)

    if not left_set or not right_set:
        raise ValueError("Sets empty in mode. Ejemplo válido: '0vs1', '0vs12'.")

    # Decide mapping: map left_set -> 0, right_set -> 1
    allowed = left_set.union(right_set)
    mask = np.isin(y, list(allowed))
    if mask.sum() == 0:
        raise ValueError("No samples for the requested mode.")

    # remap
    y_sub = y[mask].copy()
    y_new = np.zeros_like(y_sub)
    for orig in right_set:
        y_new[y_sub == orig] = 1
    for orig in left_set:
        y_new[y_sub == orig] = 0

    # class names
    left_name = "+".join(str(i) for i in sorted(left_set))
    right_name = "+".join(str(i) for i in sorted(right_set))
    class_names = [f"Class {left_name}", f"Class {right_name}"]
    return np.where(mask)[0], y_new, class_names, 2
