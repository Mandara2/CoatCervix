import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def save_confusion_matrix(y_true, y_pred, class_names, save_path, file_name="confusion_matrix.png", title="Confusion Matrix"):
    """
    Guarda una matriz de confusión en formato imagen.

    Parámetros:
    - y_true: etiquetas verdaderas
    - y_pred: etiquetas predichas
    - class_names: lista de nombres de clases
    - save_path: carpeta donde guardar la imagen
    - file_name: nombre del archivo PNG
    - title: título de la gráfica
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    file_path = os.path.join(save_path, file_name)
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Matriz de confusión guardada en: {file_path}")
    return file_path
