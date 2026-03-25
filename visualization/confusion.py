import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def save_confusion_matrix(y_true, y_pred, class_names, save_path, file_name="confusion_matrix.png", title="Confusion Matrix"):
    """
    Generate and save a confusion matrix plot.

    Parameters:
    - y_true: true labels
    - y_pred: predicted labels
    - class_names: list of class names
    - save_path: folder where to save the image
    - file_name: name of the PNG file
    - title: title of the plot
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
    print(f"📊 Confusion matrix saved in: {file_path}")
    return file_path
