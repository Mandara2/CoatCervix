import torch
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from visualization.confusion import save_confusion_matrix
from visualization.plots import save_roc_curve
from visualization.gradcam import generate_gradcam_new
import os

def compute_and_save_metrics(
    model,
    X_test_loader,
    class_names,
    save_dir,
    model_name,
    timestamp,
    mean=None,
    std=None,
    device="cpu"
):
    """
    Calcula métricas, matriz de confusión, ROC, Grad-CAM y CSV usando un DataLoader.
    Devuelve un dict con métricas.
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in X_test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Métricas principales
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro")
    rec = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")

    # Matriz de confusión
    save_confusion_matrix(
        all_labels, all_preds, class_names, save_dir,
        file_name=f"confusion_matrix_{model_name}_{timestamp}.png",
        title=f"Matriz de confusión (Test) - {model_name}"
    )

    # Curva ROC
    try:
        save_roc_curve(all_labels, all_probs, class_names, save_dir, epoch=f"{model_name}_TEST_{timestamp}")
    except Exception as e:
        print(f"⚠️ No se pudo generar ROC: {e}")

    # Grad-CAM de la primera imagen del primer batch
    try:
        imgs_one, labels_one = next(iter(X_test_loader))
        img_one = imgs_one[0].unsqueeze(0).to(device)
        label_one = int(labels_one[0].item())
        generate_gradcam_new(
            model=model,
            image_tensor=img_one,
            target_class=label_one,
            save_path=save_dir,
            class_names=class_names,
            model_name=f"{model_name}_TEST_best_{timestamp}",
            mean=mean,
            std=std
        )
    except Exception as e:
        print(f"⚠️ Error generando Grad-CAM: {e}")

    # Guardar métricas en CSV
    csv_path = os.path.join(save_dir, f"resultados_TEST_{model_name}_{timestamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Accuracy", "Precision", "Recall", "F1"])
        writer.writerow([model_name, f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}"])

    print(f"✅ Métricas guardadas en: {csv_path}")
    print(f"📊 Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

    return {
        "model": model_name,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "all_labels": all_labels,
        "all_preds": all_preds,
        "all_probs": all_probs
    }
