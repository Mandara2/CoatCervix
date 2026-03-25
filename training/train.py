import os
import numpy as np
import torch
from torch import nn, optim
from sklearn.metrics import f1_score
from models.custom_coat import CustomCoaT
from visualization.plots import save_loss_curve
from config import settings
from training.utils import get_parameter_summary, set_seed


def train_model(train_loader, val_loader, seed=settings.SEED):
    """
    Train the CustomCoaT model using the provided training and validation DataLoaders.
    """

    set_seed(seed)
    np.random.seed(seed)

    # === Paths ===
    weights_path = os.path.join(settings.SAVE_GRAPHICS_PATH, f"{settings.MODEL_NAME}.pth")
    os.makedirs(settings.SAVE_GRAPHICS_PATH, exist_ok=True)

    # === Initialize model ===
    model = CustomCoaT(pretrained=True, num_classes=settings.num_classes).to(settings.device)
    get_parameter_summary(model)

    # === Configure optimizer and criterion ===
    optimizer = optim.Adam(model.parameters(), lr=settings.lr)
    
    criterion = nn.CrossEntropyLoss()

    # optional: scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # === variables of interest ===
    history_train_loss, history_val_loss = [], []
    best_val_f1, best_val_acc = 0.0, 0.0
    best_model_state = None

    # === training cycle ===
    for epoch in range(1, settings.epochs + 1):
        model.train()
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(settings.device), labels.to(settings.device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        epoch_loss_train = running_loss / len(train_loader.dataset)
        history_train_loss.append(epoch_loss_train)

        # === Validation ===
        model.eval()
        val_running_loss = 0.0
        labels_all, preds_all = [], []

        with torch.no_grad():
            for imgs_v, labels_v in val_loader:
                imgs_v, labels_v = imgs_v.to(settings.device), labels_v.to(settings.device)
                outputs_v = model(imgs_v)
                loss_v = criterion(outputs_v, labels_v)
                val_running_loss += loss_v.item() * imgs_v.size(0)

                _, preds = torch.max(outputs_v, 1)
                labels_all.extend(labels_v.cpu().numpy())
                preds_all.extend(preds.cpu().numpy())

        epoch_loss_val = val_running_loss / len(val_loader.dataset)
        history_val_loss.append(epoch_loss_val)

        f1 = f1_score(labels_all, preds_all, average="macro")
        acc = np.mean(np.array(preds_all) == np.array(labels_all)) * 100

        # === save best model ===
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_val_acc = acc
            best_model_state = model.state_dict()
            torch.save(best_model_state, weights_path)
            print(f"💾 best weights save here: {weights_path}")

        print(f"[Epoch {epoch}/{settings.epochs}] "
              f"Train Loss: {epoch_loss_train:.4f} | "
              f"Val Loss: {epoch_loss_val:.4f} | "
              f"Val Acc: {acc:.2f}% | F1: {f1:.4f}")

    # === Restaurar los mejores pesos ===
    if best_model_state:
        model.load_state_dict(best_model_state)

    # === Guardar curva de pérdida ===
    save_loss_curve(
        history_train_loss,
        history_val_loss,
        settings.SAVE_GRAPHICS_PATH,
        epochs=settings.epochs
    )

    print(f"✅ Training completed | Best F1: {best_val_f1:.4f} | Best Accuracy: {best_val_acc:.2f}%")

    return {
        "model": model,
        "history_train_loss": history_train_loss,
        "history_val_loss": history_val_loss,
        "best_val_f1": best_val_f1,
        "best_val_acc": best_val_acc
    }
