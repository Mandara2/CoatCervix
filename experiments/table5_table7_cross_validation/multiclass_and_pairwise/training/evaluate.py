from datetime import datetime
import os
import torch
from models.custom_coat import CustomCoaT
from config import settings
from training.utils import clean_cuda, get_parameter_summary
from training.metrics import compute_and_save_metrics

def evaluate_saved_model(model_path, model, X_test_loader, mean, std):
    device = settings.device
    model_name = settings.MODEL_NAME
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_dir = os.path.join(settings.SAVE_GRAPHICS_PATH, f"{model_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # ==================================================
    # 🔍 1. Verificar si existen pesos previos
    # ==================================================
    if os.path.exists(model_path):
        print(f"📥 Cargando pesos guardados: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"⚠️ No existen pesos previos, se evalúa con el modelo entrenado en memoria.")

    model.eval()
    get_parameter_summary(model)

    # ==================================================
    # 📊 2. Evaluación
    # ==================================================
    results = compute_and_save_metrics(
        model=model,
        X_test_loader=X_test_loader,
        class_names=settings.class_names,
        save_dir=save_dir,
        model_name=model_name,
        timestamp=timestamp,
        mean=mean,
        std=std,
        device=device
    )

    clean_cuda()
    return results
