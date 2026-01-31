from datetime import datetime
import os
import torch
from models.custom_coat import CustomCoaT
from config import settings
from training.utils import clean_cuda, get_parameter_summary
from training.metrics import compute_and_save_metrics

def evaluate_saved_model(model_path, X_test_loader, mean, std):
    device = settings.device
    model_name = settings.MODEL_NAME
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_dir = os.path.join(settings.SAVE_GRAPHICS_PATH, f"{model_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    model = CustomCoaT(num_classes=settings.num_classes, pretrained=False).to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    get_parameter_summary(model)

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
