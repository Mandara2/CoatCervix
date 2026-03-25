import os
import math
import numpy as np
import cv2
import torch
import torch.nn as nn

# Para Grad-CAM
from pytorch_grad_cam import LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Si usas settings para device
from config import settings
device = settings.device


def _find_target_layer_and_transform(model):
    """
    find a good target layer for Grad-CAM:
      - Preferences for Conv2d (CNN).
      - if it doesn't exist Conv2d, takes the last LayerNorm o BatchNorm
        and returns a reshape_transform that manage (B, N, C) -> (B, C, H, W).
    return(layer, reshape_transform_or_None, layer_name).
    """
    # 1) search Conv2d (ideal para CNNs)
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d):
            return module, None, name

    # 2) search LayerNorm or BatchNorm (posible ViT/CoaT)
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            def reshape_transform(tensor):
                # tensor could be  (B, C, H, W) o (B, N, C)
                if tensor.ndim == 4:
                    return tensor
                if tensor.ndim == 3:
                    # tensor shape: (B, N, C) -> reshape a (B, C, H, W) asume N es h^2 o h^2+1 (class token)
                    B, N, C = tensor.shape
                    # if N is a perfect square, no class token, else remove 1 token and check if it's a perfect square
                    h = int(math.sqrt(N))
                    if h * h == N:
                        # no class token
                        result = tensor.reshape(B, h, h, C).permute(0, 3, 1, 2)
                        return result
                    else:
                        # remove token 0 (posible class token)
                        result = tensor[:, 1:, :].reshape(B, int(math.sqrt(N-1)), int(math.sqrt(N-1)), C)
                        result = result.permute(0, 3, 1, 2)
                        return result
                raise ValueError("reshape_transform: tensor.ndim inesperado: %d" % tensor.ndim)
            return module, reshape_transform, name

    # 3) Fallback: take the last module 
    all_mods = list(model.named_modules())
    if len(all_mods) > 0:
        name, module = all_mods[-1]
        return module, None, name
    raise RuntimeError("module not found in the model (this should not happen).")


def generate_gradcam_new(
    model, image_tensor, target_class, save_path,
    class_names=None, model_name=None, mean=None, std=None,
    use_eigen_smooth=True, verbose=True
):
    """
    Robust version of Grad-CAM that automatically detects the target layer
    and handles CNN and transformer-like models (CoaT, ViT).
    - image_tensor: tensor shape (1, C, H, W) in the format that comes from the DataLoader (already normalized).
    - mean/std: arrays (3,) that are used to denormalize (values in [0,1]).
    """

    # --- checks básicos ---
    if image_tensor.dim() != 4 or image_tensor.size(0) != 1:
        raise ValueError("image_tensor must be (1, C, H, W)")

    #  float32 and on  device
    image_tensor = image_tensor.to(device).float()

    # find target layer and reshape_transform
    target_layer, reshape_transform, layer_name = _find_target_layer_and_transform(model)
    if verbose:
        print(f"[GradCAM] target layer: {layer_name}, reshape_transform: {reshape_transform is not None}")

    # create CAM object
    if reshape_transform is not None:
        cam = LayerCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform)
    else:
        cam = LayerCAM(model=model, target_layers=[target_layer])

    # prepare image for visualization (denormalize and convert to RGB float32 [0,1])
    img_np = image_tensor[0].cpu().permute(1, 2, 0).numpy()  # (H,W,C) but normalizado

    if (mean is None) or (std is None):
        # try to used global settings if exist
        raise ValueError("Pasa mean y std para revertir la normalización (arrays de 3 elementos).")
    mean_arr = np.array(mean).reshape(1,1,-1).astype(np.float32)
    std_arr  = np.array(std).reshape(1,1,-1).astype(np.float32)

    img_vis = (img_np * std_arr) + mean_arr     # revertir (X - mean)/std -> X
    img_vis = np.clip(img_vis, 0.0, 1.0).astype(np.float32)

    # generate the cam (shape: (batch, Hc, Wc) or (batch, 1, Hc, Wc) depending from the target layer)
    targets = [ClassifierOutputTarget(int(target_class))]
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets, eigen_smooth=use_eigen_smooth)

    # grayscale_cam  could be(1, Hc, Wc) o (1,1,Hc,Wc) -> manage both cases:
    if grayscale_cam.ndim == 4 and grayscale_cam.shape[1] == 1:
        grayscale_cam = grayscale_cam[:, 0, :, :]  # -> (1, Hc, Wc)

    # extract 2D map
    cam_map = grayscale_cam[0]   # (Hc, Wc)

    # resize cam_map to image size
    H, W, _ = img_vis.shape
    cam_map_resized = cv2.resize(cam_map, (W, H), interpolation=cv2.INTER_LINEAR)
    cam_map_resized = np.clip(cam_map_resized, 0.0, 1.0)

    # generate RGB visualization uint8 [0,255]
    visualization = show_cam_on_image(img_vis, cam_map_resized, use_rgb=True, colormap=cv2.COLORMAP_JET)

    # prepare saving
    os.makedirs(save_path, exist_ok=True)
    target_name = (class_names[int(target_class)] if class_names is not None else f"class_{int(target_class)}")
    fname = f"gradcam_{model_name or 'model'}_{target_name}.png"
    file_path = os.path.join(save_path, fname)

    # visualization is RGB uint8, but if it's not, convert it before saving
    if visualization.dtype != np.uint8:
        visualization = (visualization * 255).astype(np.uint8)
    visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, visualization_bgr)

    if verbose:
        print(f"[GradCAM] saved in: {file_path}    (img shape: {img_vis.shape}, cam shape: {cam_map_resized.shape})")

    return file_path
