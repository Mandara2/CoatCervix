# xai_quant.py
"""
Quantitative interpretability of the final CoatCervix model (PeerJ revision, reviewer comment 8).
Single file: place it next to main.py of the code that trained the model
(Coat/new_model_interpretability, copied to revision_peerj/xai_quant) and run:

  python xai_quant.py verify                 # re-evaluates the checkpoint(s): must give the paper metrics
  python xai_quant.py run --n-images 3       # quick test (timing + everything runs)
  python xai_quant.py run                    # whole test set: faithfulness + sanity check + summary
  python xai_quant.py run --cam-layer backbone.serial_blocks4.7 --methods layercam gradcam random \
      --stages faithfulness summary --out <other folder>   # CAMs on the last transformer block

run:
  faithfulness  attribution maps (LayerCAM, Grad-CAM, Integrated Gradients, SHAP, random) and
                MoRF/LeRF perturbation curves on a 7 x 7 patch grid -> AOPC; AOPC bounds by beam
                search -> NAOPC (Edin et al., 2025, arXiv:2408.08137)
  sanity        cascading model-parameter randomization (Adebayo et al., 2018)
  summary       tables with bootstrap CIs, Wilcoxon tests vs. the random map and between methods, figures
Results are appended image by image, so an interrupted run continues where it stopped.
"""
import argparse
import copy
import csv
import datetime
import hashlib
import itertools
import json
import math
import os
import platform
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.nn.init import trunc_normal_

from config import settings
from data.loaders import load_data
from data.preprocessing import preprocess_all_datasets
from models.custom_coat import CustomCoaT
from training.utils import set_seed

# ---- final model of the paper: acc 0.8133 | prec 0.8396 | rec 0.7560 | macro F1 0.7810 (225 test images)
FINAL_WEIGHTS = ("/data/home/rtabares/Cervical/Coat/new_coat/results/graphics/"
                 "CoatCervixInterpretability/CoatCervixInterpretability.pth")
OTHER_MATCHING_WEIGHTS = [
    "/data/home/rtabares/Cervical/Coat/new_coat_int/results/graphics/NewCoatInt/NewCoatInt.pth",
    "/data/home/rtabares/Cervical/Coat/new_coat/results/graphics/NewCoat_mean_std/NewCoat_mean_std.pth",
]
DATA_PATH = "/data/home/rtabares/Cervical/Cervical_clean"
OUT_DIR = "/data/home/rtabares/Cervical/revision_peerj/xai_quant/results/main"
PAPER = {"acc": 0.8133, "prec": 0.8396, "rec": 0.7560, "f1": 0.7810}

ALL_METHODS = ["layercam", "gradcam", "ig", "shap", "random"]
METHOD_LABELS = {"layercam": "LayerCAM", "gradcam": "Grad-CAM", "ig": "Integrated Gradients",
                 "shap": "SHAP", "random": "Random"}
METHOD_COLORS = {"layercam": "#2a78d6", "gradcam": "#eb6834", "ig": "#1baf7a",
                 "shap": "#e87ba4", "random": "#8a8a85"}
METRICS = ["aopc_morf", "aopc_lerf", "naopc_morf", "naopc_lerf"]


def safe_wilcoxon(a, b, **kw):
    """Wilcoxon signed-rank; identical paired samples (all differences 0) give p = 1."""
    if np.allclose(a, b):
        return type("W", (), {"statistic": float("nan"), "pvalue": 1.0})()
    return wilcoxon(a, b, **kw)


# =====================================================================================
# Data and model (same loading and preprocessing as pipeline/pipeline_runner.py)
# =====================================================================================
def load_test_data(data_path):
    settings.DATA_PATH = data_path
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    X_train, X_val, X_test, mean, std = preprocess_all_datasets(X_train, X_val, X_test, target_size=settings.target_size)
    X_test_t = torch.from_numpy(np.transpose(X_test, (0, 3, 1, 2))).float()
    return X_train, X_test_t, np.asarray(y_test).astype(int), mean, std


def load_model(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model = CustomCoaT(num_classes=settings.num_classes, pretrained=False).to(device)
    model.load_state_dict(state, strict=True)
    return model.eval()


@torch.no_grad()
def predict_probs(model, X, device, batch_size=32):
    out = [torch.softmax(model(X[i:i + batch_size].to(device)), dim=1).cpu() for i in range(0, len(X), batch_size)]
    return torch.cat(out)


def to_rgb(x_norm, mean, std):
    """Undo the z-score for display: (C, H, W) tensor -> (H, W, 3) in [0, 1]."""
    img = x_norm.cpu().permute(1, 2, 0).numpy() * (std + 1e-8) + mean
    return np.clip(img, 0, 1)


# =====================================================================================
# verify: which checkpoint reproduces the paper
# =====================================================================================
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_verify(args):
    device = settings.device
    weights = args.weights or [FINAL_WEIGHTS] + OTHER_MATCHING_WEIGHTS
    _, X_test, y_test, mean, std = load_test_data(args.data_path)
    print(f"📥 Test set: {len(y_test)} images | class counts: {np.bincount(y_test).tolist()} | device: {device}")
    print(f"   train mean {np.round(mean, 4).tolist()} | std {np.round(std, 4).tolist()}")

    rows = []
    for path in weights:
        row = {"path": path, "modified": datetime.datetime.fromtimestamp(os.path.getmtime(path)).isoformat(sep=" ", timespec="seconds"),
               "sha256": sha256(path)[:16]}
        try:
            model = load_model(path, device)
        except Exception as e:  # noqa: BLE001 - reported, not fatal
            row["status"] = f"not a CoatCervix checkpoint ({type(e).__name__})"
            rows.append(row)
            print(f"⏭️  {path}: {row['status']}")
            continue
        preds = predict_probs(model, X_test, device).argmax(dim=1).numpy()
        metrics = {"acc": accuracy_score(y_test, preds),
                   "prec": precision_score(y_test, preds, average="macro", zero_division=0),
                   "rec": recall_score(y_test, preds, average="macro", zero_division=0),
                   "f1": f1_score(y_test, preds, average="macro", zero_division=0)}
        max_diff = max(abs(metrics[k] - PAPER[k]) for k in PAPER)
        row.update({k: round(v, 4) for k, v in metrics.items()})
        row.update({"correct": int((preds == y_test).sum()), "max_diff_vs_paper": round(max_diff, 4),
                    "status": "MATCHES PAPER" if max_diff <= args.tol else "different",
                    "confusion": confusion_matrix(y_test, preds).tolist()})
        rows.append(row)
        print(f"{'✅' if row['status'] == 'MATCHES PAPER' else '❌'} {path}\n"
              f"     {row['modified']} | sha256 {row['sha256']} | acc {metrics['acc']:.4f} prec {metrics['prec']:.4f} "
              f"rec {metrics['rec']:.4f} f1 {metrics['f1']:.4f} | {row['correct']}/{len(y_test)} correct | "
              f"confusion {row['confusion']}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights_verification.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["status", "path", "modified", "sha256", "acc", "prec", "rec", "f1",
                                               "correct", "max_diff_vs_paper", "confusion"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n📄 Report: {out}")
    matches = [r for r in rows if r["status"] == "MATCHES PAPER"]
    if matches and matches[0]["path"] == weights[0]:
        print(f"🎯 Confirmed final model: {weights[0]}")
    else:
        print("⚠️ The first checkpoint does not reproduce the paper metrics; do not run the analysis with it.")


# =====================================================================================
# Attribution maps
# =====================================================================================
class TargetLogit(nn.Module):
    """Wraps the classifier so it outputs only the logit of one class, shape (B, 1)."""

    def __init__(self, model, target):
        super().__init__()
        self.model = model
        self.target = int(target)

    def forward(self, x):
        return self.model(x)[:, self.target:self.target + 1]


def tokens_to_map(tensor):
    """(B, 1 + h*w, C) tokens with a class token, or (B, h*w, C) -> (B, C, h, w)."""
    if tensor.ndim == 4:
        return tensor
    b, n, c = tensor.shape
    side = int(math.isqrt(n))
    if side * side != n:
        tensor, n = tensor[:, 1:, :], n - 1
        side = int(math.isqrt(n))
    return tensor.reshape(b, side, side, c).permute(0, 3, 1, 2)


def _shap_to_map(values):
    """SHAP returns a list or an array (1, C, H, W[, 1]) depending on the version -> (H, W)."""
    if isinstance(values, (list, tuple)):
        values = values[0]
    values = np.asarray(values)
    if values.ndim == 5:
        values = values[..., 0]
    return values[0].sum(axis=0).astype(np.float32)


class Explainer:
    """
    explain(method, x, target) -> (H, W) float32 map, higher = more evidence for `target`.
      layercam  LayerCAM, the method behind the paper figures (visualization/gradcam.py)
      gradcam   classic Grad-CAM on the same layer
      ig        Integrated Gradients (Captum), zero baseline = training-set mean after z-score
      shap      SHAP DeepExplainer; if it is not exact for CoaT (additivity check fails on the
                first image) GradientExplainer (expected gradients) is used and the reason is kept
      random    uniform random map (control)
    cam_layer "auto" = layer chosen by visualization/gradcam.py; or a module name, e.g.
    "backbone.serial_blocks4.7" (tokens of the last transformer block as a 7 x 7 map).
    """

    def __init__(self, model, device, background=None, ig_steps=50, grad_batch_size=16,
                 shap_explainer="auto", shap_nsamples=200, cam_eigen_smooth=True, cam_layer="auto", seed=42):
        self.model = model
        self.device = device
        self.background = background
        self.ig_steps = ig_steps
        self.grad_batch_size = grad_batch_size
        self.shap_explainer = shap_explainer
        self.shap_used = None if shap_explainer == "auto" else shap_explainer
        self.shap_fallback_reason = None
        self.shap_nsamples = shap_nsamples
        self.cam_eigen_smooth = cam_eigen_smooth
        self.cam_layer = cam_layer
        self.seed = seed
        self.target_layer_name = None

    def explain(self, method, x, target, image_id=0):
        x = x.to(self.device).float()
        if method in ("layercam", "gradcam"):
            return self._cam(method, x, target)
        if method == "ig":
            return self._ig(x, target)
        if method == "shap":
            return self._shap(x, target)
        if method == "random":
            rng = np.random.default_rng(self.seed + int(image_id))
            return rng.random(x.shape[-2:]).astype(np.float32)
        raise ValueError(f"Unknown attribution method: {method}")

    def _cam(self, method, x, target):
        from pytorch_grad_cam import GradCAM, LayerCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from visualization.gradcam import _find_target_layer_and_transform
        if self.cam_layer == "auto":
            layer, reshape_transform, name = _find_target_layer_and_transform(self.model)
        else:
            layer, reshape_transform, name = self.model.get_submodule(self.cam_layer), tokens_to_map, self.cam_layer
        self.target_layer_name = name
        cam_cls = LayerCAM if method == "layercam" else GradCAM
        # the context manager removes the hooks afterwards, otherwise every later forward
        # pass (perturbation curves) would keep storing activations
        with cam_cls(model=self.model, target_layers=[layer], reshape_transform=reshape_transform) as cam:
            grayscale = cam(input_tensor=x, targets=[ClassifierOutputTarget(int(target))],
                            eigen_smooth=self.cam_eigen_smooth)
        grayscale = np.asarray(grayscale)
        if grayscale.ndim == 4:
            grayscale = grayscale[:, 0]
        return grayscale[0].astype(np.float32)

    def _ig(self, x, target):
        from captum.attr import IntegratedGradients
        attr = IntegratedGradients(self.model).attribute(
            x, baselines=torch.zeros_like(x), target=int(target),
            n_steps=self.ig_steps, internal_batch_size=self.grad_batch_size)
        # sum over color channels keeps the sign: positive = supports the target class
        return attr[0].sum(dim=0).detach().cpu().numpy().astype(np.float32)

    def _shap(self, x, target):
        if self.shap_used is None:  # auto mode, the first image decides
            try:
                values = self._shap_deep(x, target, check_additivity=True)
                self.shap_used = "deep"
                return values
            except Exception as e:  # noqa: BLE001 - any failure means DeepExplainer is not exact here
                self.shap_used = "gradient"
                self.shap_fallback_reason = f"{type(e).__name__}: {str(e)[:300]}"
                print(f"[SHAP] DeepExplainer not usable ({self.shap_fallback_reason}); using GradientExplainer.")
        if self.shap_used == "deep":
            return self._shap_deep(x, target, check_additivity=self.shap_explainer == "deep")
        return self._shap_gradient(x, target)

    def _shap_deep(self, x, target, check_additivity):
        import shap
        explainer = shap.DeepExplainer(TargetLogit(self.model, target).eval(), self.background.to(self.device))
        return _shap_to_map(explainer.shap_values(x, check_additivity=check_additivity))

    def _shap_gradient(self, x, target):
        import shap
        explainer = shap.GradientExplainer(TargetLogit(self.model, target).eval(), self.background.to(self.device),
                                           batch_size=self.grad_batch_size)
        np.random.seed(self.seed)  # reproducible expected-gradients sampling
        torch.manual_seed(self.seed)
        return _shap_to_map(explainer.shap_values(x, nsamples=self.shap_nsamples))


# =====================================================================================
# Perturbation faithfulness on a grid of patches
#   curve[k] = p(target | first k patches of the order replaced by the baseline), k = 0..P
#   AOPC = 1/(P+1) * sum_k (curve[0] - curve[k])
#   MoRF (attribution order): higher AOPC = more faithful; LeRF (reverse order): lower is better
#   NAOPC = (AOPC - AOPC_min) / (AOPC_max - AOPC_min), bounds over all orders by beam search
# =====================================================================================
def patch_masks(grid, height, width, device):
    ys = np.linspace(0, height, grid + 1).astype(int)
    xs = np.linspace(0, width, grid + 1).astype(int)
    masks = torch.zeros(grid * grid, height, width, dtype=torch.bool, device=device)
    for i in range(grid):
        for j in range(grid):
            masks[i * grid + j, ys[i]:ys[i + 1], xs[j]:xs[j + 1]] = True
    return masks


def patch_scores(attr_map, grid):
    h, w = attr_map.shape
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    return np.array([attr_map[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].sum()
                     for i in range(grid) for j in range(grid)], dtype=np.float64)


def make_baseline(x, kind="mean", blur_sigma=10.0):
    """mean -> zeros (= training-set mean color after z-score); blur -> blurred copy of the image."""
    if kind == "mean":
        return torch.zeros_like(x)
    img = x[0].detach().cpu().permute(1, 2, 0).numpy()
    blurred = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=blur_sigma)
    return torch.from_numpy(blurred).permute(2, 0, 1).unsqueeze(0).to(x.device, x.dtype)


class PatchPerturber:
    """Target-class probability for many sets of removed patches of one image."""

    def __init__(self, model, x, baseline, target, masks, batch_size=64, amp=False):
        self.model = model
        self.x = x
        self.baseline = baseline
        self.target = int(target)
        self.masks_flat = masks.reshape(masks.shape[0], -1).float()
        self.h, self.w = masks.shape[1:]
        self.batch_size = batch_size
        self.amp = amp
        self.n_forward = 0

    @torch.no_grad()
    def probs(self, removed):
        """removed: bool (N, P), True = patch replaced by the baseline -> (N,) probabilities."""
        out = []
        for start in range(0, removed.shape[0], self.batch_size):
            member = removed[start:start + self.batch_size].float()
            m = (member @ self.masks_flat).clamp_(max=1.0).view(-1, 1, self.h, self.w)
            imgs = self.x * (1.0 - m) + self.baseline * m
            with torch.autocast(device_type=self.x.device.type, enabled=self.amp):
                logits = self.model(imgs)
            out.append(torch.softmax(logits.float(), dim=1)[:, self.target])
            self.n_forward += imgs.shape[0]
        return torch.cat(out)

    def curve(self, order):
        n_patches = len(order)
        removed = torch.zeros(n_patches + 1, n_patches, dtype=torch.bool, device=self.x.device)
        for k in range(1, n_patches + 1):
            removed[k] = removed[k - 1]
            removed[k, int(order[k - 1])] = True
        return self.probs(removed).cpu().numpy()


def aopc(curve):
    curve = np.asarray(curve, dtype=np.float64)
    return float(np.mean(curve[0] - curve))


def morf_lerf_curves(perturber, attr_map, grid):
    order_morf = np.argsort(-patch_scores(attr_map, grid), kind="stable")
    return perturber.curve(order_morf), perturber.curve(order_morf[::-1].copy())


def aopc_bounds(perturber, n_patches, beam_size=5, maximize=True):
    """
    Beam search for the highest (maximize) or lowest AOPC over removal orders. The probability
    after removing a set of patches does not depend on the order, so states with the same set
    are merged keeping the best partial sum. Cost: about beam_size * P^2 / 2 forward passes.
    """
    device = perturber.x.device
    p0 = float(perturber.probs(torch.zeros(1, n_patches, dtype=torch.bool, device=device))[0])
    beams = [(np.zeros(n_patches, dtype=bool), 0.0, [])]  # (removed set, sum of p0 - p_k, order)
    for _ in range(n_patches):
        candidates = {}
        for removed, partial, order in beams:
            for j in np.flatnonzero(~removed):
                new_removed = removed.copy()
                new_removed[j] = True
                key = new_removed.tobytes()
                if key not in candidates:
                    candidates[key] = [new_removed, []]
                candidates[key][1].append((partial, order + [int(j)]))
        keys = list(candidates.keys())
        probs = perturber.probs(torch.from_numpy(np.stack([candidates[k][0] for k in keys])).to(device)).cpu().numpy()
        merged = []
        for key, p in zip(keys, probs):
            new_removed, parents = candidates[key]
            partial, order = (max if maximize else min)(parents, key=lambda t: t[0])
            merged.append((new_removed, partial + (p0 - float(p)), order))
        merged.sort(key=lambda t: t[1], reverse=maximize)
        beams = merged[:beam_size]
    _, best_partial, best_order = beams[0]
    return best_partial / (n_patches + 1), best_order  # k = 0 term is zero


# =====================================================================================
# Sanity check: cascading randomization from the head to stage 1
# =====================================================================================
def cascade_groups(model):
    bb = model.backbone
    head = [model.classifier] + [getattr(bb, n) for n in ("norm4", "norm", "head")
                                 if isinstance(getattr(bb, n, None), nn.Module)]
    groups = [("head", head, [])]
    for s in (4, 3, 2, 1):
        modules = [getattr(bb, n) for n in (f"serial_blocks{s}", f"crpe{s}", f"cpe{s}", f"patch_embed{s}", f"norm{s}")
                   if isinstance(getattr(bb, n, None), nn.Module)]
        params = [getattr(bb, f"cls_token{s}")] if isinstance(getattr(bb, f"cls_token{s}", None), nn.Parameter) else []
        if modules or params:
            groups.append((f"stage{s}", modules, params))
    return groups


@torch.no_grad()
def randomize_(modules, params):
    for module in modules:
        for sub in module.modules():
            if isinstance(sub, nn.Linear):
                trunc_normal_(sub.weight, std=0.02)
                if sub.bias is not None:
                    nn.init.zeros_(sub.bias)
            elif hasattr(sub, "reset_parameters") and not list(sub.children()):
                sub.reset_parameters()  # Conv2d, LayerNorm, ...
    for p in params:
        trunc_normal_(p, std=0.02)


def map_similarity(original, randomized):
    from skimage.metrics import structural_similarity
    a, b = original.ravel(), randomized.ravel()

    def _norm(m):
        m = np.abs(m)
        rng = m.max() - m.min()
        return (m - m.min()) / rng if rng > 0 else np.zeros_like(m)

    return (float(spearmanr(np.abs(a), np.abs(b)).correlation), float(spearmanr(a, b).correlation),
            float(structural_similarity(_norm(original), _norm(randomized), data_range=1.0)))


def run_cascade(model, explainer, images, targets, methods, seed=42):
    rows, example_maps, original = [], {m: [] for m in methods}, {}
    for i in range(len(images)):
        for m in methods:
            original[(i, m)] = explainer.explain(m, images[i:i + 1], targets[i], image_id=i)
    for m in methods:
        example_maps[m].append(original[(0, m)])

    torch.manual_seed(seed)
    random_model = copy.deepcopy(model).eval()
    trained_model, explainer.model = explainer.model, random_model
    try:
        for step, (name, modules, params) in enumerate(cascade_groups(random_model), start=1):
            randomize_(modules, params)
            for i in range(len(images)):
                for m in methods:
                    if m == "random":
                        continue  # does not depend on the model
                    rand_map = explainer.explain(m, images[i:i + 1], targets[i], image_id=i)
                    sp_abs, sp, ssim = map_similarity(original[(i, m)], rand_map)
                    rows.append({"image": i, "method": m, "step": step, "randomized_up_to": name,
                                 "spearman_abs": sp_abs, "spearman": sp, "ssim": ssim})
                    if i == 0:
                        example_maps[m].append(rand_map)
            print(f"[Sanity] randomized up to {name}")
    finally:
        explainer.model = trained_model
    return rows, example_maps


# =====================================================================================
# Summary: NAOPC, tables, tests, figures
# =====================================================================================
def add_naopc(per_image, bounds):
    """Beam search approximates the bounds, so they are widened with every evaluated order's AOPC."""
    df = per_image.merge(bounds[["idx", "aopc_max", "aopc_min"]], on="idx", how="inner")
    seen = df.groupby("idx")[["aopc_morf", "aopc_lerf"]].agg(["max", "min"])
    upper_seen = seen[[("aopc_morf", "max"), ("aopc_lerf", "max")]].max(axis=1).rename("upper_seen")
    lower_seen = seen[[("aopc_morf", "min"), ("aopc_lerf", "min")]].min(axis=1).rename("lower_seen")
    df = df.merge(upper_seen, left_on="idx", right_index=True).merge(lower_seen, left_on="idx", right_index=True)
    df["upper"] = np.maximum(df["aopc_max"], df["upper_seen"])
    df["lower"] = np.minimum(df["aopc_min"], df["lower_seen"])
    span = (df["upper"] - df["lower"]).replace(0, np.nan)
    df["naopc_morf"] = (df["aopc_morf"] - df["lower"]) / span
    df["naopc_lerf"] = (df["aopc_lerf"] - df["lower"]) / span
    first = df.groupby("idx").first()
    info = {"images": int(len(first)), "upper_widened": int((first["upper_seen"] > first["aopc_max"]).sum()),
            "lower_widened": int((first["lower_seen"] < first["aopc_min"]).sum())}
    return df.drop(columns=["upper_seen", "lower_seen"]), info


def bootstrap_ci(values, n_boot=10000, seed=42):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    means = np.random.default_rng(seed).choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(df, group_cols):
    rows = []
    for keys, g in df.groupby(group_cols):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row["n"] = len(g)
        for metric in METRICS:
            lo, hi = bootstrap_ci(g[metric])
            row.update({f"{metric}_mean": g[metric].mean(), f"{metric}_std": g[metric].std(),
                        f"{metric}_ci95_low": lo, f"{metric}_ci95_high": hi})
        rows.append(row)
    return pd.DataFrame(rows)


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    adjusted, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def _paired(df, metric, a, b):
    wide = df.pivot(index="idx", columns="method", values=metric)[[a, b]].dropna()
    return wide[a].to_numpy(), wide[b].to_numpy()


def stats_vs_random(df, methods):
    rows = []
    for metric, alternative in (("naopc_morf", "greater"), ("naopc_lerf", "less"),
                                ("aopc_morf", "greater"), ("aopc_lerf", "less")):
        for m in methods:
            if m == "random":
                continue
            a, b = _paired(df, metric, m, "random")
            res = safe_wilcoxon(a, b, alternative=alternative)
            rows.append({"metric": metric, "method": m, "alternative": f"{m} {'>' if alternative == 'greater' else '<'} random",
                         "n": len(a), "median_diff": float(np.median(a - b)),
                         "statistic": float(res.statistic), "p_value": float(res.pvalue)})
    out = pd.DataFrame(rows)
    out["p_holm"] = out.groupby("metric")["p_value"].transform(lambda s: holm(s.to_numpy()))
    return out


def stats_pairwise(df, methods):
    rows = []
    for metric in ("naopc_morf", "naopc_lerf"):
        for a_name, b_name in itertools.combinations([m for m in methods if m != "random"], 2):
            a, b = _paired(df, metric, a_name, b_name)
            res = safe_wilcoxon(a, b, alternative="two-sided")
            rows.append({"metric": metric, "method_a": a_name, "method_b": b_name, "n": len(a),
                         "median_diff_a_minus_b": float(np.median(a - b)),
                         "statistic": float(res.statistic), "p_value": float(res.pvalue)})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = out.groupby("metric")["p_value"].transform(lambda s: holm(s.to_numpy()))
    return out


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#b5b4ad")
    ax.spines["bottom"].set_color("#b5b4ad")
    ax.tick_params(colors="#55554f", labelsize=9)
    ax.grid(axis="y", color="#e6e5df", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_curves(df, methods, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, key, title in ((axes[0], "morf", "MoRF: most relevant patches removed first (lower is better)"),
                           (axes[1], "lerf", "LeRF: least relevant patches removed first (higher is better)")):
        for m in methods:
            curves = np.stack(df.loc[df["method"] == m, key].to_list())
            mean = curves.mean(axis=0)
            ci = 1.96 * curves.std(axis=0, ddof=1) / np.sqrt(len(curves))
            frac = np.linspace(0, 1, curves.shape[1])
            ax.plot(frac, mean, "--" if m == "random" else "-", color=METHOD_COLORS[m], linewidth=2, label=METHOD_LABELS[m])
            ax.fill_between(frac, mean - ci, mean + ci, color=METHOD_COLORS[m], alpha=0.12, linewidth=0)
        ax.set_title(title, fontsize=10, color="#1f1f1d", loc="left")
        ax.set_xlabel("Fraction of the image removed", fontsize=9, color="#55554f")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        _style(ax)
    axes[0].set_ylabel("Probability of the explained class", fontsize=9, color="#55554f")
    axes[1].legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_sanity(sanity_df, path):
    steps = sanity_df.drop_duplicates("step").sort_values("step")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, metric, title in ((axes[0], "spearman_abs", "Spearman rank correlation of |map|"),
                              (axes[1], "ssim", "SSIM of |map|")):
        for m in [k for k in METHOD_LABELS if k in set(sanity_df["method"])]:
            agg = sanity_df[sanity_df["method"] == m].groupby("step")[metric].agg(["mean", "std"]).reindex(steps["step"])
            x = np.arange(1, len(agg) + 1)
            ax.plot(x, agg["mean"], "-o", color=METHOD_COLORS[m], linewidth=2, markersize=5, label=METHOD_LABELS[m])
            ax.fill_between(x, agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                            color=METHOD_COLORS[m], alpha=0.12, linewidth=0)
        ax.set_xticks(np.arange(1, len(steps) + 1))
        ax.set_xticklabels([f"+{n}" for n in steps["randomized_up_to"]], fontsize=8)
        ax.set_xlabel("Cumulative randomization (output -> input)", fontsize=9, color="#55554f")
        ax.set_title(title, fontsize=10, color="#1f1f1d", loc="left")
        ax.set_ylim(-0.2, 1.05)
        _style(ax)
    axes[0].set_ylabel("Similarity to the trained-model map", fontsize=9, color="#55554f")
    axes[1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _show_map(ax, m):
    lim = np.abs(m).max() or 1.0
    if m.min() < 0:
        ax.imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim)
    else:
        ax.imshow(m, cmap="viridis", vmin=0, vmax=lim)
    ax.axis("off")


def plot_map_grid(rows, col_titles, path):
    """rows: list of (row_title, image_rgb, [maps])"""
    n_cols = 1 + len(col_titles)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(1.7 * n_cols, 1.8 * len(rows)), squeeze=False)
    for r, (row_title, img, maps) in enumerate(rows):
        axes[r, 0].imshow(img)
        axes[r, 0].axis("off")
        axes[r, 0].set_title(row_title, fontsize=7, loc="left")
        for c, amap in enumerate(maps, start=1):
            _show_map(axes[r, c], amap)
            if r == 0:
                axes[r, c].set_title(col_titles[c - 1], fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# =====================================================================================
# run: faithfulness + sanity + summary
# =====================================================================================
def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def cmd_run(args):
    device = settings.device
    set_seed(args.seed)
    os.makedirs(os.path.join(args.out, "maps"), exist_ok=True)
    info_path = os.path.join(args.out, "run_info.json")
    run_info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    run_info.update({"args": vars(args), "device": str(device), "torch": torch.__version__,
                     "python": platform.python_version()})

    print("📥 Loading and preprocessing data...")
    X_train, X_test_t, y_test, mean, std = load_test_data(args.data_path)
    n_images = len(X_test_t) if args.n_images is None else min(args.n_images, len(X_test_t))
    bg_idx = np.random.default_rng(args.seed).choice(len(X_train), size=min(args.shap_background, len(X_train)), replace=False)
    background = torch.from_numpy(np.transpose(X_train[bg_idx], (0, 3, 1, 2))).float()

    model = load_model(args.weights, device)
    for p in model.parameters():
        p.requires_grad_(True)
    preds_all = predict_probs(model, X_test_t, device).argmax(dim=1).numpy()
    run_info["test_accuracy_full"] = float((preds_all == y_test).mean())
    print(f"✅ Model loaded | test accuracy {run_info['test_accuracy_full']:.4f} on {len(y_test)} images "
          f"(paper: {PAPER['acc']:.4f}) | using {n_images} images")
    if abs(run_info["test_accuracy_full"] - PAPER["acc"]) > 1e-4:
        print("⚠️ Accuracy differs from the paper: check --weights and --data-path before trusting the results.")

    explainer = Explainer(model, device, background=background, ig_steps=args.ig_steps,
                          grad_batch_size=args.grad_batch_size, shap_explainer=args.shap_explainer,
                          shap_nsamples=args.shap_nsamples, cam_eigen_smooth=not args.no_cam_eigen_smooth,
                          cam_layer=args.cam_layer, seed=args.seed)
    if run_info.get("shap_used") and args.shap_explainer == "auto":
        explainer.shap_used = run_info["shap_used"]  # keep the same SHAP variant when resuming

    def save_info():
        run_info.update({"shap_used": explainer.shap_used,
                         "shap_fallback_reason": explainer.shap_fallback_reason or run_info.get("shap_fallback_reason"),
                         "cam_target_layer": explainer.target_layer_name or run_info.get("cam_target_layer")})
        with open(info_path, "w") as f:
            json.dump(run_info, f, indent=2)

    targets = preds_all if args.target == "pred" else y_test
    per_image_path = os.path.join(args.out, "per_image.jsonl")
    bounds_path = os.path.join(args.out, "bounds.jsonl")

    if "faithfulness" in args.stages:
        done = {(r["idx"], r["method"]) for r in read_jsonl(per_image_path)}
        done_bounds = {r["idx"] for r in read_jsonl(bounds_path)}
        masks = patch_masks(args.grid, *X_test_t.shape[-2:], device=device)
        n_patches = args.grid * args.grid
        for idx in range(n_images):
            todo = [m for m in args.methods if (idx, m) not in done]
            if not todo and idx in done_bounds:
                continue
            t0 = time.time()
            x = X_test_t[idx:idx + 1].to(device)
            target = int(targets[idx])
            perturber = PatchPerturber(model, x, make_baseline(x, args.baseline), target, masks,
                                       batch_size=args.batch_size, amp=args.amp)
            maps_file = os.path.join(args.out, "maps", f"{idx:04d}.npz")
            maps = dict(np.load(maps_file)) if os.path.exists(maps_file) else {}
            # every map is stored before its curves are logged, so a resumed run never ends up
            # with results whose map is missing (also fills maps lost by an interrupted run)
            missing = [m for m in args.methods if m not in maps]
            for m in missing:
                maps[m] = explainer.explain(m, x, target, image_id=idx)
            if missing:
                np.savez_compressed(maps_file, **{k: v.astype(np.float16) for k, v in maps.items()})
            for m in todo:
                morf, lerf = morf_lerf_curves(perturber, maps[m].astype(np.float32), args.grid)
                append_jsonl(per_image_path, {"idx": idx, "label": int(y_test[idx]), "pred": int(preds_all[idx]),
                                              "target": target, "method": m, "p0": float(morf[0]),
                                              "aopc_morf": aopc(morf), "aopc_lerf": aopc(lerf),
                                              "morf": morf.tolist(), "lerf": lerf.tolist()})
            if idx not in done_bounds:
                upper, upper_order = aopc_bounds(perturber, n_patches, args.beam_size, maximize=True)
                lower, lower_order = aopc_bounds(perturber, n_patches, args.beam_size, maximize=False)
                append_jsonl(bounds_path, {"idx": idx, "aopc_max": upper, "aopc_min": lower,
                                           "order_max": upper_order, "order_min": lower_order})
            save_info()
            print(f"[{idx + 1}/{n_images}] class {target} | {len(todo)} methods | "
                  f"{perturber.n_forward} forwards | {time.time() - t0:.1f}s", flush=True)

    if "sanity" in args.stages:
        n_sanity = min(args.sanity_images, n_images)
        print(f"🧪 Sanity check on {n_sanity} images...")
        rows, example_maps = run_cascade(model, explainer, X_test_t[:n_sanity], targets[:n_sanity],
                                         args.methods, seed=args.seed)
        sanity_df = pd.DataFrame(rows)
        sanity_df.to_csv(os.path.join(args.out, "sanity.csv"), index=False)
        plot_sanity(sanity_df, os.path.join(args.out, "sanity_cascade.png"))
        step_names = ["trained"] + [f"+{name}" for name, _, _ in cascade_groups(model)]
        img0 = to_rgb(X_test_t[0], mean, std)
        plot_map_grid([(METHOD_LABELS[m], img0, example_maps[m]) for m in args.methods if m != "random"],
                      step_names, os.path.join(args.out, "sanity_example.png"))
        save_info()

    if "summary" in args.stages:
        per_image = pd.DataFrame(read_jsonl(per_image_path))
        bounds = pd.DataFrame(read_jsonl(bounds_path))
        if per_image.empty or bounds.empty:
            print("⚠️ No faithfulness results yet, run the 'faithfulness' stage first.")
        else:
            methods = [m for m in args.methods if m in set(per_image["method"])]
            df, widen = add_naopc(per_image, bounds)
            df["class"] = df["label"].map(lambda i: settings.class_names[int(i)])
            summarize(df, ["method"]).to_csv(os.path.join(args.out, "summary_overall.csv"), index=False)
            summarize(df, ["class", "method"]).to_csv(os.path.join(args.out, "summary_by_class.csv"), index=False)
            if "random" in methods:
                stats_vs_random(df, methods).to_csv(os.path.join(args.out, "stats_vs_random.csv"), index=False)
            stats_pairwise(df, methods).to_csv(os.path.join(args.out, "stats_pairwise.csv"), index=False)
            df.drop(columns=["morf", "lerf"]).to_csv(os.path.join(args.out, "per_image_naopc.csv"), index=False)
            plot_curves(df, methods, os.path.join(args.out, "perturbation_curves.png"))
            run_info["naopc_bounds_widened"] = widen

            examples = []  # first correctly classified image of every class with all its maps
            for c in range(settings.num_classes):
                for i in sorted(set(per_image["idx"])):
                    maps_file = os.path.join(args.out, "maps", f"{i:04d}.npz")
                    if y_test[i] != c or preds_all[i] != c or not os.path.exists(maps_file):
                        continue
                    maps = dict(np.load(maps_file))
                    if all(m in maps for m in methods):
                        examples.append((settings.class_names[c], to_rgb(X_test_t[i], mean, std),
                                         [maps[m].astype(np.float32) for m in methods]))
                        break
            if examples:
                plot_map_grid(examples, [METHOD_LABELS[m] for m in methods],
                              os.path.join(args.out, "attribution_examples.png"))
            save_info()
            print(pd.read_csv(os.path.join(args.out, "summary_overall.csv"))
                  [["method", "n", "aopc_morf_mean", "aopc_lerf_mean", "naopc_morf_mean", "naopc_lerf_mean"]]
                  .to_string(index=False))
    print(f"🎯 Done. Results in: {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    v = sub.add_parser("verify", help="re-evaluate checkpoints against the paper metrics")
    v.add_argument("--weights", nargs="*", default=None, help=f"default: {FINAL_WEIGHTS} and the other matching runs")
    v.add_argument("--data-path", default=DATA_PATH)
    v.add_argument("--tol", type=float, default=0.00006)

    r = sub.add_parser("run", help="quantitative interpretability")
    r.add_argument("--weights", default=FINAL_WEIGHTS)
    r.add_argument("--data-path", default=DATA_PATH)
    r.add_argument("--out", default=OUT_DIR)
    r.add_argument("--stages", nargs="+", default=["faithfulness", "sanity", "summary"],
                   choices=["faithfulness", "sanity", "summary"])
    r.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    r.add_argument("--n-images", type=int, default=None, help="only the first N test images (quick test)")
    r.add_argument("--target", default="pred", choices=["pred", "label"])
    r.add_argument("--grid", type=int, default=7, help="patches per side (7 -> 49 patches of 32 px)")
    r.add_argument("--baseline", default="mean", choices=["mean", "blur"])
    r.add_argument("--beam-size", type=int, default=5, help="beam width for the NAOPC bounds (1 = greedy)")
    r.add_argument("--batch-size", type=int, default=64)
    r.add_argument("--amp", action="store_true")
    r.add_argument("--ig-steps", type=int, default=50)
    r.add_argument("--grad-batch-size", type=int, default=16)
    r.add_argument("--shap-explainer", default="auto", choices=["auto", "deep", "gradient"])
    r.add_argument("--shap-background", type=int, default=50)
    r.add_argument("--shap-nsamples", type=int, default=200)
    r.add_argument("--no-cam-eigen-smooth", action="store_true", help="the paper figures used eigen_smooth")
    r.add_argument("--cam-layer", default="auto")
    r.add_argument("--sanity-images", type=int, default=30)
    r.add_argument("--seed", type=int, default=settings.SEED)

    args = parser.parse_args()
    cmd_verify(args) if args.command == "verify" else cmd_run(args)


if __name__ == "__main__":
    main()

