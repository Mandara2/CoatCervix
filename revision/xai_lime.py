"""
Revision PeerJ: evaluates LIME with the same faithfulness protocol as xai_quant.py (NAOPC, MoRF/LeRF).

Does not modify xai_quant.py or its results: it imports its functions and writes only to results/lime/.
  - LIME with the configuration of the paper (Figure 8): SLIC 100/10/1, 1000 samples, hidden superpixels
    in black, explaining the predicted class. Map = weight of each superpixel for the predicted class
    (all superpixels, signed), painted over its pixels.
  - Same 7 x 7 grid, same baseline (mean of the training set) and same probability curves as xai_quant.py.
  - NAOPC with the bounds of results/final/bounds.jsonl, so no beam search is repeated.
  - Resumable: one line per image in results/lime/per_image.jsonl.

Run from ~/Cervical/revision_peerj/xai_quant with the environment active:
  CUDA_VISIBLE_DEVICES=<free gpu> nohup python -u xai_lime.py > lime.log 2>&1 &
  python xai_lime.py --summary-only        # table again, without recomputing
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

import xai_quant as xq

FINAL = "results/final"
OUT = "results/lime"
SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument("--n-images", type=int, default=None)
parser.add_argument("--summary-only", action="store_true")
args = parser.parse_args()
os.makedirs(os.path.join(OUT, "maps"), exist_ok=True)
per_image_path = os.path.join(OUT, "per_image.jsonl")


def lime_map(explainer, segmenter, visible, classify, target):
    exp = explainer.explain_instance(visible.astype(np.float64), classify, labels=(target,), top_labels=None,
                                     hide_color=0, num_samples=1000, num_features=100000,
                                     segmentation_fn=segmenter, random_seed=SEED)
    weights = dict(exp.local_exp[target])
    out = np.zeros(exp.segments.shape, dtype=np.float32)
    for seg, w in weights.items():
        out[exp.segments == seg] = w
    return out


if not args.summary_only:
    from lime import lime_image
    from lime.wrappers.scikit_image import SegmentationAlgorithm

    device = xq.settings.device
    xq.set_seed(SEED)
    X_train, X_test_t, y_test, mean, std = xq.load_test_data(xq.DATA_PATH)
    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, -1)
    model = xq.load_model(xq.FINAL_WEIGHTS, device)
    model.eval()
    preds = xq.predict_probs(model, X_test_t, device).argmax(dim=1).numpy()
    acc = float((preds == np.asarray(y_test)).mean())
    print(f"test accuracy {acc:.4f} (paper 0.8133)", flush=True)
    if abs(acc - 0.8133) > 1e-4:
        raise SystemExit("The model does not reproduce the paper: stopping.")

    @torch.no_grad()
    def classify(images_hwc):
        x = (np.asarray(images_hwc, dtype=np.float32) - mean) / std
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))
        return torch.cat([torch.softmax(model(x[i:i + 64].to(device)), 1).cpu()
                          for i in range(0, len(x), 64)]).numpy()

    segmenter = SegmentationAlgorithm("slic", n_segments=100, compactness=10, sigma=1, random_seed=SEED)
    explainer = lime_image.LimeImageExplainer(random_state=SEED)
    masks = xq.patch_masks(7, *X_test_t.shape[-2:], device=device)
    done = {r["idx"] for r in xq.read_jsonl(per_image_path)}
    n = len(X_test_t) if args.n_images is None else min(args.n_images, len(X_test_t))
    for idx in range(n):
        if idx in done:
            continue
        t0 = time.time()
        x = X_test_t[idx:idx + 1].to(device).float()
        target = int(preds[idx])
        visible = np.clip(X_test_t[idx].numpy().transpose(1, 2, 0) * std + mean, 0, 1)
        m = lime_map(explainer, segmenter, visible, classify, target)
        np.savez_compressed(os.path.join(OUT, "maps", f"{idx:04d}.npz"), lime=m.astype(np.float16))
        perturber = xq.PatchPerturber(model, x, xq.make_baseline(x, "mean"), target, masks, batch_size=64)
        morf, lerf = xq.morf_lerf_curves(perturber, m, 7)
        xq.append_jsonl(per_image_path, {"idx": idx, "label": int(y_test[idx]), "pred": int(preds[idx]),
                                         "target": target, "method": "lime", "p0": float(morf[0]),
                                         "aopc_morf": xq.aopc(morf), "aopc_lerf": xq.aopc(lerf),
                                         "morf": morf.tolist(), "lerf": lerf.tolist()})
        print(f"[{idx + 1}/{n}] class {target} | {time.time() - t0:.1f}s", flush=True)

# ---- summary: LIME next to the methods of results/final, with the same bounds
cols = ["idx", "label", "target", "method", "aopc_morf", "aopc_lerf"]
final = pd.DataFrame(xq.read_jsonl(os.path.join(FINAL, "per_image.jsonl")))[cols]
lime = pd.DataFrame(xq.read_jsonl(per_image_path))[cols]
bounds = pd.DataFrame(xq.read_jsonl(os.path.join(FINAL, "bounds.jsonl")))
ids = sorted(set(lime["idx"]))
both = pd.concat([final[final["idx"].isin(ids)], lime], ignore_index=True)
df, info = xq.add_naopc(both, bounds)
print(f"\nimages: {len(ids)} | bounds widened: {info}")


def ci(v, n_boot=10000):
    v = np.asarray(v, dtype=float)
    means = np.random.default_rng(SEED).choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


rows, tests = [], []
for method, g in df.groupby("method"):
    row = {"method": method}
    for metric in ("naopc_morf", "naopc_lerf"):
        lo, hi = ci(g[metric])
        row.update({metric: g[metric].mean(), metric + "_lo": lo, metric + "_hi": hi})
    rows.append(row)
    if method != "random":
        ref = df[df["method"] == "random"].set_index("idx")
        cur = g.set_index("idx")
        for metric, alt in (("naopc_morf", "greater"), ("naopc_lerf", "less")):
            d = cur[metric] - ref.loc[cur.index, metric]
            p = 1.0 if np.allclose(d, 0) else wilcoxon(d, alternative=alt).pvalue
            tests.append({"method": method, "metric": metric, "p": p})
tests = pd.DataFrame(tests).sort_values("p").reset_index(drop=True)
k = len(tests)
tests["p_holm"] = np.minimum(1, np.maximum.accumulate([(k - i) * p for i, p in enumerate(tests["p"])]))
summary = pd.DataFrame(rows).round(3)
print("\nNAOPC (mean and 95% CI; MoRF higher is better, LeRF lower is better):")
print(summary.to_string(index=False))
print("\nWilcoxon against the random map (one sided), Holm over these tests:")
print(tests.to_string(index=False))

if len(ids) == 225:
    by_class = df[df["method"].isin(["lime", "random", "shap", "gradcam"])].groupby(["method", "label"])[
        ["naopc_morf", "naopc_lerf"]].mean().round(3)
    print("\nBy true class:")
    print(by_class.to_string())
summary.to_csv(os.path.join(OUT, "summary_with_lime.csv"), index=False)
tests.to_csv(os.path.join(OUT, "tests_vs_random.csv"), index=False)
json.dump(info, open(os.path.join(OUT, "bounds_info.json"), "w"))

