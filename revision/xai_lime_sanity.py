"""
Revision PeerJ: model parameter randomization test (Adebayo et al., 2018) for LIME, with exactly the
same procedure as xai_quant.py (run_cascade): first 30 test images, predicted class, cascade from the
head to stage 1, Spearman of |map|. Only LIME is computed; it reuses run_cascade through a small
explainer whose `model` attribute is swapped by run_cascade, as in xai_quant.py.

Does not modify xai_quant.py or its results: writes only to results/lime/sanity_lime.csv.

Run from ~/Cervical/revision_peerj/xai_quant with the environment active:
  CUDA_VISIBLE_DEVICES=<free gpu> nohup python -u xai_lime_sanity.py > lime_sanity.log 2>&1 &
"""
import os

import numpy as np
import pandas as pd
import torch

import xai_quant as xq

SEED = 42
N_IMAGES = 30
OUT = "results/lime"
os.makedirs(OUT, exist_ok=True)

from lime import lime_image  # noqa: E402
from lime.wrappers.scikit_image import SegmentationAlgorithm  # noqa: E402

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


class LimeExplainer:
    """Same interface as xq.Explainer for run_cascade: uses whatever model is in self.model."""

    def __init__(self, model):
        self.model = model
        self.segmenter = SegmentationAlgorithm("slic", n_segments=100, compactness=10, sigma=1, random_seed=SEED)
        self.lime = lime_image.LimeImageExplainer(random_state=SEED)

    @torch.no_grad()
    def _classify(self, images_hwc):
        x = (np.asarray(images_hwc, dtype=np.float32) - mean) / std
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))
        return torch.cat([torch.softmax(self.model(x[i:i + 64].to(device)), 1).cpu()
                          for i in range(0, len(x), 64)]).numpy()

    def explain(self, method, x, target, image_id=0):
        assert method == "lime"
        target = int(target)
        visible = np.clip(x[0].detach().cpu().numpy().transpose(1, 2, 0) * std + mean, 0, 1)
        exp = self.lime.explain_instance(visible.astype(np.float64), self._classify, labels=(target,),
                                         top_labels=None, hide_color=0, num_samples=1000, num_features=100000,
                                         segmentation_fn=self.segmenter, random_seed=SEED)
        out = np.zeros(exp.segments.shape, dtype=np.float32)
        for seg, w in dict(exp.local_exp[target]).items():
            out[exp.segments == seg] = w
        return out


print(f"Sanity check of LIME on {N_IMAGES} images...", flush=True)
rows, _ = xq.run_cascade(model, LimeExplainer(model), X_test_t[:N_IMAGES], preds[:N_IMAGES], ["lime"], seed=SEED)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "sanity_lime.csv"), index=False)

order = [name for name, _, _ in xq.cascade_groups(model)]
table = df.groupby("randomized_up_to")["spearman_abs"].mean().reindex(order).rename("lime").to_frame()
final_sanity = "results/final/sanity.csv"
if os.path.exists(final_sanity):
    other = pd.read_csv(final_sanity).groupby(["randomized_up_to", "method"])["spearman_abs"].mean().unstack()
    table = table.join(other.reindex(order))
print("\nSpearman of |map| with the original map, mean over the images (randomized up to ...):")
print(table.round(3).to_string())

