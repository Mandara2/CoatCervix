# Builds results/final: IG, SHAP and random from results/main + LayerCAM/Grad-CAM at
# backbone.serial_blocks3.9 (no eigen_smooth) from results/cam_block39. Same NAOPC bounds.
import json
import os
import shutil

import numpy as np

R = "results"
OUT = os.path.join(R, "final")
N = 225
os.makedirs(os.path.join(OUT, "maps"), exist_ok=True)


def rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


keep = [r for r in rows(f"{R}/main/per_image.jsonl") if r["method"] in ("ig", "shap", "random")]
keep += [r for r in rows(f"{R}/cam_block39/per_image.jsonl") if r["method"] in ("layercam", "gradcam")]
unique = {(r["idx"], r["method"]): r for r in keep}
assert len(unique) == N * 5, f"expected {N * 5} rows, got {len(unique)}"
with open(f"{OUT}/per_image.jsonl", "w") as f:
    for r in sorted(unique.values(), key=lambda r: (r["idx"], r["method"])):
        f.write(json.dumps(r) + "\n")

shutil.copy(f"{R}/main/bounds.jsonl", f"{OUT}/bounds.jsonl")
for i in range(N):
    a = np.load(f"{R}/main/maps/{i:04d}.npz")
    b = np.load(f"{R}/cam_block39/maps/{i:04d}.npz")
    np.savez_compressed(f"{OUT}/maps/{i:04d}.npz", ig=a["ig"], shap=a["shap"], random=a["random"],
                        layercam=b["layercam"], gradcam=b["gradcam"])

info = json.load(open(f"{R}/main/run_info.json"))
with open(f"{OUT}/run_info.json", "w") as f:
    json.dump({k: info[k] for k in ("shap_used", "shap_fallback_reason") if k in info}, f, indent=2)
print("ok:", len(unique), "rows,", N, "maps ->", OUT)
