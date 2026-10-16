"""
Revision PeerJ: regenerates Figures 2 (GFLOPs) and 3 (parameters) with the values of the approved Table 8.

The submitted figures used ConvNeXt-Base values for ConvNeXt-Large (87.57 M parameters). Same style as the
submitted ones: horizontal bars, CoatCervix highlighted, dashed line at the CoatCervix value. Our
replication of HMCFormer is added because it is in Table 8. Values are typed from Table 8 (ptflops, MACs).

No model or data needed. Writes figura2_gflops_v2.{png,pdf} and figura3_params_v2.{png,pdf} to
~/Cervical/revision_peerj/figuras_eficiencia/.

  /data/rtabares/alzheimer/cervical_clean/bin/python bloque_f_figuras_eficiencia.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.expanduser("~/Cervical/revision_peerj/figuras_eficiencia")
os.makedirs(OUT, exist_ok=True)

# Table 8 (approved 2026-10-05): model -> (GFLOPs, parameters in millions)
TABLE8 = {
    "CoatCervix": (10.41, 44.32),
    "ResNet50": (4.13, 23.51),
    "DenseNet201": (4.34, 18.10),
    "VGG16": (15.52, 134.27),
    "VGG19": (19.68, 139.58),
    "Swin-Base": (15.19, 86.75),
    "ConvNeXt-Large": (34.44, 196.23),
    "EfficientNet-B0": (0.41, 4.01),
    "HMCFormer (replication)": (8.95, 54.55),
}
BAR, HIGHLIGHT = "#c8956a", "#8b5e3c"


def draw(column, xlabel, name):
    items = sorted(TABLE8.items(), key=lambda kv: kv[1][column])
    labels = [k for k, _ in items]
    values = [v[column] for _, v in items]
    ref = TABLE8["CoatCervix"][column]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = [HIGHLIGHT if lab == "CoatCervix" else BAR for lab in labels]
    bars = ax.barh(labels, values, color=colors, height=0.62)
    ax.axvline(ref, color=HIGHLIGHT, linestyle="--", linewidth=0.9, alpha=0.6)
    span = max(values)
    for bar, lab, val in zip(bars, labels, values):
        bold = lab == "CoatCervix"
        ax.text(val + span * 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center",
                fontsize=10, color=HIGHLIGHT if bold else "#444444", fontweight="bold" if bold else "normal")
    for tick in ax.get_yticklabels():
        if tick.get_text() == "CoatCervix":
            tick.set_fontweight("bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_xlim(0, span * 1.15)
    ax.grid(axis="x", color="#e6e5df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


draw(0, "GFLOPs (multiply accumulate operations)", "figura2_gflops_v2")
draw(1, "Parameters (M)", "figura3_params_v2")
print("Done:", OUT)
