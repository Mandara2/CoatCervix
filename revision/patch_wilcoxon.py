import os

p = "xai_quant.py"
s = open(p, encoding="utf-8").read()
helper = '''

def safe_wilcoxon(a, b, **kw):
    """Wilcoxon signed-rank; identical paired samples (all differences 0) give p = 1."""
    if np.allclose(a, b):
        return type("W", (), {"statistic": float("nan"), "pvalue": 1.0})()
    return wilcoxon(a, b, **kw)
'''
anchor = 'METRICS = ["aopc_morf", "aopc_lerf", "naopc_morf", "naopc_lerf"]\n'
assert s.count(anchor) == 1 and s.count("res = wilcoxon(") == 2, "unexpected file content"
s = s.replace(anchor, anchor + helper).replace("res = wilcoxon(", "res = safe_wilcoxon(")
with open(p + ".tmp", "w", encoding="utf-8", newline="\n") as f:
    f.write(s)
os.replace(p + ".tmp", p)  # the original is only replaced once the new file is fully written
print("patched")
