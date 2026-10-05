"""Compare two `swarm benchmark` summary JSONs seed by seed (paired bootstrap CI)."""
import json, sys
import numpy as np

def load(p):
    d = json.load(open(p))
    return {r["seed"]: r for g in d["group_results"].values() for r in g}

a, b = load(sys.argv[1]), load(sys.argv[2])
seeds = sorted(set(a) & set(b))
rng = np.random.default_rng(0)

def ci(diff):
    bs = rng.choice(diff, (10000, len(diff))).mean(1)
    return np.percentile(bs, [2.5, 97.5])

def line(name, ss):
    sa = np.array([a[s]["score"] for s in ss]); sb = np.array([b[s]["score"] for s in ss])
    d = sb - sa; lo, hi = ci(d)
    wa = sum(a[s]["success"] for s in ss); wb = sum(b[s]["success"] for s in ss)
    rec = sum(b[s]["success"] and not a[s]["success"] for s in ss)
    lost = sum(a[s]["success"] and not b[s]["success"] for s in ss)
    print(f"{name:16s} n={len(ss):4d}  A={sa.mean():.4f}  B={sb.mean():.4f}  B-A={d.mean():+.4f} "
          f"IC95=[{lo:+.4f},{hi:+.4f}]  succès {wa}->{wb} (+{rec} récup., -{lost} perdus)")

for grp in sorted({a[s]["group"] for s in seeds}):
    line(grp, [s for s in seeds if a[s]["group"] == grp])
line("TOTAL", seeds)
print("Seeds perdus:", [(s, a[s]["group"], round(a[s]["score"], 3), round(b[s]["score"], 3))
                        for s in seeds if a[s]["success"] and not b[s]["success"]])
