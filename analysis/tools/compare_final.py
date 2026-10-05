"""agent_v5 contre le champion, sur les mêmes seeds (60 premières par type) : écart apparié, IC bootstrap."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

D = Path(__file__).resolve().parent.parent / "data"
run = sys.argv[1] if len(sys.argv) > 1 else "v5_final"
A = {e["seed"]: e for e in map(json.loads, open(D / "champion_full" / "episodes.jsonl"))}
rows = []
for e in map(json.loads, open(D / run / "episodes.jsonl")):
    a = A[e["seed"]]
    rows.append(dict(seed=e["seed"], type=e["env_type"], champ=a["score"], v5=e["score"],
                     fa=a["failure_reason"], fb=e["failure_reason"]))
d = pd.DataFrame(rows); d["delta"] = d.v5 - d.champ
rng = np.random.default_rng(0)

def ci(x, n=5000):
    x = np.asarray(x); b = rng.choice(x, (n, len(x))).mean(1)
    return np.percentile(b, [2.5, 97.5])

out = []
for t, g in list(d.groupby("type")) + [("TOUS", d)]:
    lo, hi = ci(g.delta)
    out.append(dict(type=t, n=len(g), champion=g.champ.mean(), v5=g.v5.mean(), delta=g.delta.mean(), ic_bas=lo, ic_haut=hi,
                    récupérés=int((g.delta > 0.5).sum()), perdus=int((g.delta < -0.5).sum()),
                    autres_plus=int(((g.delta > 0.01) & (g.delta <= 0.5)).sum()), autres_moins=int(((g.delta < -0.01) & (g.delta >= -0.5)).sum())))
pd.set_option("display.width", 200)
print(pd.DataFrame(out).round(4).to_string(index=False))
print("\nseeds perdues :"); print(d[d.delta < -0.5].sort_values("type").to_string(index=False))
print("\nseeds récupérées :"); print(d[d.delta > 0.5].sort_values("type").to_string(index=False))
d.to_csv(Path(__file__).resolve().parent.parent / "results" / f"{run}_compare.csv", index=False)
