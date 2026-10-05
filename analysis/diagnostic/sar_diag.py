"""Loading, derived metrics and statistics for the SAR diagnostic notebook.

Reads the output of `analysis/collect_rollouts.py`:

  <run>/episodes.jsonl            one record per seed
  <run>/traj/<group>_<seed>.npz   per-step trajectory (+ top-down height map)
  <run>/rgb/<group>_<seed>.npz    optional probe RGB frames

and optionally a `swarm benchmark --summary-json-out` file. Only numpy, pandas and
matplotlib are required (no scipy): the few tests used are implemented here.
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants mirrored from swarm/constants.py and the SAR spec
# --------------------------------------------------------------------------- #
HORIZON = 60.0
SIM_DT = 0.02
SPEED_LIMIT = 3.0
SEARCH_R = 30.0
DWELL = 2.0
W_SUCCESS, W_TIME, W_SAFETY = 0.45, 0.45, 0.10
PARTICIPATION = 0.01
SAFE_BY_TYPE = {6: 0.6}
SAFE_DEFAULT, DANGER = 1.0, 0.2
ACT_BUDGET_MS = 600.0
FIRST_STEP_BUDGET_MS = 2000.0
RGB_BUDGET = 40

TYPE_ORDER = ["city", "open", "mountain", "village", "warehouse", "forest"]
GROUP_TO_TYPE = {
    "type1_city": "city", "type2_open": "open", "type3_mountain": "mountain",
    "type4_village": "village", "type5_warehouse": "warehouse", "type6_forest": "forest",
}
# Reference categorical palette (fixed order, never cycled): one hue per env type.
TYPE_COLORS = dict(zip(TYPE_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))

# Derived outcome taxonomy, ordered from "best" to "worst" understood.
OUTCOME_ORDER = [
    "success_fast",       # confirmed within target time
    "success_slow",       # confirmed, time term < 1
    "success_voided",     # confirmed but a collision was on record -> 0.01
    "crash",              # OBSTACLE_COLLISION / TILT
    "no_touch",           # entered the 0.8 m sphere
    "infeasible",         # truncated early: could not reach the circle in time
    "to_never_seen",      # timeout, victim never geometrically visible
    "to_seen_not_close",  # timeout, victim seen, never within 2 m horizontally
    "to_close_no_pred",   # timeout, came within 2 m but never satisfied the predicate
    "to_pred_unstable",   # timeout, predicate reached but dwell kept resetting
    "other_fail",
    "error",
]
OUTCOME_COLORS = {
    "success_fast": "#1baf7a", "success_slow": "#86b6ef", "success_voided": "#4a3aa7",
    "crash": "#e34948", "no_touch": "#e87ba4", "infeasible": "#eda100",
    "to_never_seen": "#52514e", "to_seen_not_close": "#eb6834",
    "to_close_no_pred": "#2a78d6", "to_pred_unstable": "#0d366b",
    "other_fail": "#b0aea5", "error": "#000000",
}
OUTCOME_LABELS = {
    "success_fast": "succès dans le temps cible",
    "success_slow": "succès lent (time<1)",
    "success_voided": "succès annulé (collision)",
    "crash": "crash (collision/tilt)",
    "no_touch": "sphère no-touch",
    "infeasible": "infaisable tôt (loin du cercle)",
    "to_never_seen": "temps écoulé : victime jamais visible",
    "to_seen_not_close": "temps écoulé : vue mais jamais < 2 m",
    "to_close_no_pred": "temps écoulé : < 2 m sans prédicat",
    "to_pred_unstable": "temps écoulé : prédicat instable",
    "other_fail": "autre échec",
    "error": "erreur d'éval",
}


# --------------------------------------------------------------------------- #
# Scoring helpers (re-implemented to decompose the loss)
# --------------------------------------------------------------------------- #
def target_time(d_start_centre: float, horizon: float = HORIZON) -> float:
    sweep = 0.70 * math.pi * SEARCH_R ** 2 / (24.0 * SPEED_LIMIT)
    return min(1.03 * (d_start_centre / SPEED_LIMIT + sweep + DWELL), horizon * 0.95)


def time_term(t: float, tt: float, horizon: float = HORIZON) -> float:
    if t <= tt:
        return 1.0
    return float(np.clip(1.0 - (t - tt) / (horizon - tt), 0.0, 1.0))


def safety_term(min_clearance: Optional[float], ctype: int) -> float:
    if min_clearance is None or not np.isfinite(min_clearance):
        return 1.0
    safe = SAFE_BY_TYPE.get(int(ctype), SAFE_DEFAULT)
    if min_clearance >= safe:
        return 1.0
    if min_clearance <= DANGER:
        return 0.0
    return (min_clearance - DANGER) / (safe - DANGER)


# --------------------------------------------------------------------------- #
# Statistics (no scipy)
# --------------------------------------------------------------------------- #
def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float, float]:
    if n <= 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, min(p, max(0.0, c - h)), max(p, min(1.0, c + h))


def bootstrap_mean_ci(x: Iterable[float], n_boot: int = 4000, alpha: float = 0.05,
                      seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(list(x), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    m = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def stratified_bootstrap_mean(df: pd.DataFrame, col: str, strata: str = "env_type",
                              n_boot: int = 4000, seed: int = 0) -> np.ndarray:
    """Bootstrap distribution of the overall mean, resampling within each stratum
    (the benchmark has fixed per-type proportions)."""
    rng = np.random.default_rng(seed)
    parts = [g[col].to_numpy(float) for _, g in df.groupby(strata)]
    tot = sum(len(p) for p in parts)
    out = np.zeros(n_boot)
    for p in parts:
        idx = rng.integers(0, len(p), size=(n_boot, len(p)))
        out += p[idx].sum(axis=1)
    return out / tot


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant pairs b (A only) and c (B only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, 2 * p))


def sign_flip_pvalue(d: np.ndarray, n_perm: int = 20000, seed: int = 0) -> float:
    """Paired permutation test (random sign flips) on per-seed differences."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if d.size == 0 or np.allclose(d, 0):
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    s = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    null = np.abs((s * d).mean(axis=1))
    return float((1 + np.sum(null >= obs - 1e-12)) / (n_perm + 1))


def binned_rate(x: pd.Series, y: pd.Series, bins) -> pd.DataFrame:
    """Rate of a boolean y per bin of x, with Wilson 95% CI."""
    cols = ["bin", "mid", "n", "rate", "lo", "hi"]
    bins = np.unique(np.asarray(bins, float)[np.isfinite(bins)]) if not np.isscalar(bins) else bins
    if not np.isscalar(bins) and len(bins) < 2:
        return pd.DataFrame(columns=cols, dtype=float)
    cut = pd.cut(x, bins)
    rows = []
    for iv, g in y.groupby(cut, observed=False):
        k, n = float(g.sum()), float(g.count())
        p, lo, hi = wilson(k, n)
        rows.append({"bin": iv, "mid": iv.mid if hasattr(iv, "mid") else np.nan,
                     "n": int(n), "rate": p, "lo": lo, "hi": hi})
    return pd.DataFrame(rows, columns=cols).dropna(subset=["rate"])


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_episodes(run_dir: Path | str) -> pd.DataFrame:
    run_dir = Path(run_dir)
    recs = []
    with open(run_dir / "episodes.jsonl", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    # resumed runs may contain duplicates / failed attempts: keep the last ok record
    df = pd.DataFrame(recs)
    if "ok" not in df:
        df["ok"] = True
    df["_ord"] = np.arange(len(df))
    df = (df.sort_values(["ok", "_ord"]).drop_duplicates(["group", "seed"], keep="last")
            .sort_values("_ord").drop(columns="_ord").reset_index(drop=True))
    return _derive(df)


def _col(df, name, default):
    return df[name] if name in df else pd.Series(default, index=df.index)


def _vec(df, col, i):
    return df[col].map(lambda v: v[i] if isinstance(v, (list, tuple)) and len(v) > i else np.nan)


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "env_type" not in df or df["env_type"].isna().any():
        df["env_type"] = df.get("env_type", pd.Series(index=df.index, dtype=object)).fillna(
            df["group"].map(GROUP_TO_TYPE))
    df["env_type"] = pd.Categorical(df["env_type"], TYPE_ORDER, ordered=True)
    for c, default in [("success", False), ("collision", False), ("score", 0.0),
                       ("failure_reason", "NONE"), ("sim_time", np.nan)]:
        if c not in df:
            df[c] = default
    df["success"] = df["success"].fillna(False).astype(bool)
    df["collision"] = df["collision"].fillna(False).astype(bool)

    # Score decomposition: the maximum is 1.0, every seed loses (1 - score) points.
    if "target_time" not in df and "d_start_centre" in df:
        df["target_time"] = df["d_start_centre"].map(target_time)
    tt = _col(df, "target_time", np.nan)
    t_conf = df["t_confirm"] if "t_confirm" in df else df["sim_time"]
    t_conf = t_conf.fillna(df["sim_time"])
    df["t_confirm_eff"] = np.where(df["success"], t_conf, np.nan)
    if "time_term" not in df:
        df["time_term"] = [time_term(t, T) if s and np.isfinite(T) else 0.0
                           for s, t, T in zip(df["success"], t_conf, tt)]
    if "safety_term" not in df:
        df["safety_term"] = [safety_term(mc, ct) if s else 0.0 for s, mc, ct in
                             zip(df["success"], _col(df, "min_clearance", np.nan),
                                 _col(df, "challenge_type", 0))]
    good = df["success"] & ~df["collision"]
    df["loss_total"] = 1.0 - df["score"]
    df["loss_fail"] = np.where(good, 0.0, 1.0 - df["score"])
    df["loss_time"] = np.where(good, W_TIME * (1 - df["time_term"]), 0.0)
    df["loss_safety"] = np.where(good, W_SAFETY * (1 - df["safety_term"]), 0.0)
    df["excess_time"] = np.where(good, df["t_confirm_eff"] - tt, np.nan)

    # Geometry
    for col, name in [("victim_centre", "victim"), ("search_centre", "centre"), ("start", "start")]:
        if col in df:
            df[f"{name}_x"] = _vec(df, col, 0)
            df[f"{name}_y"] = _vec(df, col, 1)
    if "victim_extent" in df:
        df["victim_height"] = _vec(df, "victim_extent", 2)
    if "victim_centre" in df and "start" in df:
        df["victim_dz"] = _vec(df, "victim_centre", 2) - _vec(df, "start", 2)
    if {"victim_x", "centre_x", "start_x"} <= set(df.columns):
        # where is the victim relative to the start -> centre axis?
        ax = np.stack([df["centre_x"] - df["start_x"], df["centre_y"] - df["start_y"]], 1)
        vv = np.stack([df["victim_x"] - df["centre_x"], df["victim_y"] - df["centre_y"]], 1)
        n = np.linalg.norm(ax, axis=1, keepdims=True)
        axu = ax / np.where(n > 1e-6, n, 1)
        df["victim_along"] = (vv * axu).sum(1)       # >0 : beyond the centre
        df["victim_across"] = vv[:, 0] * axu[:, 1] - vv[:, 1] * axu[:, 0]
        df["victim_bearing_deg"] = np.degrees(np.arctan2(df["victim_across"], df["victim_along"]))
    df["outcome"] = df.apply(_classify, axis=1)
    df["outcome"] = pd.Categorical(df["outcome"], OUTCOME_ORDER, ordered=True)
    df["fail"] = ~(df["success"] & ~df["collision"])
    return df


def _classify(r) -> str:
    if not r.get("ok", True) or r.get("failure_reason") == "EVAL_ERROR":
        return "error"
    if r["success"]:
        if r["collision"]:
            return "success_voided"
        return "success_fast" if r.get("time_term", 0) >= 0.999 else "success_slow"
    fr = str(r.get("failure_reason", ""))
    if fr in ("OBSTACLE_COLLISION", "TILT"):
        return "crash"
    if fr == "NO_TOUCH_SPHERE":
        return "no_touch"
    if fr == "INFEASIBLE":
        # infeasible() fires as soon as the remaining time cannot cover the 2 s dwell, i.e. at
        # t ~ 58 s for a drone inside the circle: that is the de facto timeout. Only an early
        # truncation (drone far outside the circle) is a distinct class.
        if float(r.get("sim_time", 0.0) or 0.0) < float(r.get("horizon", 60.0) or 60.0) - 5.0:
            return "infeasible"
        fr = "TIMEOUT"
    if fr == "TIMEOUT":
        fv = r.get("first_visible_t")
        mh = r.get("sar_min_hdist", np.nan)
        md = r.get("sar_max_dwell", 0.0) or 0.0
        if fv is None or (isinstance(fv, float) and not np.isfinite(fv)):
            return "to_never_seen"
        if md > 0:
            return "to_pred_unstable"
        if np.isfinite(mh) and mh <= 2.0:
            return "to_close_no_pred"
        return "to_seen_not_close"
    return "other_fail"


def load_benchmark(path: Path | str) -> pd.DataFrame:
    """Rows of a `swarm benchmark --summary-json-out` file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [dict(r) for g in payload.get("group_results", {}).values() for r in g]
    df = pd.DataFrame(rows)
    if len(df):
        df["env_type"] = pd.Categorical(df["group"].map(GROUP_TO_TYPE), TYPE_ORDER, ordered=True)
    return df


# --------------------------------------------------------------------------- #
# Trajectories
# --------------------------------------------------------------------------- #
def traj_path(run_dir: Path | str, group: str, seed: int) -> Path:
    return Path(run_dir) / "traj" / f"{group}_{int(seed)}.npz"


def load_traj(run_dir: Path | str, group: str, seed: int) -> Optional[dict]:
    p = traj_path(run_dir, group, seed)
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    if "t" not in d or d["t"].size == 0:
        return None
    d["speed"] = np.sqrt(d["vx"] ** 2 + d["vy"] ** 2 + d["vz"] ** 2)
    d["hspeed"] = np.sqrt(d["vx"] ** 2 + d["vy"] ** 2)
    return d


def decode_codes(table: dict | None) -> dict[int, str]:
    if not isinstance(table, dict):
        return {}
    return {int(v): str(k) for k, v in table.items()}


def _search_grid(cx: float, cy: float, step: float = 2.0) -> np.ndarray:
    xs = np.arange(-SEARCH_R, SEARCH_R + 1e-6, step)
    X, Y = np.meshgrid(xs, xs)
    m = X ** 2 + Y ** 2 <= SEARCH_R ** 2
    return np.stack([X[m] + cx, Y[m] + cy], 1)


def traj_features(rec: pd.Series, tr: dict, cover_r: float = 10.0) -> dict:
    """Per-episode features computed from the per-step trajectory."""
    t = tr["t"].astype(float)
    x, y = tr["x"].astype(float), tr["y"].astype(float)
    f: dict = {}
    n = t.size
    f["n_steps"] = n
    hd = tr["hdist"].astype(float)
    ha = tr["h_above"].astype(float)
    sp = tr["speed"].astype(float)
    agl = tr["agl"].astype(float)
    pred = tr["pred"].astype(int)
    dwell = tr["dwell"].astype(float)
    vis = tr["vis"].astype(int)

    def first(mask):
        i = np.flatnonzero(mask)
        return float(t[i[0]]) if i.size else np.nan

    cx, cy = rec.get("centre_x", np.nan), rec.get("centre_y", np.nan)
    if np.isfinite(cx):
        dc = np.hypot(x - cx, y - cy)
        f["t_reach_circle"] = first(dc <= SEARCH_R)
        f["t_reach_centre10"] = first(dc <= 10.0)
        f["min_d_centre"] = float(dc.min())
    f["t_first_vis"] = first(vis == 1)
    checked = vis >= 0
    f["vis_frac"] = float((vis[checked] == 1).mean()) if checked.any() else np.nan
    f["vis_time"] = float((vis == 1).sum() * 5 * SIM_DT)  # approx: checked every 5 steps
    f["t_first_20m"] = first(hd <= 20.0)
    f["t_first_10m"] = first(hd <= 10.0)
    f["t_first_6m"] = first(hd <= 6.0)
    f["t_first_2m"] = first(hd <= 2.0)
    f["t_first_pred"] = first(pred == 1)
    f["pred_time"] = float(pred.sum() * SIM_DT)
    # dwell resets: dwell drops to 0 after being > 0
    drops = (dwell[:-1] > 0) & (dwell[1:] <= 0) if n > 1 else np.array([], bool)
    f["n_dwell_resets"] = int(drops.sum())
    f["max_dwell_traj"] = float(dwell.max()) if n else 0.0
    f["min_hdist_traj"] = float(np.nanmin(hd)) if np.isfinite(hd).any() else np.nan
    f["min_d3_traj"] = float(np.nanmin(tr["d3"])) if np.isfinite(tr["d3"]).any() else np.nan
    f["final_hdist"] = float(hd[-1])
    f["final_h_above"] = float(ha[-1])
    # hover quality when horizontally on target
    near = hd <= 2.0
    f["near_time"] = float(near.sum() * SIM_DT)
    if near.any():
        f["near_band_frac"] = float(((ha[near] >= 2) & (ha[near] <= 4)).mean())
        f["near_slow_frac"] = float((sp[near] <= 1.0).mean())
        f["near_h_above_med"] = float(np.median(ha[near]))
        f["near_speed_med"] = float(np.median(sp[near]))
    # which predicate condition fails most while close
    if near.any():
        f["near_fail_low"] = float((ha[near] < 2).mean())
        f["near_fail_high"] = float((ha[near] > 4).mean())
        f["near_fail_fast"] = float((sp[near] > 1).mean())
    # kinematics
    f["mean_speed"] = float(sp.mean())
    f["mean_hspeed"] = float(tr["hspeed"].mean())
    f["frac_fast"] = float((sp >= 2.5).mean())
    f["frac_slow"] = float((sp <= 0.3).mean())
    f["agl_med"] = float(np.nanmedian(agl))
    f["frac_agl_lt1"] = float((agl < 1.0).mean())
    f["frac_agl_ge19"] = float((agl >= 19.0).mean())
    f["z_range"] = float(tr["z"].max() - tr["z"].min())
    f["mean_cmd_speed"] = float(tr["a_speed"].mean())
    f["frac_cmd_zero"] = float((tr["a_speed"] <= 0.02).mean())
    f["yaw_travel_rad"] = float(np.abs(np.diff(np.unwrap(tr["yaw"]))).sum()) if n > 1 else 0.0
    f["dmin_c_min"] = float(np.nanmin(tr["depth_min_c"]))
    f["frac_dmin_lt2"] = float((tr["depth_min_c"] < 2.0).mean())
    # stagnation: longest window with < 2 m of net displacement
    step = max(1, int(round(1.0 / SIM_DT)))
    xs, ys = x[::step], y[::step]
    longest = 0
    for w in (5, 10, 15, 20):
        if xs.size > w:
            disp = np.hypot(xs[w:] - xs[:-w], ys[w:] - ys[:-w])
            if (disp < 2.0).any():
                longest = w
    f["stuck_s"] = float(longest)
    # search-disk coverage (proxy: grid point within cover_r of the path)
    if np.isfinite(cx):
        g = _search_grid(cx, cy)
        p = np.stack([x[::10], y[::10]], 1)
        d = np.sqrt(((g[:, None, :] - p[None, :, :]) ** 2).sum(-1))
        hit = d <= cover_r
        f["coverage"] = float(hit.any(1).mean())
        first_idx = np.where(hit.any(1), hit.argmax(1), np.iinfo(np.int32).max)
        tt10 = t[::10]
        for T in (10, 20, 30, 45, 60):
            k = np.searchsorted(tt10, T, side="right")
            f[f"coverage_{T}s"] = float((first_idx < k).mean())
    # route & escape usage
    codes = decode_codes(rec.get("route_codes"))
    r = tr["route"].astype(int)
    labels = np.array([codes.get(int(c), "none") for c in r])
    for lab in ("king", "king+tour", "king+rgb"):
        base = np.char.replace(labels.astype(str), "+av", "")
        f[f"route_{lab}"] = float((base == lab).mean())
    f["route_av"] = float(np.char.endswith(labels.astype(str), "+av").mean())
    f["t_tour_on"] = first(np.char.find(labels.astype(str), "tour") >= 0)
    f["t_rgb_route"] = first(np.char.find(labels.astype(str), "rgb") >= 0)
    ecodes = decode_codes(rec.get("esc_codes"))
    e = tr["esc"].astype(int)
    elab = np.array([ecodes.get(int(c), "none") for c in e]).astype(str)
    f["esc_frac"] = float(np.isin(elab, ["scan", "creep", "climb"]).mean())
    f["rgb_served_traj"] = int(tr["rgb_served"].sum())
    f["t_first_rgb"] = first(tr["rgb_served"] == 1)
    f["act_ms_p99"] = float(np.percentile(tr["act_ms"], 99))
    f["path_len_traj"] = float(np.hypot(np.diff(x), np.diff(y)).sum())
    return f


def trajectory_table(run_dir: Path | str, ep: pd.DataFrame, cache: bool = True,
                     verbose: bool = True) -> pd.DataFrame:
    """Per-episode trajectory features for every episode that has a .npz (cached)."""
    run_dir = Path(run_dir)
    cache_p = run_dir / "_traj_features.pkl"
    key = sorted((str(g), int(s)) for g, s in zip(ep["group"], ep["seed"]))
    if cache and cache_p.exists():
        try:
            with open(cache_p, "rb") as fh:
                ck, tab = pickle.load(fh)
            if ck == key:
                return tab
        except Exception:
            pass
    rows = []
    for i, (_, r) in enumerate(ep.iterrows()):
        tr = load_traj(run_dir, r["group"], r["seed"])
        if tr is None:
            continue
        f = traj_features(r, tr)
        f.update(group=r["group"], seed=int(r["seed"]))
        rows.append(f)
        if verbose and (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(ep)} trajectories")
    tab = pd.DataFrame(rows)
    if cache and len(tab):
        with open(cache_p, "wb") as fh:
            pickle.dump((key, tab), fh)
    return tab


def series_bundle(run_dir, ep: pd.DataFrame, cols: Iterable[str], grid: np.ndarray,
                  align: Optional[str] = None) -> dict[str, pd.DataFrame]:
    """{col: DataFrame (episodes x grid)} of trajectory columns on a common time grid,
    loading each .npz once. Outside the episode, values are NaN. With `align`, time is
    shifted so that 0 = the value of that episode column (e.g. "t_confirm_eff")."""
    cols = list(cols)
    out = {c: np.full((len(ep), grid.size), np.nan, np.float32) for c in cols}
    for i, (_, r) in enumerate(ep.iterrows()):
        tr = load_traj(run_dir, r["group"], r["seed"])
        if tr is None:
            continue
        t = tr["t"].astype(float)
        if align is not None:
            t0 = r.get(align, np.nan)
            if t0 is None or not np.isfinite(t0):
                continue
            t = t - float(t0)
        m = (grid >= t[0]) & (grid <= t[-1])
        for c in cols:
            if c in tr:
                out[c][i, m] = np.interp(grid[m], t, tr[c].astype(float))
    return {c: pd.DataFrame(v, index=ep.index, columns=grid) for c, v in out.items()}


def events_table(ep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in ep.iterrows():
        for e in r.get("events") or []:
            rows.append({"group": r["group"], "seed": r["seed"], "env_type": r["env_type"],
                         "success": r["success"], **e})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# A/B helpers
# --------------------------------------------------------------------------- #
def paired(a: pd.DataFrame, b: pd.DataFrame, cols=("score", "success", "sim_time", "outcome")) -> pd.DataFrame:
    ka = a[["group", "seed", "env_type", *cols]]
    kb = b[["group", "seed", *cols]]
    m = ka.merge(kb, on=["group", "seed"], suffixes=("_a", "_b"))
    m["d_score"] = m["score_b"] - m["score_a"]
    return m


def ab_summary(m: pd.DataFrame, n_boot: int = 4000) -> pd.DataFrame:
    rows = []
    for name, g in [("ALL", m)] + [(t, m[m["env_type"] == t]) for t in TYPE_ORDER]:
        if not len(g):
            continue
        d = g["d_score"].to_numpy(float)
        mean, lo, hi = bootstrap_mean_ci(d, n_boot=n_boot)
        b = int((g["success_a"] & ~g["success_b"]).sum())
        c = int((~g["success_a"] & g["success_b"]).sum())
        rows.append({"subset": name, "n": len(g),
                     "score_A": g["score_a"].mean(), "score_B": g["score_b"].mean(),
                     "delta": mean, "ci_lo": lo, "ci_hi": hi,
                     "p_perm": sign_flip_pvalue(d, n_perm=5000),
                     "succ_A": g["success_a"].mean(), "succ_B": g["success_b"].mean(),
                     "lost(A ok,B ko)": b, "won(A ko,B ok)": c, "p_mcnemar": mcnemar_exact(b, c)})
    return pd.DataFrame(rows).set_index("subset")
