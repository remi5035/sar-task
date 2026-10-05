"""Données FACTICES au format de collect_rollouts.py, pour tester le notebook sans simulateur.

Usage : python make_fake_data.py ../fake   puis DATA = HERE.parent / "fake" dans le notebook.
Ne jamais interpréter les résultats obtenus sur ces données."""
import json, math, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sar_diag as sd

TRAJ_COLS = ("t", "x", "y", "z", "vx", "vy", "vz", "roll", "pitch", "yaw",
             "a_dx", "a_dy", "a_dz", "a_speed", "a_yaw", "a_rgb",
             "agl", "hdist", "h_above", "d3", "pred", "dwell",
             "vis", "vis_u", "vis_v", "vis_px", "rgb_served",
             "depth_min_c", "clue_dx", "clue_dy", "act_ms", "route", "esc")
INT = {"pred", "vis", "rgb_served", "route", "esc"}
GROUPS = list(sd.GROUP_TO_TYPE)
FAIL_P = {"city": .08, "open": .03, "mountain": .2, "village": .1, "warehouse": .12, "forest": .15}


def episode(out, group, seed, bonus):
    rng = np.random.default_rng(seed)
    et = sd.GROUP_TO_TYPE[group]; ct = GROUPS.index(group) + 1
    start = np.array([0., 0., 0.2])
    d_sc = rng.uniform(15, 50)
    ang = rng.uniform(0, 2 * np.pi)
    sc = np.array([d_sc * math.cos(ang), d_sc * math.sin(ang)])
    r = 30 * math.sqrt(rng.uniform()); a2 = rng.uniform(0, 2 * np.pi)
    vc = np.array([sc[0] + r * math.cos(a2), sc[1] + r * math.sin(a2), rng.uniform(-3, 5)])
    ext = np.array([0.5, 1.7, 0.4]) if rng.uniform() < .5 else np.array([0.5, 0.4, 1.7])
    fail = rng.uniform() < FAIL_P[et] - bonus
    mode = rng.choice(["TIMEOUT", "TIMEOUT", "TIMEOUT", "OBSTACLE_COLLISION", "NO_TOUCH_SPHERE"]) if fail else None
    t_find = rng.uniform(15, 45) if not fail else 60.0
    if mode in ("OBSTACLE_COLLISION", "NO_TOUCH_SPHERE"):
        t_find = rng.uniform(10, 50)
    n = int(t_find / 0.02)
    t = np.arange(1, n + 1) * 0.02
    # path: go to centre then spiral, finally converge on victim
    wp = [start[:2], sc] + [sc + 20 * np.array([math.cos(k), math.sin(k)]) * (k / 12) for k in np.linspace(0, 12, 12)]
    wp = np.array(wp)
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1); cum = np.r_[0, np.cumsum(seg)]
    s = np.minimum(t * 2.6, cum[-1])
    xy = np.stack([np.interp(s, cum, wp[:, 0]), np.interp(s, cum, wp[:, 1])], 1)
    conv = np.clip((t - (t_find - 8)) / 6, 0, 1)[:, None] if not fail or mode == "NO_TOUCH_SPHERE" else 0
    xy = xy * (1 - conv) + vc[:2] * conv + rng.normal(0, .05, xy.shape)
    z = vc[2] + ext[2] + 3 + 3 * (1 - (conv if np.ndim(conv) else 0 * t[:, None]))[:, 0] + rng.normal(0, .05, n)
    v = np.gradient(np.c_[xy, z], 0.02, axis=0)
    hd = np.linalg.norm(xy - vc[:2], axis=1)
    top = vc[2] + ext[2] / 2
    ha = z - top
    d3 = np.linalg.norm(np.c_[xy, z] - vc, axis=1)
    sp = np.linalg.norm(v, axis=1)
    pred = ((hd <= 2) & (ha >= 2) & (ha <= 4) & (sp <= 1)).astype(int)
    dwell = np.zeros(n)
    for i in range(1, n):
        dwell[i] = dwell[i - 1] + 0.02 if pred[i] else 0
    vis = np.full(n, -1); vis[4::5] = (hd[4::5] < rng.uniform(8, 20)).astype(int)
    route = np.where(t > 8, 1, 0); route[(hd < 6) & (t > 8)] = 2
    esc = np.where(t < 1.5, 0, 1) if rng.uniform() < .1 else np.zeros(n, int) - 1
    cols = dict(t=t, x=xy[:, 0], y=xy[:, 1], z=z, vx=v[:, 0], vy=v[:, 1], vz=v[:, 2],
                roll=rng.normal(0, .05, n), pitch=rng.normal(0, .05, n), yaw=np.cumsum(rng.normal(0, .02, n)),
                a_dx=rng.uniform(-1, 1, n), a_dy=rng.uniform(-1, 1, n), a_dz=rng.normal(0, .1, n),
                a_speed=np.clip(sp / 3, 0, 1), a_yaw=rng.uniform(-1, 1, n), a_rgb=np.zeros(n),
                agl=np.clip(z - vc[2] + rng.normal(0, .3, n), 0, 20), hdist=hd, h_above=ha, d3=d3,
                pred=pred, dwell=dwell, vis=vis, vis_u=rng.uniform(-1, 1, n), vis_v=rng.uniform(-1, 1, n),
                vis_px=rng.uniform(2, 40, n), rgb_served=(rng.uniform(size=n) < .01).astype(int),
                depth_min_c=rng.uniform(.5, 30, n), clue_dx=sc[0] - xy[:, 0], clue_dy=sc[1] - xy[:, 1],
                act_ms=rng.gamma(4, 8, n), route=route, esc=esc)
    hmx = np.arange(-60, 61, 1.0); X, Y = np.meshgrid(hmx, hmx)
    hm = {"hm_z": (np.sin(X / 9) * np.cos(Y / 11) * 4).astype(np.float32), "hm_cat": np.zeros(X.shape, np.int8),
          "hm_x": hmx.astype(np.float32), "hm_y": hmx.astype(np.float32),
          "hm_cat_names": np.array(["ground"]), "hm_z_top": np.float32(80)}
    np.savez_compressed(out / "traj" / f"{group}_{seed}.npz",
                        **{c: cols[c].astype(np.int16 if c in INT else np.float32) for c in TRAJ_COLS}, **hm)
    success = not fail
    tt = sd.target_time(d_sc)
    mc = rng.uniform(0.3, 3)
    tterm = sd.time_term(t[-1], tt) if success else 0
    sterm = sd.safety_term(mc, ct) if success else 0
    score = .45 + .45 * tterm + .1 * sterm if success else .01
    fv = t[vis == 1]
    rec = dict(seed=seed, group=group, ok=True, challenge_type=ct, env_type=et, horizon=60.0,
               start=start.tolist(), fov_deg=90.0, victim_centre=vc.tolist(),
               victim_aabb=[(vc - ext / 2).tolist(), (vc + ext / 2).tolist()], victim_top_z=float(top),
               victim_extent=ext.tolist(), victim_lying=bool(ext[2] < 1), support_category=rng.choice(["ground", "roof", "rock"]),
               surface_z=float(vc[2] - ext[2] / 2), search_centre=sc.tolist(), d_start_centre=d_sc,
               d_start_victim=float(np.linalg.norm(vc[:2])), d_centre_victim=r, victim_origin_dist=float(np.linalg.norm(vc[:2])),
               victim_dz_start=float(top - start[2]), target_time=tt, success=success, score=score,
               success_term=float(success), time_term=tterm, safety_term=sterm, participation_term=0.0 if success else .01,
               final_score=score, failure_reason="NONE" if success else mode, sim_time=float(t[-1]),
               t_confirm=float(t[-1]) if success else None, collision=mode == "OBSTACLE_COLLISION",
               min_clearance=mc, sar_min_hdist=float(hd.min()), sar_min_d3=float(d3.min()),
               sar_max_dwell=float(dwell.max()), steps=n, act_errors=0, rgb_used=int(cols["rgb_served"].sum()),
               first_visible_t=float(fv[0]) if fv.size else None, clue_point=sc.tolist(),
               events=[{"t": 2.0, "kind": "latched", "value": [et]}, {"t": 9.0, "kind": "tour_on", "value": True}]
               + ([{"t": 30.0, "kind": "rgb_latch", "value": vc.tolist(), "err_m": float(rng.gamma(2, 1))}] if rng.uniform() < .4 else []),
               route_codes={"king": 0, "king+tour": 1, "king+rgb": 2}, esc_codes={"None": -1, "creep": 0, "climb": 1},
               cls_probs=list(rng.dirichlet(np.ones(6) * .3 + np.eye(6)[ct - 1] * 5)), cls_n=40,
               cls_argmax=ct - 1 if rng.uniform() < .85 else int(rng.integers(6)),
               cls_latched=[et], cls_is_mountain=et == "mountain", cls_p40=.6, cls_mtn_ok=et == "mountain",
               stype=et if et in ("forest", "village", "city", "mountain") else None, champ_route=et in ("village", "warehouse"),
               tour_waypoints=12, tour_wp_index=int(rng.integers(12)), av_count=int(rng.poisson(1)), cv_state="ok",
               rgb_n_req=int(rng.integers(0, 40)), rgb_n_fix=int(rng.integers(0, 6)), rgb_n_latch=int(rng.integers(0, 2)),
               rgb_n_dud=int(rng.integers(0, 2)), rgb_detector_loaded=True, rgb_bad_spots=[],
               esc_trap=bool(esc.max() >= 0), esc_final_phase=None,
               path_len=float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()), max_tilt_deg=8.0,
               act_ms_mean=float(cols["act_ms"].mean()), act_ms_p95=float(np.percentile(cols["act_ms"], 95)),
               act_ms_max=float(cols["act_ms"].max()), act_ms_first=900.0, wall_sec=float(rng.uniform(20, 200)))
    if rng.uniform() < .02:
        rec["success"] = False; rec["failure_reason"] = "INFEASIBLE"; rec["score"] = .01
    return rec


def run(out, bonus, per=12):
    out = Path(out); (out / "traj").mkdir(parents=True, exist_ok=True)
    with open(out / "episodes.jsonl", "w") as fh:
        for g in GROUPS:
            for k in range(per):
                fh.write(json.dumps(episode(out, g, 1000 + 37 * k + GROUPS.index(g), bonus), default=float) + "\n")
    rows = {}
    for line in open(out / "episodes.jsonl"):
        r = json.loads(line)
        rows.setdefault(r["group"], []).append({"group": r["group"], "seed": r["seed"], "challenge_type": r["challenge_type"],
                                                "score": r["score"] + np.random.normal(0, .005), "success": r["success"],
                                                "sim_time": r["sim_time"], "wall_time": r["wall_sec"] * 1.3, "execution_status": "ok"})
    Path(str(out) + "_bench.json").write_text(json.dumps({"group_results": rows}))


if __name__ == "__main__":
    run(sys.argv[1] + "/champion", 0.0)
    run(sys.argv[1] + "/variant", 0.05)
