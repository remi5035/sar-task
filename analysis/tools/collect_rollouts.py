#!/usr/bin/env python3
"""Instrumented in-process rollouts for cf_search_and_rescue.

`swarm benchmark --summary-json-out` only records score / success / sim_time per
seed: no failure reason, no clearance, no geometry, no trajectory. This script
replays the same seeds in process (no Docker) with the benchmark's own task
builder, env factory and action canonicalisation, and records everything the
diagnostic notebook needs:

  <out>/episodes.jsonl        one JSON line per seed (outcome, score terms, task
                              geometry, victim visibility, agent internals)
  <out>/traj/<group>_<seed>.npz  per-step trajectory + top-down height map
  <out>/rgb/<group>_<seed>.npz   optional probe RGB frames (--probe-rgb-every)
  <out>/run_config.json

What it does not reproduce: the per-step timing strikes of the Docker runner
(act times are recorded raw, uncalibrated) and the container's CPU limits.
Compare a run against the Docker benchmark on the same seeds in the notebook.

Usage (from the repo root, inside the activated environment):

  python analysis/tools/collect_rollouts.py --model TASK/champion/submission.zip \
      --seed-file TASK/practice_seeds.json --out analysis/data/champion --workers 4

  # quick subset: N seeds per environment type
  python analysis/tools/collect_rollouts.py ... --per-type 20

  # A/B an agent switch (the champion reads KT_* variables at import time)
  python analysis/tools/collect_rollouts.py ... --agent-env KT_RP=0 --out analysis/data/no_rgb

Re-running with the same --out resumes: seeds already in episodes.jsonl are skipped.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

FAMILY_ID = "cf_search_and_rescue"
GROUP_TO_TYPE = {
    "type1_city": 1,
    "type2_open": 2,
    "type3_mountain": 3,
    "type4_village": 4,
    "type5_warehouse": 5,
    "type6_forest": 6,
}
TYPE_NAMES = {1: "city", 2: "open", 3: "mountain", 4: "village", 5: "warehouse", 6: "forest"}

VIS_EVERY = 5            # victim visibility check cadence (steps)
HM_RES = 1.0             # height-map cell size (m)
HM_MARGIN = 10.0         # height-map margin around start / search circle / victim (m)
HM_MAX_SPAN = 180.0      # cap on the height-map side (m)
RAY_BATCH = 8000

# Per-step trajectory columns (float32 unless noted in _INT_COLS).
TRAJ_COLS = (
    "t", "x", "y", "z", "vx", "vy", "vz", "roll", "pitch", "yaw",
    "a_dx", "a_dy", "a_dz", "a_speed", "a_yaw", "a_rgb",
    "agl", "hdist", "h_above", "d3", "pred", "dwell",
    "vis", "vis_u", "vis_v", "vis_px", "rgb_served",
    "depth_min_c", "clue_dx", "clue_dy", "act_ms",
    "route", "esc",
)
_INT_COLS = {"pred", "vis", "rgb_served", "route", "esc"}


# --------------------------------------------------------------------------- #
# Agent loading
# --------------------------------------------------------------------------- #
_AGENT_MOD = None
_AGENT_DIR: Optional[Path] = None


def _load_agent_module(extracted: Path):
    if str(extracted) not in sys.path:
        sys.path.insert(0, str(extracted))
    spec = importlib.util.spec_from_file_location("drone_agent", str(extracted / "drone_agent.py"))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import drone_agent.py")
    mod = importlib.util.module_from_spec(spec)
    prev = os.getcwd()
    os.chdir(str(extracted))
    try:
        spec.loader.exec_module(mod)
    finally:
        os.chdir(prev)
    return mod


def _new_agent():
    prev = os.getcwd()
    os.chdir(str(_AGENT_DIR))
    try:
        agent = _AGENT_MOD.DroneFlightController()
        reset = getattr(agent, "reset", None)
        if callable(reset):
            reset()
    finally:
        os.chdir(prev)
    return agent


def _worker_init(extracted: str, agent_env: dict[str, str]) -> None:
    global _AGENT_MOD, _AGENT_DIR
    os.environ.update(agent_env)
    _AGENT_DIR = Path(extracted)
    _AGENT_MOD = _load_agent_module(_AGENT_DIR)


# --------------------------------------------------------------------------- #
# Agent introspection (champion-specific attributes; absent ones are skipped)
# --------------------------------------------------------------------------- #
class _Codes:
    """Stable string -> small int mapping, stored alongside each trajectory."""

    def __init__(self) -> None:
        self.table: dict[str, int] = {}

    def __call__(self, value: Any) -> int:
        if value is None:
            return -1
        key = str(value)
        if key not in self.table:
            self.table[key] = len(self.table)
        return self.table[key]


def _agent_step_view(agent) -> dict[str, Any]:
    v: dict[str, Any] = {"route": getattr(agent, "route", None)}
    esc = getattr(agent, "_esc", None)
    v["esc"] = getattr(esc, "phase", None) if esc is not None else None
    cls = getattr(agent, "_cls", None)
    if cls is not None:
        v["latched"] = tuple(sorted(getattr(cls, "latched", ())))
        v["is_mountain"] = bool(getattr(cls, "is_mountain", False))
    v["tour_on"] = bool(getattr(agent, "_on", False))
    rp = getattr(agent, "_rp", None)
    if rp is not None:
        latch = getattr(rp, "latch", None)
        v["rgb_latch"] = None if latch is None else [float(x) for x in np.asarray(latch).reshape(-1)[:3]]
        v["rgb_n_dud"] = int(getattr(rp, "n_dud", 0))
    return v


def _agent_final_view(agent) -> dict[str, Any]:
    out: dict[str, Any] = {}
    cls = getattr(agent, "_cls", None)
    if cls is not None:
        with contextlib.suppress(Exception):
            n = max(1, int(getattr(cls, "n", 0)))
            ps = np.asarray(getattr(cls, "ps"), dtype=float) / n
            out["cls_probs"] = [float(x) for x in ps]
            out["cls_n"] = int(getattr(cls, "n", 0))
            out["cls_argmax"] = int(np.argmax(ps)) if ps.sum() > 0 else -1
            out["cls_latched"] = sorted(getattr(cls, "latched", ()))
            out["cls_is_mountain"] = bool(getattr(cls, "is_mountain", False))
            p40 = getattr(cls, "p40", None)
            out["cls_p40"] = None if p40 is None else float(p40)
            out["cls_mtn_ok"] = bool(cls.mtn_ok()) if hasattr(cls, "mtn_ok") else None
    with contextlib.suppress(Exception):
        out["stype"] = agent._stype() if hasattr(agent, "_stype") and cls is not None else None
    with contextlib.suppress(Exception):
        out["champ_route"] = bool(agent._champ()) if hasattr(agent, "_champ") else None
    w = getattr(agent, "_w", None)
    out["tour_waypoints"] = None if w is None else int(len(w))
    out["tour_wp_index"] = int(getattr(agent, "_i", 0)) if w is not None else None
    out["av_count"] = int(getattr(agent, "_av_n", 0)) if hasattr(agent, "_av_n") else None
    out["cv_state"] = getattr(agent, "_cv", None)
    rp = getattr(agent, "_rp", None)
    if rp is not None:
        for k in ("n_req", "n_fix", "n_latch", "n_dud"):
            out[f"rgb_{k}"] = int(getattr(rp, k, 0))
        out["rgb_detector_loaded"] = getattr(rp, "sess", None) is not None
        out["rgb_bad_spots"] = [[float(b[0]), float(b[1])] for b in getattr(rp, "bad", [])]
    esc = getattr(agent, "_esc", None)
    if esc is not None:
        out["esc_trap"] = bool(getattr(esc, "trap", False))
        out["esc_final_phase"] = getattr(esc, "phase", None)
    return out


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _height_map(cli, world, start_xyz, challenge_type: int) -> dict[str, np.ndarray]:
    import pybullet as p

    sc = np.asarray(world.search_centre, dtype=float)
    vc = np.asarray(world.victim_centre, dtype=float)
    st = np.asarray(start_xyz, dtype=float)
    xs_all = [st[0], sc[0] - 32.0, sc[0] + 32.0, vc[0]]
    ys_all = [st[1], sc[1] - 32.0, sc[1] + 32.0, vc[1]]
    x0, x1 = min(xs_all) - HM_MARGIN, max(xs_all) + HM_MARGIN
    y0, y1 = min(ys_all) - HM_MARGIN, max(ys_all) + HM_MARGIN
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hx = min(HM_MAX_SPAN, x1 - x0) / 2
    hy = min(HM_MAX_SPAN, y1 - y0) / 2
    xs = np.arange(cx - hx, cx + hx + 1e-6, HM_RES)
    ys = np.arange(cy - hy, cy + hy + 1e-6, HM_RES)
    X, Y = np.meshgrid(xs, ys)
    top_ref = max(float(st[2]), float(world.victim_aabb[1][2]))
    # Indoors a ray from above hits the roof: start just above flight level instead.
    z_top = top_ref + (4.0 if challenge_type == 5 else 80.0)
    z_bot = min(float(st[2]), float(world.victim_aabb[0][2])) - 80.0
    frm = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z_top)], 1)
    to = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z_bot)], 1)
    hz = np.full(X.size, np.nan, np.float32)
    cat = np.full(X.size, -1, np.int8)
    tags = getattr(world, "body_tags", {}) or {}
    cat_names = sorted({str(v) for v in tags.values()})
    cat_idx = {n: i for i, n in enumerate(cat_names)}
    for i in range(0, X.size, RAY_BATCH):
        res = p.rayTestBatch(frm[i:i + RAY_BATCH].tolist(), to[i:i + RAY_BATCH].tolist(),
                             physicsClientId=cli)
        for j, r in enumerate(res):
            if int(r[0]) >= 0:
                hz[i + j] = float(r[3][2])
                cat[i + j] = cat_idx.get(str(tags.get(int(r[0]), "")), -1)
    return {
        "hm_z": hz.reshape(X.shape),
        "hm_cat": cat.reshape(X.shape),
        "hm_x": xs.astype(np.float32),
        "hm_y": ys.astype(np.float32),
        "hm_cat_names": np.array(cat_names),
        "hm_z_top": np.float32(z_top),
    }


def _camera_frame(env):
    import pybullet as p

    from swarm.constants import CAMERA_EYE_FWD_M, CAMERA_EYE_UP_M

    pos = np.asarray(env.pos[0, :], dtype=float)
    rot = np.array(p.getMatrixFromQuaternion(env.quat[0, :])).reshape(3, 3)
    fwd = rot @ np.array([1.0, 0.0, 0.0])
    up = rot @ np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    cam = pos + fwd * CAMERA_EYE_FWD_M + up * CAMERA_EYE_UP_M
    return cam, fwd, up, right


def _victim_visibility(env, world, cli, victim_uids: set[int]):
    """Geometric ground truth: is the victim inside the frustum, within 30 m, unoccluded?

    Returns (visible, u, v, apparent_px); u/v in [-1, 1] image coordinates with the
    same convention as the champion (u right, v up)."""
    import pybullet as p

    cam, fwd, up, right = _camera_frame(env)
    lo = np.asarray(world.victim_aabb[0], dtype=float)
    hi = np.asarray(world.victim_aabb[1], dtype=float)
    vc = (lo + hi) / 2.0
    rel = vc - cam
    x = float(rel @ fwd)
    dist = float(np.linalg.norm(rel))
    if x <= 0.3 or dist > 30.0:
        return 0, np.nan, np.nan, np.nan
    t = math.tan(math.radians(float(getattr(env, "_fov", 90.0)) / 2.0))
    u = float(rel @ right) / (x * t)
    v = float(rel @ up) / (x * t)
    if abs(u) > 1.0 or abs(v) > 1.0:
        return 0, u, v, np.nan
    targets = [vc.tolist(), [vc[0], vc[1], hi[2] - 0.03]]
    hits = p.rayTestBatch([cam.tolist()] * len(targets), targets, physicsClientId=cli)
    seen = any(int(h[0]) in victim_uids or int(h[0]) < 0 for h in hits)
    extent = float(np.max(hi - lo))
    px = extent / (x * 2.0 * t) * 256.0
    return int(seen), u, v, px


# --------------------------------------------------------------------------- #
# One episode
# --------------------------------------------------------------------------- #
def run_episode(job: dict[str, Any]) -> dict[str, Any]:
    import pybullet as p

    from swarm.challenge_families import build_random_task, get_challenge_family
    from swarm.constants import SIM_DT
    from swarm.core.action import canonicalize_action
    from swarm.validator.reward import _calculate_sar_target_time
    from swarm.utils.env_factory import make_env_with_initial_obs

    seed, group = int(job["seed"]), str(job["group"])
    out_dir = Path(job["out"])
    rec: dict[str, Any] = {"seed": seed, "group": group, "ok": False}
    t_wall0 = time.time()
    env = None
    try:
        task = build_random_task(sim_dt=SIM_DT, seed=seed, family_id=FAMILY_ID)
        ctype = int(task.challenge_type)
        rec.update(challenge_type=ctype, env_type=TYPE_NAMES.get(ctype, str(ctype)),
                   horizon=float(task.horizon))
        with contextlib.redirect_stdout(io.StringIO()):
            env, obs = make_env_with_initial_obs(task, gui=False)
        cli = getattr(env, "CLIENT", 0)
        world = env.sar_world
        start = np.asarray(task.start, dtype=float)
        rec["start"] = start.tolist()
        rec["fov_deg"] = float(getattr(env, "_fov", 90.0))
        if world is not None:
            vc = np.asarray(world.victim_centre, dtype=float)
            lo, hi = (np.asarray(a, dtype=float) for a in world.victim_aabb)
            sc = np.asarray(world.search_centre, dtype=float)
            rec.update(
                victim_centre=vc.tolist(),
                victim_aabb=[lo.tolist(), hi.tolist()],
                victim_top_z=float(hi[2]),
                victim_extent=(hi - lo).tolist(),
                victim_lying=bool((hi[2] - lo[2]) < 0.6 * max(hi[0] - lo[0], hi[1] - lo[1])),
                support_category=str(getattr(world.support_category, "value", world.support_category)),
                surface_z=float(world.surface_z),
                search_centre=sc.tolist(),
                d_start_centre=float(np.linalg.norm(start[:2] - sc)),
                d_start_victim=float(np.linalg.norm(start[:2] - vc[:2])),
                d_centre_victim=float(np.linalg.norm(sc - vc[:2])),
                victim_origin_dist=float(np.linalg.norm(vc[:2])),
                victim_dz_start=float(hi[2] - start[2]),
                target_time=float(_calculate_sar_target_time(task)),
            )
            victim_uids = {int(u) for u in world.victim_uids}
            if job.get("heightmap", True):
                hm = _height_map(cli, world, start, ctype)
            else:
                hm = {}
        else:
            victim_uids, hm = set(), {}

        agent = _new_agent()
        codes_route, codes_esc = _Codes(), _Codes()
        n_drones = int(getattr(env, "NUM_DRONES", 1))
        act_dim = int(env.action_space.shape[-1])
        lo_a, hi_a = env.action_space.low.flatten(), env.action_space.high.flatten()

        rows: list[list[float]] = []
        events: list[dict[str, Any]] = []
        prev_view: dict[str, Any] = {}
        act_errors = 0
        probe_every = int(job.get("probe_rgb_every") or 0)
        probe_frames, probe_meta = [], []
        t_sim, step = 0.0, 0
        terminated = truncated = False
        info: dict[str, Any] = {}
        clue0 = None
        first_vis_t = None

        while t_sim < task.horizon and not (terminated or truncated):
            ta = time.perf_counter()
            try:
                raw = agent.act(obs)
                if raw is None:
                    raw = np.zeros(act_dim, np.float32)
            except Exception:
                act_errors += 1
                raw = np.zeros(act_dim, np.float32)
            act_ms = (time.perf_counter() - ta) * 1000.0
            act = canonicalize_action(raw, lo_a, hi_a, n_drones=None, act_dim=act_dim)
            n_rgb_before = int(env._rgb_request_count[0]) if hasattr(env, "_rgb_request_count") else 0
            obs, _r, terminated, truncated, info = env.step(act[None, :])
            t_sim += SIM_DT
            step += 1
            rgb_served = int(hasattr(env, "_rgb_request_count")
                             and int(env._rgb_request_count[0]) > n_rgb_before)

            st = np.asarray(env._getDroneStateVector(0), dtype=float)
            pos, rpy, vel = st[0:3], st[7:10], st[10:13]
            ob_state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
            if clue0 is None and ob_state.size >= 2:
                clue0 = (pos[:2] + ob_state[-2:]).tolist()
            depth = np.asarray(obs["depth"], dtype=np.float32).reshape(256, 256)
            dmin_c = float(depth[96:160, 96:160].min()) * 29.5 + 0.5

            if world is not None:
                hdist = float(np.linalg.norm(pos[:2] - vc[:2]))
                h_above = float(pos[2] - hi[2])
                d3 = float(np.linalg.norm(pos - vc))
            else:
                hdist = h_above = d3 = np.nan
            pred = int(bool(getattr(env, "_sar_predicate_active", False)))
            dwell = float(getattr(env, "_sar_dwell_time", 0.0))

            vis = -1
            vu = vv = vpx = np.nan
            if world is not None and (step % VIS_EVERY == 0 or rgb_served):
                vis, vu, vv, vpx = _victim_visibility(env, world, cli, victim_uids)
                if vis == 1 and first_vis_t is None:
                    first_vis_t = t_sim

            if probe_every and world is not None and step % probe_every == 0:
                with contextlib.suppress(Exception):
                    fr = env._render_onboard_rgb(0)
                    pv = _victim_visibility(env, world, cli, victim_uids) if vis < 0 else (vis, vu, vv, vpx)
                    small = fr.reshape(128, 2, 128, 2, 3).mean(axis=(1, 3))
                    probe_frames.append((small * 255.0 + 0.5).astype(np.uint8))
                    probe_meta.append([t_sim, *pv, dmin_c, hdist,
                                       float(np.linalg.norm(pos - vc))])

            view = _agent_step_view(agent)
            for key in ("latched", "tour_on", "is_mountain", "esc"):
                if key in view and view.get(key) != prev_view.get(key):
                    events.append({"t": t_sim, "kind": key, "value": view[key]})
            if "rgb_latch" in view:
                was = prev_view.get("rgb_latch")
                now = view["rgb_latch"]
                if now is not None and was is None:
                    err = float(np.linalg.norm(np.asarray(now[:2]) - vc[:2])) if world is not None else None
                    events.append({"t": t_sim, "kind": "rgb_latch", "value": now, "err_m": err})
                elif now is None and was is not None:
                    events.append({"t": t_sim, "kind": "rgb_unlatch", "value": was})
            prev_view = view

            clue = ob_state[-2:] if ob_state.size >= 2 else (np.nan, np.nan)
            rows.append([
                t_sim, *pos, *vel, *rpy, *act.tolist(),
                float(ob_state[162]) * 20.0 if ob_state.size > 162 else np.nan,
                hdist, h_above, d3, pred, dwell,
                vis, vu, vv, vpx, rgb_served,
                dmin_c, float(clue[0]), float(clue[1]), act_ms,
                codes_route(view.get("route")), codes_esc(view.get("esc")),
            ])

        success = bool(info.get("success", False))
        family = get_challenge_family(FAMILY_ID)
        ev = family.evaluate_rollout(
            task=task, success=success, t=t_sim, horizon=task.horizon,
            min_clearance=info.get("min_clearance"), collision=bool(info.get("collision", False)),
            failure_reason=str(info.get("failure_reason", "NONE")),
        )
        traj = np.asarray(rows, dtype=np.float64)
        rec.update(
            ok=True,
            success=success,
            score=float(ev.score),
            **{k: float(v) for k, v in ev.normalized_metrics.items()},
            failure_reason=str(info.get("failure_reason", "NONE")),
            sim_time=float(t_sim),
            t_confirm=info.get("t_to_confirm"),
            collision=bool(info.get("collision", False)),
            min_clearance=None if info.get("min_clearance") is None else float(info["min_clearance"]),
            sar_min_hdist=float(info.get("sar_min_horizontal_distance", np.nan)),
            sar_min_d3=float(info.get("sar_min_sphere_distance", np.nan)),
            sar_max_dwell=float(info.get("sar_max_dwell", 0.0)),
            steps=step,
            act_errors=act_errors,
            rgb_used=int(env._rgb_request_count[0]) if hasattr(env, "_rgb_request_count") else None,
            first_visible_t=first_vis_t,
            clue_point=clue0,
            events=events,
            route_codes=codes_route.table,
            esc_codes=codes_esc.table,
            **_agent_final_view(agent),
        )
        if len(traj):
            rec.update(
                path_len=float(np.sum(np.linalg.norm(np.diff(traj[:, 1:4], axis=0), axis=1))),
                max_tilt_deg=float(np.degrees(np.max(np.abs(traj[:, 7:9])))),
                act_ms_mean=float(np.mean(traj[:, TRAJ_COLS.index("act_ms")])),
                act_ms_p95=float(np.percentile(traj[:, TRAJ_COLS.index("act_ms")], 95)),
                act_ms_max=float(np.max(traj[:, TRAJ_COLS.index("act_ms")])),
                act_ms_first=float(traj[0, TRAJ_COLS.index("act_ms")]),
            )
        if job.get("traj", True):
            arrays = {c: traj[:, i].astype(np.int16 if c in _INT_COLS else np.float32)
                      for i, c in enumerate(TRAJ_COLS)} if len(traj) else {}
            arrays.update(hm)
            np.savez_compressed(out_dir / "traj" / f"{group}_{seed}.npz", **arrays)
        if probe_frames:
            np.savez_compressed(
                out_dir / "rgb" / f"{group}_{seed}.npz",
                frames=np.stack(probe_frames),
                meta=np.asarray(probe_meta, dtype=np.float32),
                meta_cols=np.array(["t", "vis", "u", "v", "px", "depth_min_c", "hdist", "d3"]),
            )
    except Exception as exc:  # the record says what failed; the run goes on
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc()
    finally:
        if env is not None:
            with contextlib.suppress(Exception):
                env.close()
        rec["wall_sec"] = time.time() - t_wall0
    return rec


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _load_seeds(path: Path, per_type: Optional[int], groups: Optional[list[str]]) -> list[tuple[str, int]]:
    d = json.loads(path.read_text())
    type_seeds = d.get("type_seeds", d)
    jobs = []
    for g, seeds in type_seeds.items():
        if groups and g not in groups:
            continue
        for s in (seeds[:per_type] if per_type else seeds):
            jobs.append((g, int(s)))
    # interleave types so a partial run stays balanced
    jobs.sort(key=lambda gs: (type_seeds[gs[0]].index(gs[1]), gs[0]))
    return jobs


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="submission.zip or an extracted directory")
    ap.add_argument("--seed-file", type=Path, default=REPO / "TASK" / "practice_seeds.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--per-type", type=int, default=None, help="first N seeds of each type")
    ap.add_argument("--groups", nargs="*", default=None, help="restrict to these groups (e.g. type3_mountain)")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="restrict to these seeds")
    ap.add_argument("--agent-env", nargs="*", default=[], help="KEY=VALUE set before importing the agent")
    ap.add_argument("--no-traj", action="store_true", help="skip per-step trajectories")
    ap.add_argument("--no-heightmap", action="store_true")
    ap.add_argument("--probe-rgb-every", type=int, default=0,
                    help="render an extra RGB frame every N steps for detector analysis "
                         "(not shown to the agent, does not use its budget)")
    args = ap.parse_args(argv)

    out = args.out.resolve()
    (out / "traj").mkdir(parents=True, exist_ok=True)
    if args.probe_rgb_every:
        (out / "rgb").mkdir(parents=True, exist_ok=True)

    agent_env = dict(kv.split("=", 1) for kv in args.agent_env)
    if args.model.is_dir():
        extracted = args.model.resolve()
        tmp = None
    else:
        tmp = Path(tempfile.mkdtemp(prefix="sar_agent_"))
        with zipfile.ZipFile(args.model) as zf:
            zf.extractall(tmp)
        extracted = tmp

    jobs = _load_seeds(args.seed_file, args.per_type, args.groups)
    if args.seeds:
        wanted = set(args.seeds)
        jobs = [j for j in jobs if j[1] in wanted]
    ep_path = out / "episodes.jsonl"
    done = set()
    if ep_path.exists():
        for line in ep_path.read_text().splitlines():
            with contextlib.suppress(Exception):
                r = json.loads(line)
                if r.get("ok"):
                    done.add((r["group"], int(r["seed"])))
    todo = [j for j in jobs if j not in done]

    (out / "run_config.json").write_text(json.dumps({
        "model": str(args.model.resolve()),
        "seed_file": str(args.seed_file.resolve()),
        "agent_env": agent_env,
        "per_type": args.per_type,
        "probe_rgb_every": args.probe_rgb_every,
        "traj_cols": TRAJ_COLS,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2))
    print(f"{len(jobs)} seeds, {len(done)} already done, {len(todo)} to run, {args.workers} workers -> {out}",
          flush=True)

    payload = [{"seed": s, "group": g, "out": str(out), "traj": not args.no_traj,
                "heightmap": not args.no_heightmap, "probe_rgb_every": args.probe_rgb_every}
               for g, s in todo]
    t0 = time.time()
    n_ok = n_succ = 0
    ctx = mp.get_context("spawn")
    try:
        with ctx.Pool(args.workers, initializer=_worker_init, initargs=(str(extracted), agent_env),
                      maxtasksperchild=25) as pool, ep_path.open("a") as fh:
            for i, rec in enumerate(pool.imap_unordered(run_episode, payload), 1):
                fh.write(json.dumps(rec, default=float) + "\n")
                fh.flush()
                n_ok += int(rec.get("ok", False))
                n_succ += int(bool(rec.get("success")))
                el = time.time() - t0
                eta = el / i * (len(payload) - i)
                tag = "OK  " if rec.get("success") else ("ERR " if not rec.get("ok") else "FAIL")
                print(f"[{i}/{len(payload)}] {tag} {rec['group']:<16} {rec['seed']:>8} "
                      f"score={rec.get('score', 0):.3f} {rec.get('failure_reason', rec.get('error', ''))} "
                      f"| success {n_succ}/{n_ok} | eta {eta / 60:.0f} min", flush=True)
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
