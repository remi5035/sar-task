"""Capture ce que voient les caméras (profondeur + RGB) à quelques instants clés d'un épisode.

Usage : python capture_views.py <model_dir> <seed,seed,...> <out_dir>
Instants gardés par seed : t = 1 s, la vue où la victime paraît la plus grande, et le premier pas
à moins de 2 m horizontalement (victime sous le drone, hors champ).
"""
import sys, io, contextlib
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "analysis"))
import collect_rollouts as cr
from swarm.challenge_families import build_random_task
from swarm.constants import SIM_DT
from swarm.utils.env_factory import make_env_with_initial_obs
from swarm.core.action import canonicalize_action

model_dir = Path(sys.argv[1]); seeds = [int(s) for s in sys.argv[2].split(",")]; out = Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
cr._worker_init(str(model_dir), {})
for seed in seeds:
    task = build_random_task(sim_dt=SIM_DT, seed=seed, family_id=cr.FAMILY_ID)
    with contextlib.redirect_stdout(io.StringIO()):
        env, obs = make_env_with_initial_obs(task, gui=False)
    cli = getattr(env, "CLIENT", 0); world = env.sar_world
    uids = {int(u) for u in world.victim_uids}
    lo, hi = (np.asarray(a, float) for a in world.victim_aabb); vc = (lo + hi) / 2
    agent = cr._new_agent()
    lo_a, hi_a = env.action_space.low.flatten(), env.action_space.high.flatten()
    shots = {}; best_px = 0.0; t = 0.0; step = 0

    def grab(tag):
        st = np.asarray(env._getDroneStateVector(0), float)
        vis, u, v, px = cr._victim_visibility(env, world, cli, uids)
        shots[tag] = dict(depth=np.asarray(obs["depth"], np.float32).reshape(256, 256),
                          rgb=np.asarray(env._render_onboard_rgb(0), np.float32).reshape(256, 256, 3),
                          t=t, pos=st[0:3], rpy=st[7:10], vis=vis, u=u, v=v, px=px,
                          hdist=float(np.linalg.norm(st[0:2] - vc[0:2])), h_above=float(st[2] - hi[2]))

    while t < task.horizon:
        a = np.asarray(agent.act(obs), np.float32)
        obs, _r, term, trunc, info = env.step(canonicalize_action(a, lo_a, hi_a, n_drones=None, act_dim=6)[None, :])
        t += SIM_DT; step += 1
        if step == 50: grab("start")
        if step % 10 == 0:
            vis, u, v, px = cr._victim_visibility(env, world, cli, uids)
            if vis == 1 and px > best_px:
                best_px = px; grab("victim")
        hd = float(np.linalg.norm(np.asarray(env.pos[0, :2]) - vc[:2]))
        if hd < 2.0 and "above" not in shots: grab("above")
        if term or trunc: break
    for tag, s in shots.items():
        np.savez_compressed(out / f"{seed}_{tag}.npz", **s)
        print(seed, tag, f"t={s['t']:.1f} vis={s['vis']} px={s['px']:.0f} hdist={s['hdist']:.1f} h_above={s['h_above']:.1f}", flush=True)
    env.close()
