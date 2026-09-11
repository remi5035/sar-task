# Environment Notes

Read this before installing. Each item below is a real constraint of this codebase, verified
against it, and each one has cost somebody a working day.

## Platform

The simulator depends on a custom Bullet fork, `swarm-bullet3`, which publishes exactly two wheels
to PyPI and no source distribution:

```
swarm_bullet3-2.0.0.3-cp310-cp310-manylinux_2_34_x86_64.whl
swarm_bullet3-2.0.0.3-cp311-cp311-manylinux_2_34_x86_64.whl
```

That means **Linux x86-64, glibc 2.34 or newer, Python 3.10 or 3.11**. There is no macOS build, no
Windows build, no ARM build, and no Python 3.12 build. Installation fails outright on anything
else, and there is no workaround short of building the fork yourself.

If the machine you have does not meet that, ask us and we will give you one that does.

## Docker

`swarm benchmark` runs your model inside a sandboxed container and there is no path around it. The
setup scripts in this repository do not install Docker, but `swarm doctor` requires it, so install
it separately:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER" && newgrp docker
```

Each container is then locked down from the host using `nsenter` and `iptables`. If those two are
missing, or if you run as a user without the capability to use them, **every seed fails** with an
infrastructure error that names Docker rather than permissions. Either run the benchmark with
`sudo -E`, or grant the capabilities once:

```bash
sudo apt install -y iptables util-linux
sudo setcap cap_sys_admin+ep "$(which nsenter)"
sudo setcap cap_net_admin+ep "$(readlink -f "$(which iptables)")"
```

The first benchmark run builds a container image of roughly 2 GB, which takes several minutes.
Before building it, the tooling prunes stopped containers, dangling images and unused volumes on
your machine. If you have other Docker work in progress, be aware of that.

`swarm model verify` and `swarm video --backend local` both run in-process and need no Docker, but
neither produces a score.

## Save the CUDA download

The host requirements pull the CUDA build of PyTorch, roughly 4 GB of wheels that nothing here
uses. No GPU is required anywhere in this task. Install the CPU build first and the rest of the
install will treat it as satisfied:

```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e . --no-deps
```

## The reinforcement learning starter does not train

`miner/src/RL/` contains a PPO example that packages a valid, contract-compliant submission. It
cannot learn anything.

The reward it passes to the learner is the per-step change in the rollout score. That score is a
terminal quantity: until an episode ends it is a constant, and the change in a constant is zero. A
full episode therefore delivers a reward of exactly 0.0 on every one of its 3,000 steps, so there
is no gradient signal of any kind.

It also flattens the depth image straight into a 64-unit dense layer with no convolution, and never
requests a colour frame.

You are welcome to build your own training loop. Do not assume this one is a working starting
point, and do not spend days waiting on it.

## Compute and run times

Measured on eight cores. A full 60-second episode costs, per seed on one worker:

| Environment | CPU time |
|---|---|
| city | 19 s |
| open | 60 s |
| village | 90 s |
| mountain | 215 s |
| forest | 225 s |
| warehouse | 292 s |

Rendering the depth camera is 45 to 80 percent of every simulation step, and it is CPU only.
Episodes that succeed end early, so real runs are faster than the table implies.

A full 1,100-seed run is several hours. While iterating, make a smaller file by trimming the lists
in `practice_seeds.json`, keeping all six groups present and non-empty:

```python
import json
d = json.load(open("TASK/practice_seeds.json"))
d["type_seeds"] = {k: v[:10] for k, v in d["type_seeds"].items()}
json.dump(d, open("quick_seeds.json", "w"), indent=2, sort_keys=True)
```

Use the full set for any number you report.

## Measuring honestly

Per-seed scores on this benchmark are close to bimodal: an episode either confirms and scores near
0.95, or fails and scores 0.01. The per-seed standard deviation is therefore large, around 0.25, so
a twenty-seed run tells you very little.

Compare runs on **identical seed lists**, and treat a change smaller than a couple of points on a
small sample as unproven. Saying that a result is inconclusive is a valid finding here.

## Useful flags

| Flag | Why |
|---|---|
| `--relax-timeouts` | Raises the per-step compute budget. On a laptop, without it, a slow model accumulates strikes and fails seeds for reasons unrelated to its behaviour. |
| `--seed-file` | Replays an exact seed set. Always pass it, so your runs are comparable. |
| `--summary-json-out` | Per-seed results, including the failure reason for every seed. This is the most useful artifact the benchmark produces. |
| `--workers N` | One worker per two cores. |
| `--family-id` | Always pass `cf_search_and_rescue`. Several commands default to a different family. |

## Runtime limits your model must respect

| Limit | Value |
|---|---|
| Uncompressed archive | 50 MiB |
| Container resources | 6 GB memory, 2 CPUs |
| Per-step budget | 0.6 baseline-equivalent seconds |
| First step | 2.0 seconds |
| Package whitelist | see the Docker whitelist in `miner/docs/miner.md` |

Timing is hardware normalised, so a slower machine is not penalised.
