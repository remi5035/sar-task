# Environment Notes

Platform requirements, installation, and behaviours of this codebase that are not apparent from
the source.

## Platform requirements

The simulator depends on `swarm-bullet3`, a Bullet fork that publishes two wheels to PyPI and no
source distribution:

```
swarm_bullet3-2.0.0.3-cp310-cp310-manylinux_2_34_x86_64.whl
swarm_bullet3-2.0.0.3-cp311-cp311-manylinux_2_34_x86_64.whl
```

The requirement is therefore **Linux x86-64, glibc 2.34 or newer, Python 3.10 or 3.11**. There is
no macOS, Windows, ARM or Python 3.12 build, and installation fails on any of them with no
available workaround.

The VPS provided with this task meets these requirements. Its details are in the email.

A GPU is not required at any point.

## Installation

The repository ships its own scripts, which handle system packages, Python 3.11, the virtual
environment and every dependency:

```bash
chmod +x miner/src/scripts/install_dependencies.sh miner/src/scripts/setup.sh
./miner/src/scripts/install_dependencies.sh
./miner/src/scripts/setup.sh
source miner_env/bin/activate
```

Two things those scripts do not cover.

### PyTorch

The requirements resolve to the CUDA build of PyTorch, several gigabytes of wheels that nothing in
this task uses. To avoid the download, install the CPU build into the virtual environment before
running `setup.sh`:

```bash
python3.11 -m venv miner_env && source miner_env/bin/activate
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
./miner/src/scripts/setup.sh
```

### Docker

`swarm benchmark` executes models inside a sandboxed container, and there is no alternative path to
a score. Docker is required by `swarm doctor` but is not installed by the setup scripts in this
repository:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER" && newgrp docker
```

Each container is then isolated from the host using `nsenter` and `iptables`. If either is absent,
or the invoking user lacks the capability to use them, every seed fails with an infrastructure
error that reports Docker rather than permissions. Run the benchmark under `sudo -E`, or grant the
capabilities once:

```bash
sudo apt install -y iptables util-linux
sudo setcap cap_sys_admin+ep "$(which nsenter)"
sudo setcap cap_net_admin+ep "$(readlink -f "$(which iptables)")"
```

The first benchmark run builds a container image of roughly 2 GB. Before building, the tooling
prunes stopped containers, dangling images and unused volumes on the host.

`swarm model verify` and `swarm video --backend local` run in process and require no Docker.
Neither produces a score.

## Known issue: the bundled reinforcement learning example

`miner/src/RL/` contains a PPO example that produces a valid, contract-compliant submission. It
cannot learn.

The reward passed to the learner is the per-step change in the rollout score. That score is a
terminal quantity: it remains constant until the episode ends, and the change in a constant is
zero. Every step of every episode therefore returns exactly 0.0, and no gradient signal reaches
the policy.

The example also flattens the depth image directly into a 64-unit dense layer with no convolution,
and never requests a colour frame.

It is included because it ships with the upstream repository, not as a recommended starting point.

## Run times

Depth rendering accounts for most of each simulation step and runs on CPU, so evaluation is
CPU-bound and scales with worker count. Cost per seed varies substantially by environment, and
successful episodes terminate early rather than running the full 60-second horizon.

A complete 1,100-seed run takes several hours. To iterate against a subset, trim the seed lists,
keeping all six groups present and non-empty:

```python
import json
d = json.load(open("TASK/practice_seeds.json"))
d["type_seeds"] = {k: v[:10] for k, v in d["type_seeds"].items()}
json.dump(d, open("quick_seeds.json", "w"), indent=2, sort_keys=True)
```

Figures reported in the write-up should come from the full set.

## Interpreting results

Per-seed scores are close to bimodal: an episode either confirms and scores near 0.95, or fails and
scores 0.01. Per-seed standard deviation is consequently around 0.25, so small samples carry little
information.

Compare runs on identical seed lists. On a small sample, a difference of a few points is not
evidence of an improvement, and reporting a result as inconclusive is a legitimate outcome.

## Command reference

| Flag | Effect |
|---|---|
| `--family-id cf_search_and_rescue` | Required. Several commands default to a different family. |
| `--seed-file` | Replays an exact seed set. Pass it on every run so results remain comparable. |
| `--summary-json-out` | Per-seed results including the failure reason for each seed. |
| `--relax-timeouts` | Raises the per-step compute budget. Without it on modest hardware, a slow model accumulates strikes and fails seeds for reasons unrelated to its behaviour. |
| `--workers N` | One worker per two cores. |

## Model constraints

| Constraint | Value |
|---|---|
| Uncompressed archive | 50 MiB |
| Container resources | 6 GB memory, 2 CPUs |
| Per-step budget | 0.6 baseline-equivalent seconds |
| First step | 2.0 seconds |
| Permitted packages | Docker whitelist in [`miner/docs/miner.md`](../miner/docs/miner.md) |

Timing is normalised against a hardware baseline, so slower machines are not penalised.
