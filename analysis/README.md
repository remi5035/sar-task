# Search & rescue agent: analysis and improvement

Everything I added to this repository lives in this folder. The rest of the repository is the task as provided.

## Result

Official `swarm benchmark` (Docker), all 1,100 practice seeds, same seeds and same command for both agents:

| | Champion | **agent_final** |
|---|---|---|
| Mean score | 0.8917 | **0.9038** |
| Successes | 993 / 1,100 | **1,008 / 1,100** |

**+0.0121, paired bootstrap 95% CI [+0.0039, +0.0207]**: 21 seeds recovered, 6 lost.

| Terrain | Champion | agent_final | Change | 95% CI | Recovered / lost |
|---|---|---|---|---|---|
| mountain | 0.7799 | 0.8423 | **+0.062** | [+0.019, +0.108] | 17 / 4 |
| warehouse | 0.8791 | 0.8935 | +0.014 | [−0.000, +0.034] | 3 / 0 |
| city | 0.9561 | 0.9520 | −0.004 | [−0.022, +0.012] | 1 / 2 |
| open, village, forest | | | 0 (identical) | | 0 / 0 |

The champion itself measures 0.8917 (95% CI [0.872, 0.907]), not the published 0.9151.

## What to open

| Path | What it is |
|---|---|
| [`agent_final/`](agent_final/) | **The final agent.** `submission.zip` is the file to submit; the same files are unpacked next to it. |
| [`agent_final/champion_vs_agent_final.diff`](agent_final/champion_vs_agent_final.diff) | Every line changed relative to the champion (+172 / −11, all in `drone_agent.py`). |
| [`diagnostic/sar_diagnostic.ipynb`](diagnostic/sar_diagnostic.ipynb) | The analysis notebook, with its outputs: where the champion loses score and why (sections 1–16), the fixes (17), the final benchmark (18). |
| [`diagnostic/sar_diagnostic_rapport.pdf`](diagnostic/sar_diagnostic_rapport.pdf) | The same notebook as a PDF report, without code. |
| [`results/JOURNAL.md`](results/JOURNAL.md) | The full lab notebook (French): every hypothesis, test and number, in order. |
| [`results/`](results/) | Comparison tables: 1,100-seed benchmark, 360-seed validation of agent_final, 180-seed validation of the rejected v7, and the deck figures. |
| [`data/`](data/) | Inputs to the above: both benchmark JSONs and the per-episode logs (`episodes.jsonl`) of each run. |
| [`tools/`](tools/) | Scripts: instrumented rollouts, paired comparisons, figures. |

## What changed in the agent

The neural networks are untouched: `policy.onnx` has the same graph and weights as the champion's and only exposes two
internal detector outputs, used by an option that is off by default. All changes are in the Python wrapper
`drone_agent.py`, each targets one cause measured in the diagnostic, and each can be switched off by an environment variable.

| Change | Champion problem | Fix | Terrain | Switch |
|---|---|---|---|---|
| Take-off escape | Starts under a shelf and crashes on the way up | Obstacle just above at start → move out sideways first | warehouse | `KT_TKO=0` |
| Fast descent | Descends at 0.45 m/s after locking on the victim; a 6 s timer abandons it mid-descent | Descent up to 1.5 m/s; timer only counts at hover height | mountain | `KT_TDESC=0` |
| Relaxed RGB lock | The RGB detector sees the victim, but a filter drops 17 true detections to remove 2 false ones | Wider filter, 25 m range, one very confident detection is enough | mountain | `KT_RP_HMAX=1 KT_RP_RANGE=20 KT_RP_ONE=0` |
| Height cap | Starting on a peak, the drone searches 20–40 m above a victim in the valley | At most 12 m above ground inside the 30 m hint circle | mountain | `KT_AGL_CAP=0` |
| Vertical sweep | Fixed hover height misses standing victims | Slow descent from 5.0 to 2.4 m above the locked point | mountain | `KT_SWEEP=0 KT_HOVER=3.2` |

Village, forest and open keep the champion's behaviour exactly, and their scores are identical seed for seed.

## Tried and rejected

| Variant | Idea | Result | Decision |
|---|---|---|---|
| v4 / v5 | Lock on the internal depth detector | Recovered 5 of 12 targeted failures, but −0.002 [−0.025, +0.020] on 360 seeds: false locks in forest and village | Off by default (`KT_DL=1` re-enables it) |
| v7 | Same lock, only after 35 s | −0.018 [−0.038, −0.003] vs agent_final on 180 seeds | Rejected |

Earlier attempts (wall-backed take-off, long yaw scan) are described in the journal.
The intermediate versions v1–v7, probes, videos and raw trajectories are not versioned; the journal records what each one did and measured.

## How it was measured

1. **Diagnostic**: instrumented rollouts of the champion on all 1,100 seeds (`tools/collect_rollouts.py`): trajectory,
   every detector output and every wrapper decision per step.
2. **Targeted tests**: each change is first run on the 10–20 seeds it aims at, plus successful control seeds that must not change.
3. **Broad validation**: first 60 seeds of each terrain (360, not hand-picked), paired against the champion → +0.015 [−0.003, +0.034].
4. **Official benchmark**: `swarm benchmark` in Docker on all 1,100 seeds (above).

Confidence intervals are paired bootstraps over seeds (`tools/compare_bench.py`): both agents play the same seeds, the
per-seed differences are resampled 10,000 times. They answer "would the gain hold on other seeds from the same
distribution"; they do not include run-to-run noise, which exists on a few city seeds (two runs of the same champion disagree on 6 of 1,100).

## Reproducing

From the repository root, on Linux with the miner environment:

```bash
# official benchmark of the final agent (≈ 6.5 h with 6 workers on 12 vCPU)
swarm benchmark --model analysis/agent_final/submission.zip --family-id cf_search_and_rescue \
  --seed-file TASK/practice_seeds.json --workers 6 --relax-timeouts --summary-json-out bench.json
python analysis/tools/compare_bench.py analysis/data/bench/bench_champion_full.json bench.json

# instrumented rollouts (input of the notebook)
python analysis/tools/collect_rollouts.py --model analysis/agent_final/submission.zip \
  --seed-file TASK/practice_seeds.json --out analysis/data/my_run --workers 4
```

Runs longer than 6 hours hit a harness bug: the host speed calibration expires after 6 h
(`_CALIBRATION_MAX_AGE_SEC` in `swarm/validator/docker/docker_evaluator_parts/batch.py`) and re-calibrating from a
worker process crashes the whole run. The final benchmark was run with that constant raised to 24 h, which only affects
timing normalisation, not scores.

Raw trajectories (`traj/`, about 300 MB) are not versioned. The notebook's trajectory sections need them, so
regenerate them with `collect_rollouts.py` before re-running it.
