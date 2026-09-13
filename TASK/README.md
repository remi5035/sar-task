# Search and Rescue: Engineering Task

This repository is a copy of the production codebase for Bittensor subnet 124. The simulator, the
scoring and the 1,100-seed evaluation are the same ones the live network runs.

An Ubuntu VPS is provided for this task, with the details in the accompanying email. Read
[`NOTES.md`](NOTES.md) before starting: it covers installation, how to run the benchmark, and
several behaviours of this codebase that are not obvious from the source.

## The challenge

A drone lifts off with a coarse pointer to a search area. A person is on the ground somewhere
inside a 30 metre circle around that point, and never at its centre. The episode lasts 60 seconds
with a 3 m/s speed limit.

The mission succeeds when the drone holds a hover 2 to 4 metres above the person, within 2 metres
horizontally, below 1 m/s, for two continuous seconds. Approaching within 0.8 metres terminates the
episode as a failure. Depth is available continuously to 30 metres. Colour is available on request,
40 frames per episode.

```
score = 0.45 x mission success
      + 0.45 x time
      + 0.10 x clearance held

any outcome short of a confirmed hover = 0.01
```

A model's score is the mean across 1,100 procedurally generated seeds spanning six environment
types. The complete specification is in
[`docs/families/search_and_rescue.md`](../docs/families/search_and_rescue.md).

[`search_and_rescue.mp4`](search_and_rescue.mp4) shows the mission running across those
environments: the terrain, the casualties on the ground, and what the drone sees on approach.

## Starting point

`champion/submission.zip` is the model currently holding the crown on this challenge, scoring
**0.9151**. Every champion on this network is published with its source and weights, so it is
yours to read, fork, modify, replace or disregard.

## The exercise

Find where that model loses score, establish why, and improve it.

Three deliverables:

| | Deliverable | |
|---|---|---|
| 1 | **Analysis** | Where the score is lost and why, with the evidence behind the conclusion rather than the conclusion alone. |
| 2 | **A change** | Any improvement you can demonstrate, with before and after figures on the practice seed set and an assessment of whether the difference is significant. |
| 3 | **A plan** | One page. Given a month and a GPU, what you would build and what you would measure to establish that it worked. |

The exercise weights 1 and 3 above 2. This benchmark has been open to a competitive network for
months, so a large score improvement in a few days is not the expected outcome. What is being
assessed is how you approach an unfamiliar system and an open problem.

## Evaluation

Submissions are scored on a separate 1,100-seed set, generated the same way as the practice set and
withheld. The two sets do not overlap. A change tuned to the practice seeds will not carry across,
and the difference between the two figures is itself informative.

Scoring runs on a single machine for all submissions, because per-step timing strikes make results
from different hardware incomparable.

## Working with the benchmark

Confirm the environment first. Every line marked required must report OK:

```bash
swarm doctor
```

Establish the baseline:

```bash
./TASK/run_baseline.sh
```

Evaluate your own model against the same seeds:

```bash
swarm model package --source ./my_model --family-id cf_search_and_rescue
swarm model verify --model Submission/submission.zip
swarm benchmark --model Submission/submission.zip \
  --family-id cf_search_and_rescue \
  --seed-file TASK/practice_seeds.json \
  --workers 4 --relax-timeouts \
  --summary-json-out results.json
```

`practice_seeds.json` holds 1,100 seeds in the same per-map proportions as the live benchmark, so a
complete run is directly comparable to published leaderboard figures. A complete run takes several
hours; `NOTES.md` covers how to work against a subset while iterating.

Individual failures can be inspected rather than inferred. List the seeds that did not succeed,
then render one:

```bash
swarm visualize --summary-json results.json --failed
swarm video --model Submission/submission.zip --seed <seed> --type <type> \
  --family-id cf_search_and_rescue --backend local --mode depth --out ./videos
```

## Submitting

Push to this repository: your code, your model artifact, and the write-up as a markdown file at the
repository root.

## Reference

| Document | Covers |
|---|---|
| [`NOTES.md`](NOTES.md) | Platform requirements, installation, run times, known issues |
| [`docs/families/search_and_rescue.md`](../docs/families/search_and_rescue.md) | Full task specification: observations, actions, episode rules, scoring |
| [`miner/docs/miner.md`](../miner/docs/miner.md) | Packaging, submission format, runtime limits |
| [`docs/CLI_readme.md`](../docs/CLI_readme.md) | Every command and flag |
