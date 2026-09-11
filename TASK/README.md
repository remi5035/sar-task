# Search and Rescue: Engineering Task

This is a take-home exercise on a live benchmark. You are working in a copy of the production
repository for Bittensor subnet 124, against the same simulator, the same scoring and the same
1,100-seed evaluation the network itself runs.

Read [`NOTES.md`](NOTES.md) before installing anything. Every item in it has cost somebody a day.

## The problem

A drone lifts off with a rough pointer to a search area. A person is on the ground somewhere inside
a 30 metre circle around that point, never at its centre. The drone has 60 seconds and a 3 m/s
speed limit.

To succeed it must hold a hover 2 to 4 metres above the person, within 2 metres horizontally, under
1 m/s, for two unbroken seconds. Coming within 0.8 metres ends the mission in failure. It sees
depth continuously out to 30 metres, and colour only on request, 40 times per flight.

```
score = 0.45 x found them
      + 0.45 x how fast
      + 0.10 x how safely it flew

anything short of a confirmed hover = 0.01
```

The full specification is in [`docs/families/search_and_rescue.md`](../docs/families/search_and_rescue.md).
It is thorough and worth reading properly before you start.

## The starting point

`champion/` holds the model that currently holds the crown on this challenge. It scores **0.9151**
across 1,100 seeds spanning six environment types. Its source and weights are published, as every
champion's are, and you are free to fork it, modify it, replace it or ignore it.

## What we are asking

**Find where that model loses score. Work out why. Improve it. Show us how you measured.**

Hand back three things:

1. **What you found.** Where the score goes, and why. Include the evidence you used to decide, not
   only the conclusion.
2. **What you changed.** Any improvement, with a before and after on the practice seed set, and an
   honest read on whether the difference is real or noise.
3. **What you would do next.** One page. Given a month and a GPU, what would you build, and what
   would you measure to know it had worked.

Item 2 carries less weight than items 1 and 3. Nobody is expected to beat the champion in a few
days, and a large jump in score is not what we are looking for. We want to see how you approach a
system you did not write, on a problem where the answer is not known.

## Running it

Install, then confirm the environment is healthy:

```bash
swarm doctor
```

Every line marked required must read OK. If Docker is not among them, see `NOTES.md`.

Benchmark the champion to get your baseline:

```bash
./run_baseline.sh
```

Benchmark your own model against the same seeds:

```bash
swarm model package --source ./my_model --family-id cf_search_and_rescue
swarm model verify --model Submission/submission.zip
swarm benchmark --model Submission/submission.zip \
  --family-id cf_search_and_rescue \
  --seed-file TASK/practice_seeds.json \
  --workers 4 --relax-timeouts \
  --summary-json-out results.json
```

`practice_seeds.json` holds 1,100 seeds in the same per-map proportions the live benchmark uses, so
a full run is directly comparable to the published leaderboard figures. A full run is also several
hours. See `NOTES.md` for how to cut it down while you are iterating.

To see a failure rather than read about it, take a seed that scored badly and render the flight:

```bash
swarm visualize --summary-json results.json --failed
swarm video --model Submission/submission.zip --seed <seed> --type <type> \
  --family-id cf_search_and_rescue --backend local --mode depth --out ./videos
```

## How your work is scored

Your submission is evaluated on a **separate 1,100-seed set that you have not seen**, drawn the same
way as the practice set and held back. Tuning to the practice seeds will not transfer, and the gap
between the two numbers is itself something we look at.

We run that evaluation ourselves on one machine, so hardware differences between candidates do not
affect the result.

## Handing it back

Push everything to this repository: your code, your model artifact, and your write-up as a markdown
file at the repository root. Commit history is welcome and is not judged.

## Further reading

| Document | Why |
|---|---|
| [`docs/families/search_and_rescue.md`](../docs/families/search_and_rescue.md) | The complete task specification |
| [`miner/docs/miner.md`](../miner/docs/miner.md) | Packaging, submission format, runtime limits |
| [`docs/CLI_readme.md`](../docs/CLI_readme.md) | Every command and flag |
| [`NOTES.md`](NOTES.md) | Environment constraints and known traps |
