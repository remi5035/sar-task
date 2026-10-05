# Diagnostic du champion SAR

| Fichier | Rôle |
|---|---|
| `sar_diagnostic.ipynb` | Le notebook, **avec ses sorties** (lisible directement sur GitHub) : environ 50 figures en 18 sections (score, échecs, temps, sécurité, géométrie, perception, hover, internes de l'agent, cinématique, calcul, galerie, cohérence Docker, A/B, puissance statistique, priorisation, analyses complémentaires, correctifs, résultat final) |
| `sar_diagnostic_rapport.pdf` | Le même notebook en PDF, sans le code |
| `sar_diag.py` | Chargement, taxonomie des issues, décomposition de la perte, features de trajectoire, statistiques (Wilson, bootstrap, McNemar, permutation) — numpy/pandas uniquement |
| `build_pdf.sh` | Réexécute le notebook sur place et régénère le PDF (Windows, Git Bash, Edge pour l'impression) |
| `make_fake_data.py` | Données **factices** au même format, pour tester le notebook sans simulateur (ne pas interpréter) |

## Entrées

Le notebook lit `../data/` :
- `champion_full/episodes.jsonl` : les 1 100 épisodes instrumentés du champion (`tools/collect_rollouts.py`) ;
- `bench/bench_champion_full.json` et `bench/bench_v6_full.json` : les deux benchmarks Docker officiels ;
- `v1_targeted/`, `v1_reg_mtn/`, `v2_rgb/` : les tests ciblés de la section 17.

Les trajectoires par pas (`<run>/traj/*.npz`, environ 300 Mo) ne sont pas versionnées : les sections qui en ont besoin
(features de trajectoire, galerie, figures avant/après) demandent de les régénérer avant de réexécuter le notebook.

## Utilisation

Sur une machine Linux avec le simulateur (venv activé, depuis la racine du repo) :

```bash
python analysis/tools/collect_rollouts.py --model TASK/champion/submission.zip \
    --seed-file TASK/practice_seeds.json --out analysis/data/champion_full --workers 4
pip install pandas matplotlib jupyter
cd analysis/diagnostic && jupyter lab sar_diagnostic.ipynb
```

Les chemins se règlent dans la cellule « Configuration » (`RUN_A`, `RUN_B`, `BENCH_A`). Les sections dont les données
manquent sont ignorées. Les features de trajectoire sont mises en cache dans `<run>/_traj_features.pkl`.

Pour prévisualiser sans simulateur : `python make_fake_data.py ../fake`, puis `DATA = HERE.parent / "fake"`
et `RUN_B = DATA / "variant"`.
