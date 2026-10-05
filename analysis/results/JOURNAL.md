# Journal de travail : analyse et amélioration de l'agent SAR (1–5 octobre 2026)

> Journal tenu au fil du travail. **Résultat final en section 12** : agent_final (= agent_v6) +0.0121, IC 95 % [+0.0039, +0.0207]
> sur le benchmark officiel de 1 100 seeds.
>
> Les chemins cités sont ceux de l'époque. Correspondance avec le dossier actuel :
> `fixes/agent_v6/` → `agent_final/` ; `fixes/NOTES_2026-10-03.md` → ce fichier ; `fixes/*.py`, `collect_rollouts.py` → `tools/` ;
> `*_compare.csv`, résumés → `results/`. Les versions intermédiaires (`fixes/agent_v1` … `agent_v7`), les sondes, les vidéos et
> les trajectoires ne sont pas versionnées : ce journal garde ce que chacune faisait et ce qu'elle a mesuré.

## 1. Runs complets (VPS, 1–2 oct.)
- Rollouts instrumentés, 1 100 seeds : `analysis/data/champion_full/` (episodes.jsonl + traj/*.npz).
- Benchmark Docker officiel : `analysis/data/bench/bench_champion_full.json`.
- Score sur les practice seeds : **0.8917 en Docker / 0.8900 en process**, et non 0.9151 (l'IC 95 % [0.872, 0.907] exclut 0.9151).
- Process et Docker concordent : 6 désaccords sur 1 100, tous en city (city légèrement non déterministe).

## 2. Notebook mis à jour
- `analysis/diagnostic/sar_diagnostic.ipynb` lit maintenant champion_full + le JSON du benchmark.
- Ajouts : un debrief en tête, des encadrés « Lecture » par section et une section 16 (victimes allongées,
  décollage warehouse, visible mais pas détectée, coût du benchmark, cascade des gains). Ces cellules portent le tag `debrief-1100`.
- PDF : `analysis/diagnostic/pdf/sar_diagnostic_rapport.pdf` (+ `_code.pdf`), à régénérer avec `bash build_pdf.sh`.

## 3. Constats clés
- 89 % de la perte vient des échecs (0.098 pt) ; le temps (0.0065) et la sécurité (0.0055) pèsent peu.
- Par type : mountain 0.780, forest 0.835, warehouse 0.879, village 0.897, city 0.946, open 1.000.
- 109 échecs : vue mais jamais approchée 44, jamais vue 29, crash 22, < 2 m sans prédicat 10, infaisable 3, instable 1.
- Victime allongée : 70 % de succès contre 92 % (n = 77) ; mountain 61 %, forest 55 %.
- « Vue sans approche » : victime petite (21 px contre 47), lointaine (> 11 m) et visible environ 5 s → gisement de +0.039.

## 4. Architecture du champion (`TASK/champion/submission.zip`, extrait dans `analysis/champion_src/`)
- **Pas de RL.** `policy.onnx` = module PyTorch exporté qui contient :
  - un classifieur de type sur la profondeur (`/challenge`) ;
  - un détecteur de victime sur la profondeur (`/detector`) et un MLP pixel→rayon ;
  - des détecteurs RGB (`/victim_rgb*`) ;
  - une navigation **écrite à la main et compilée** (`/nav`, ~1 700 constantes, Clip/ReduceMin sur la profondeur).
- `memory_tensor` (65 floats) = mémoire d'automate.
- `drone_agent.py` = wrapper Python : classifieur `typenet.bin`, tour de waypoints, `_RgbPrimary` (mountain),
  `_Escape` (piège au décollage), évitement (forest/city), route `champ` (village/warehouse = ONNX seul).
- Action : `[dir_x, dir_y, dir_z, speed, yaw, rgb_request]`, vitesse réelle = speed × 3 m/s.

## 5. Correctifs : `analysis/fixes/agent_v1/` (zip : `agent_v1.zip`)
Coupure par variable d'environnement : `KT_TKO=0` (fix 1), `KT_TDESC=0` (fix 2).

**Fix 2 : descente au-dessus du verrou RGB (mountain).**
- Cause : la descente était plafonnée à 0.45 m/s, et le minuteur « dud » (6 s) abandonnait une cible juste pendant la descente.
- Correctif : descente proportionnelle jusqu'à 1.5 m/s ; le minuteur ne compte qu'à l'altitude de hover.
- Résultat : **4/4 récupérées** (1000040 → 0.976, 1000238 → 1.000, 1000245 → 1.000, 1000616 → 0.742).
  15 succès avec verrou testés : aucun changement (score et temps identiques).

**Fix 1 : décollage warehouse.**
- Cause : départ sous une étagère (obstacle à 0.33 m au-dessus, confirmé par rayTest) ; l'escape existait mais était désactivé en warehouse.
- Correctif : au premier pas, si z ≈ 0.191 et que les lignes 0:16 de la profondeur ont un min ≤ 0.6 m, on force l'escape.
- Résultat : **3/7 récupérées** (1000193, 1000448, 1000900 → 0.90).
  - Indétectables (étagère hors champ) : 1000059, 1000950.
  - Dos au mur : 1000397, 1000875. Recul et balayage long testés, tous deux pires → abandonnés.
- 2 témoins inchangés.

**Total : +6.35 pts sur 18 seeds ≈ +0.0058 sur 1 100** (extrapolé : le code modifié n'est appelé qu'en
warehouse au départ et en mountain avec verrou).
Données : `analysis/data/v1_targeted/`, `analysis/data/v1_reg_mtn/`.

## 6. Vidéos (`analysis/fixes/videos/`, script `run_videos.sh`, rendu dans WSL)
- Prêtes : `wh1000193_champion/…chase.mp4` (crash à 0.3 s) et `wh1000193_corrige/…chase.mp4` (réussite).
- Mountain 1000245 : **non produites** (rendu tué par la limite de 30 min ; environ 45 min pour la version corrigée,
  ~2 h 30 pour le champion). Les fichiers `.tmp_*.mp4` sont incomplets et à supprimer.
- OpenCV headless a été installé dans le venv WSL (nécessaire à `generate_video.py`). Il faut utiliser des chemins absolus
  pour `--out` et `--model` (bug de chemin relatif).

## 7. À reprendre
1. Mountain 1000245 : figure avant/après tirée des trajectoires (recommandé), ou vidéo de la version corrigée seule (~45 min, en détaché).
2. Ajouter une section « Correctifs » au notebook (tableau avant/après + trajectoires).
3. Prochains chantiers : mémoriser les détections fugaces (+0.039 au plus), recherche en mountain / taux de verrou RGB, crashs forest.
4. Validation finale éventuelle de agent_v1 en Docker sur le VPS (demander avant tout run long).
5. Vérifier d'où vient le 0.9151 publié.

## 8. Suite (après-midi du 3 oct.) : verrou RGB assoupli, agent_v2
- Notebook : section 17 « Correctifs » (tableau avant/après, figures 17_fix2_mountain_1000245 et 17_fix1_warehouse_1000193, 17.3 agent_v2). PDF **non régénéré** (`bash build_pdf.sh`).
- Sonde RGB (`agent_v2` + `KT_RPLOG`, `probe_rgb/`, `analyse_probe_rgb.py`, `probe_rgb/frames.csv`) : 12 échecs mountain « vue sans approche ».
  27 détections à p ≥ 0.5 : 24 vraies. Filtre h > 1 : 17 vraies + 2 fausses rejetées. Profondeur > 20 m : 3 vraies rejetées. 3 acceptées mais isolées.
  Les 3 fausses ont une profondeur au pixel ≥ 26.8 m.
- `agent_v2/` (`agent_v2.zip`) = v1 + `RP_HMAX` 2.0, `RP_RANGE_MAX` 25 m, verrou sur une seule détection à p ≥ 0.95 (`KT_RP_HMAX`, `KT_RP_RANGE`, `KT_RP_ONE`, `KT_RP_N`).
- Validation (`data/v2_rgb`, `compare_v2.py`, `v2_compare.csv`) : 12 échecs + 8 témoins → 4 échecs récupérés (226, 363, 640, 536), 0 témoin dégradé, 1000614 : 0.57 → 1.00. **+4.10 pts / 20 seeds (≥ +0.0037 sur 1 100).**
  Restent : verrou trop tardif (303, 728, 790), aucune détection (083, 221, 310, 422, 592).
- Vues caméra : `capture_views.py`, `plot_views.py` → `views/views_<seed>.png` (mountain 1000245, forest 1000890, city 1000254).
- Doc récapitulatif « Agent SAR : comment marche chaque brique » : https://claude.ai/code/artifact/5527eda7-7c73-4a0d-9af3-cd5269d43c05
- WSL : tmux + keep-alive (6 workers mountain → OOM, utiliser 4). Les keep-alive en arrière-plan ont été tués par Claude Code (mémoire du PC basse).
- À faire : run des 184 seeds mountain avec agent_v2 (long, demander d'abord) ; mémoire des détections profondeur (exposer `/detector`) ; budget RGB ; PDF.

## 9. Nuit du 3 au 4 oct. : recherche, verrou profondeur, agent_v5 (tout sur le VPS)
- **Validation agent_v2** (arrêtée à 137/184 seeds mountain, `data/v2_full`) : 0.789 → 0.817, +3.73 pts, 8 récupérées, 4 dégradées (faux verrous RGB).
- **Plafond d'AGL en mountain** (`agent_v3` → v3b → **v3c**) : les échecs « jamais vue » volaient 20–40 m au-dessus de la victime (départ sur un sommet, victime en vallée ; l'indice n'a pas de z).
  v3c = AGL ≤ 12 m, uniquement dans le cercle de 30 m, descente ≤ 1.5 m/s, dégagement 6 m, coupé si verrou RGB. 11 échecs + 7 témoins : 7 récupérés, 0 régression due au plafond (`data/v3*_agl`).
  v3 (sans restriction de zone) faisait chuter 2 témoins (descente pendant le transit, basculement 1000953).
- **Sonde du détecteur profondeur interne** (`policy_det.onnx` = sorties `/detector/Concat` [score, px, py, d] et `pixel_to_camera_mlp` [x,y,z caméra] exposées ; `probe_det/`, `analyse_probe_det.py`) :
  sur 12 échecs forest/village/warehouse « vue sans approche », le détecteur est juste (< 3 m) dans 11 seeds, souvent tôt : la nav ne s'engage pas.
  Repère : q = cam + right*x − up*y + fwd*z.
- **Verrou profondeur** (`agent_v4` → **v4b**) : score ≥ 0.95, d ≤ 30, point ≤ 30 m de l'indice, 2 détections à 3 m, rien avant 12 s, pas de surcharge si obstacle < 3 m devant.
  12 échecs + 14 témoins : 5 récupérés, 0 régression (`data/v4b_dl`). v4 (p 0.9, dès 1 s) : faux verrou au décollage warehouse + 2 collisions forest.
- **Hauteur du hover** : le point RGB verrouillé est au niveau du sol (dz_surf ≈ 0 ± 0.4) → hover à 3.2 m = 1.4–1.8 m au-dessus d'une victime debout (sous la fenêtre). Point profondeur : mi-hauteur.
  Remplacé par un **balayage vertical** (5.0 → 2.4 m au-dessus du point, 0.45 m/s, départ à la hauteur courante, pas de coupure RP_TERM_MAX pendant le balayage).
- **agent_v5** (`fixes/agent_v5`, zip) = v2 + v3c + v4b + balayage. Validation lancée le 4 oct. 00:48 : `--per-type 60` (360 seeds), `data/v5_final` sur le VPS, log `/root/results/v5_final.log`, comparaison `compare_final.py`.
- Attention : `pkill -f collect_rollouts` dans une commande ssh tue la commande elle-même (le motif apparaît dans sa ligne).
- **Validation agent_v5 (360 seeds, 60/type, `data/v5_final`, `v5_final_compare.csv`)** : 0.8983 → 0.8959 (−0.002, IC [−0.025, +0.020]).
  mountain +0.042 (6 récup., 3 perdues), forest −0.034, village −0.018, warehouse −0.003, city −0.002, open 0.
  Le verrou profondeur s'active dans 213/300 épisodes non mountain : forest −2.0 pts, village −1.1, warehouse −0.2 (faux verrous, temps perdu).
  Pertes v5 : faux verrous profondeur (forest 1000060, 1000020), basculement mountain 1000124 (plafond AGL contre une montée de la nav).
- **agent_v5b/v5c** : reconfirmation du verrou profondeur (revu dans les 5 s si > 8 m), limite 15 s ; plafond AGL inactif si la nav monte (vz ≥ 0.3).
  v5b (seuil 4 m) a rejeté un vrai verrou (1000284) → v5c seuil 8 m.
- **agent_v6** = v5c avec le verrou profondeur désactivé par défaut (`KT_DL=1` pour le réactiver). Validation 360 seeds lancée le 4 oct. 02:46 (`data/v6_final`, tmux `v6f`).
- **Validation agent_v6 (360 seeds, `data/v6_final`, `v6_final_compare.csv`)** : **0.8983 → 0.9133 (+0.015, IC [−0.003, +0.034])**, 9 récupérées, 3 perdues.
  mountain +0.058, warehouse +0.015, city +0.017 (city peu fiable), forest/village/open strictement identiques. Hors city : ≈ +0.012 sur 1 100.
  **agent_v6 = version retenue** (`fixes/agent_v6`, `agent_v6.zip`). Reste à faire : benchmark Docker officiel sur 1 100 seeds.

## 10. 4 oct. matin : mémoire des détections (agent_v7) — DERNIER CHANGEMENT avant le test final
- `agent_v7` = agent_v6 + verrou profondeur réactivé en **mémoire différée** : le verrou se forme dès 12 s (score ≥ 0.95, 2 détections à 3 m,
  point ≤ 30 m de l'indice) mais reste inerte ; il ne prend la main qu'à **35 s** (`KT_DL_ACT=1750`) si la victime n'est pas confirmée,
  et seulement en **forest/village/warehouse** (`KT_DL_TYPES`). À l'activation : limite de temps et reconfirmation repartent de zéro.
  Avant activation, une nouvelle paire cohérente ailleurs remplace la cible. Succès après 35 s chez le champion : forest 17 %, village 12 %, warehouse 7 %.
- Validation lancée le 4 oct. 11:02 sur le VPS (tmux `v7`, script `/root/v7_final.sh`), survit à la fermeture de session :
  1. `data/v7_final` : 180 seeds (60 premières forest/village/warehouse, mêmes seeds que v6_final) ;
  2. `data/v7_tgt` : les 12 échecs sondés ;
  3. résumé automatique `/root/results/v7_summary.txt` (v7 vs v6 vs champion, IC bootstrap, liste des seeds qui changent). FINISHED en fin de `/root/results/v7_final.log`.
- Décision à prendre à la reprise : si v7 − v6 ≥ 0 sans perte nette → agent_v7 pour le test final ; sinon agent_v6.
- **Point d'étape v7 (61/180 seeds)** : −2.78 pts vs v6, 0 récupération. Pertes = les 3 mêmes qu'avec v5 (forest 1000020, 1000060, village 1000099),
  succès lents du champion (confirmés à 50.7, 51.3 et 56.9 s) : la mémoire active à 35 s les détourne vers une fausse cible.
  **Recommandation provisoire : agent_v6 pour le test final**, à confirmer avec le résumé complet `/root/results/v7_summary.txt`.
- **v7 à 128/180 seeds : −3.02 pts vs v6, 0 récupération** (3 succès lents perdus + 6 succès ralentis). **Décision : agent_v6 pour le test final.**
  Le run se termine seul ; le résumé complet (dont les 12 échecs sondés) sera dans `/root/results/v7_summary.txt`.
- **Résultat final v7 (180/180 + 12 sondés, terminé le 4 oct., `data/v7_summary.txt`)** :
  forest −0.033, village −0.018, warehouse −0.004 ; **total v7 − v6 = −0.018, IC [−0.038, −0.003]** (significativement négatif).
  0 récupération sur les 180, 3 perdues (1000020, 1000060, 1000099 → INFEASIBLE), 11 succès ralentis.
  Échecs sondés : 2/12 récupérés (warehouse 1000540 → 0.665, 1001044 → 0.722), les 10 autres toujours en échec.
  Le gain sur les sondés ne compense pas les pertes sur les seeds courantes. **Décision confirmée : agent_v6 pour le test final.**
  (Le script de résumé avait perdu ses guillemets `["score"]` via le heredoc ssh ; corrigé et relancé à la main.)

## 11. 4–5 oct. : benchmark officiel agent_v6 (1 100 seeds) — ÉCHEC DU HARNAIS, pas de résultat
- Lancé le 4 oct. 14:47 (tmux `v6bench`, `/root/v6_bench.sh`, même commande que le champion, 6 workers).
- **Arrêté le 4 oct. 20:52 (exit=1) après ~990/1 100 seeds, sans JSON** (`/root/results/bench_v6_full.log`, ~3 600 lignes) :
  `RuntimeError: Host worker 1 failed on batch 355: AssertionError: daemonic processes are not allowed to have children`.
- Cause (harnais, pas l'agent) : la calibration de vitesse de l'hôte expire après 6 h (`_CALIBRATION_MAX_AGE_SEC = 6 * 3600`,
  `swarm/validator/docker/docker_evaluator_parts/batch.py:76`). À 6 h 05, le worker 1 a voulu recalibrer
  (`_ensure_host_speed_factor` → `_run_host_baseline_calibration` → `proc.start()`) depuis un processus daemon → assertion → tout le run s'arrête.
  Le benchmark du champion (2 oct.) a duré 5 h 18 et n'a donc jamais atteint l'expiration ; v6 était un peu plus lent (6 h 05).
- Les scores par seed sont perdus : le log ne contient que les durées (`Worker N complete | batch …`), et l'exception remonte
  par `asyncio.run` avant l'affectation de `results`, donc le chemin « partial results » de `swarm/benchmark/engine_parts/entry.py` ne s'exécute pas.
- Pour relancer : soit `_CALIBRATION_MAX_AGE_SEC` à 24 h sur le VPS (ne touche que la normalisation des temps, pas les scores),
  soit découper en 2 runs de 550 seeds (< 6 h chacun). Run de ~6 h → **à valider avec l'utilisateur avant de le lancer**.
- Le chiffre de référence reste la validation 360 seeds de la section 9 : **+0.015 (IC [−0.003, +0.034]) vs champion**.

## 12. 5 oct. : RÉSULTAT FINAL — benchmark officiel agent_v6, 1 100 seeds
- Relancé le 5 oct. 08:10 avec `_CALIBRATION_MAX_AGE_SEC = 24 * 3600` sur le VPS (original : `/root/batch.py.orig`), terminé à 14:46 (exit=0, 1 100/1 100 `seed_done`).
- Fichiers : `data/bench/bench_v6_full.json` (+ log), comparaison appariée `fixes/compare_bench.py` → `data/bench_v6_vs_champion.txt`.
- **Champion 0.8917 → agent_v6 0.9038 : +0.0121, IC 95 % [+0.0039, +0.0207] (significatif), succès 993 → 1 008 (+21 récupérés, −6 perdus).**

| type | champion | v6 | Δ | IC 95 % | récup. / perdus |
|---|---|---|---|---|---|
| city | 0.9561 | 0.9520 | −0.004 | [−0.022, +0.012] | +1 / −2 |
| open | 1.0000 | 1.0000 | 0 | — | 0 / 0 |
| mountain | 0.7799 | 0.8423 | **+0.062** | [+0.019, +0.108] | +17 / −4 |
| village | 0.8970 | 0.8970 | 0 | — | 0 / 0 |
| warehouse | 0.8791 | 0.8935 | +0.014 | [−0.000, +0.034] | +3 / 0 |
| forest | 0.8351 | 0.8351 | 0 | — | 0 / 0 |

- Conforme à la prévision de la validation 360 seeds (≈ +0.012 hors city) : le gain vient de mountain (correctifs descente RGB / plafond AGL) et warehouse (décollage).
  Village/forest/open strictement identiques (contournement du champion conservé).
- Seeds perdus : mountain 1000155, 1000205, 1000559, 1000759 ; city 1000638, 1000729 (bruit city déjà vu en validation).
