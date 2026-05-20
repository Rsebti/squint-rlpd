# Handoff — RLPD on Squint for SO-101 grasp

> **Pour la nouvelle session Claude Code sur la machine Linux (RTX 5090).**
> Ce fichier est un dump de contexte pour que tu démarres froid sans rien
> demander à l'utilisateur. Lis-le en entier avant d'agir.

---

## 1. Qui je suis et préférences de travail

- Utilisateur : Rayane, étudiant Master ETH Zürich — cours **Robot Learning**, projet 3.
- Communication **en français** dans le chat ; **terminologie technique et code en anglais**.
- Préférences :
  - Exemples numériques concrets avant les formules.
  - Étapes progressives, pas de saut.
  - OK avec les commandes read-only et clean-up locaux ; **demander avant** de toucher le vrai robot, de lancer du paid GPU, ou de push vers HF.
  - Code sur Linux ici (donc bash, pas PowerShell).

---

## 2. Le projet en 30 secondes

**Objectif** : Eval 2 du projet 3 — pick-and-place SO-101 avec **2 blocs colorés + bowl_xyz + target_color donnés en input**. RL obligatoire. 5 rollouts × 10 pts.

**Stratégie choisie (2026-05-20, à respecter)** :
1. **Le grasp** est appris par **RLPD** (Ball et al. 2023, ICML) dans la sim **ManiSkill3**, en utilisant le repo **Squint** comme scaffold (sim, robot SO-101, envs, training pipeline).
2. **Le place** se fait après le grasp par un **controller FK/IK scripté**, pas appris.
3. **Données offline** pour RLPD = vraies démos téléop du robot, dataset HF `Rsebti/projet3_demos_v1` (39 episodes, 12 277 frames, 30 FPS).

**À ne PAS suggérer** (déjà tried, plateau ou abandonné) :
- ❌ Isaac Lab from scratch (2 semaines de PPO state-based → plateau 2-7%, le code archivé sur la branche `archive/eval2-attempts-pre-reset` du repo principal)
- ❌ BC warmstart + DAPG
- ❌ Pick-and-place end-to-end appris par RL (le user a explicitement splité grasp RL + place FK/IK)
- ❌ Magic-attach grasp (workaround sale du past)

---

## 3. État du repo squint/ — ce qui a été fait

Le repo est un fork de `fedecomi04/squint` (Squint = SAC visuel + C51 + ensemble Q sur ManiSkill3 SO-101). On a ajouté **4 fichiers nouveaux** et laissé tout le reste intact.

### Fichiers ajoutés (à reviewer en premier)

| Fichier | Lignes | Rôle |
|---|---|---|
| `train_rlpd.py` | 1284 | Fork de `train_squint.py` avec ~5 edits chirurgicaux : Critic scalaire (C51 stripped), `num_q=10`, sample-then-min sur subset, MSE TD(0), 2ème ReplayBuffer offline + symmetric sampling 50/50 |
| `rlpd_utils.py` | 707 | Loader des démos offline. Deux sources : (a) HF LeRobot dataset (notre cas), (b) rollouts d'un checkpoint Squint pré-entraîné. CLI : `python rlpd_utils.py from_lerobot/from_ckpt ...` |
| `parse_check_rlpd.py` | 61 | Sanity check pur-syntax. Pas de dépendances. À runner partout en premier. |
| `test_rlpd_smoke.py` | 137 | Smoke fonctionnel : Critic forward, sample-then-min, round-trip buffer offline. CPU-friendly mais a besoin de torch + tensordict installés. |
| `RLPD.md` | 219 | Doc complète : deltas vs Squint, workflow, tableau de conversion LeRobot→ManiSkill, caveats. **Lis-la après ce MEMORY.md.** |

### Fichiers Squint d'origine — à ne PAS modifier
- `train_squint.py` — baseline SAC, gardé intact pour pouvoir comparer
- `envs/place.py` — seul env file, paramétrable via flags (`pick_only_reward`, `n_distractors`, `use_real_bowl`)
- `envs/base_random_env.py` — domain randomization + camera config
- `envs/robot/so101.py` — robot config (`pd_joint_target_delta_pos` : ±0.05 rad arm, ±0.20 rad gripper)
- `deploy.py` — déploiement réel via Sim2RealEnv + LeRobot. **Marche tel quel sur un checkpoint RLPD** (le critic n'est pas chargé au deploy).

### Statut de validation au moment du handoff
- ✅ `parse_check_rlpd.py` passe sur le laptop (CPU-only, pas de NVIDIA)
- ❌ `test_rlpd_smoke.py` **PAS encore tourné** (laptop n'a pas tensordict)
- ❌ Aucun training lancé
- ❌ Aucune démo décodée
- ❌ Pas de deploy

C'est ton job de tout valider end-to-end sur cette machine.

---

## 4. Spec rapide de RLPD (ce qu'on a implémenté)

Le but de RLPD = exploiter des démos offline + de la collection online dans un SAC, sans pretrain offline. Deltas par rapport à un SAC classique :

1. **LayerNorm** dans chaque hidden du critic (Squint le faisait déjà)
2. **Ensemble Q large** (`num_q=10` par défaut)
3. **Sample-then-min** : à chaque pas, on tire un subset aléatoire de `subset_size=2` parmi les 10 Q-nets target, on prend le min sur ce subset pour le Bellman target (style REDQ)
4. **Symmetric sampling** : 50% du batch online + 50% offline à chaque grad step
5. **Pas de BC term**, pas de pretrain offline, pas de IQL/CQL penalty — vanilla SAC + ces 4 ingrédients

Hyperparams par défaut dans `train_rlpd.py::Args` : `num_q=10`, `subset_size=2`, `offline_ratio=0.5`, `offline_reward_mode="sparse"`. Le reste vient de Squint (gamma=0.9, tau=0.01, num_updates=256, batch_size=512, etc.).

---

## 5. Marche à suivre concrète

Tout dans le répertoire `squint/`. Ordre strict :

### Étape A — Setup env (1 fois, ~10 min)
```bash
conda env create -f environment.yaml   # crée l'env "squint" avec torch cu128, maniskill, lerobot
conda activate squint
```
⚠️ Si Blackwell (RTX 5070/5090, sm_120) → torch 2.7.1+cu128 doit marcher. En cas de wheel mismatch, fallback nightly cu128.

### Étape B — Validation code (30 sec)
```bash
python parse_check_rlpd.py    # doit dire "All files parse cleanly."
python test_rlpd_smoke.py     # doit dire "All smoke tests passed."
```
Si l'un échoue, **NE PAS CONTINUER**. Debug d'abord. Probablement install torch/tensordict incomplète.

### Étape C — Décoder les démos v1 (1 fois, ~5 min)
```bash
mkdir -p offline_bundles
python rlpd_utils.py from_lerobot \
    --repo_id=Rsebti/projet3_demos_v1 \
    --out=offline_bundles/projet3_v1_80x144.pt
```
🚨 **Le moment crucial.** À la fin, le loader imprime des stats. Inspecter :

| Si tu vois… | Action |
|---|---|
| `gripper qpos median ≈ 0.5–1.5 rad` | OK, continue |
| `gripper qpos median ≈ 50–90` | Le gripper est déjà en rad. Relancer avec `--no-gripper_in_degrees` |
| `% clipping arm > 5%` | FPS ≠ 30 ou unité fausse. Stop, investiguer. |
| `fraction grasped` < 5% ou > 95% | Tune `--grasp_threshold_rad` (essayer 0.3 / 0.7 / 1.0) |

**Mentionne EXPLICITEMENT ces stats à l'utilisateur après la première exécution** — c'est lui qui sait ce qui est attendu vu qu'il a enregistré les démos.

### Étape D — Baseline RLPD pure-online (validation algo, 15-30 min sur 5090)
```bash
python train_rlpd.py --env_id=SO101LiftCube-v1 --pick_only_reward --num_envs=256 --track
```
But : vérifier que l'**implémentation RLPD** (ensemble 10-Q, sample-then-min, MSE TD) converge sur la tâche la plus simple. Sans démos, RLPD ≈ REDQ. `eval/success_at_end` devrait monter à 30-50% après ~500k steps. Si ça stagne à 0 → bug algo, pas bug démos.

### Étape E — Le vrai run RLPD avec démos
```bash
python train_rlpd.py \
    --env_id=SO101LiftCube-v1 \
    --pick_only_reward \
    --no-env_domain_randomization \
    --offline_path=offline_bundles/projet3_v1_80x144.pt \
    --offline_ratio=0.5 \
    --num_envs=256 \
    --track
```
À monitorer dans wandb :
- `eval/success_at_end` doit monter PLUS VITE qu'en étape D
- `q_max` ne doit PAS exploser (>100)
- `critic_loss` doit décroître

---

## 6. Pièges connus et décisions à anticiper

1. **Q-value divergence** (`q_max > 100` dans wandb). Cause #1 : les 18 dims privilégiées du state (`item_pose`, `bin_pose`, `tcp_to_*`) sont à zéro dans les démos (cf. `_load_lerobot_dataset` docstring). Solutions par ordre de coût :
   - `--offline_ratio=0.25` (plus de online, moins de offline)
   - `--subset_size=5` (target moins conservatrice)
   - Implémenter sim-replay pour reconstruire le vrai state (TODO commenté dans le code, demande GPU + sim)

2. **`--no-env_domain_randomization`** au premier run avec démos. Les démos sont en conditions fixes, mélanger avec une DR sim agressive envoie des signaux contradictoires. Remettre la DR APRÈS que ça marche.

3. **`--num_envs=256`** = défaut safe. Si `nvidia-smi` montre >50% VRAM libre, monter à 512 ou 1024 pour aller plus vite.

4. **Wandb** : `wandb login` avant le premier `--track`. Sinon `--no-track` pour désactiver.

5. **Sauvegardes** : `train_rlpd.py` sauve `runs/<run_name>/ckpt.pt` et `_best.pt` à chaque eval. Reprise via `--checkpoint=path/to/ckpt.pt` ou `--checkpoint=wandb`.

---

## 7. Quand la policy marche (en sim) — déploiement réel

Le but final = déployer sur le **vrai SO-101**. Phases :

### Phase 0 — Prérequis hardware
- **Calib robot** : `lerobot-calibrate --robot.type=so101_follower --robot.port=<COM>`. Fichiers dans `~/.cache/huggingface/lerobot/calibration/`. **Sans calib propre, les actions sim→real sont fausses.**
- **Alignment camera wrist** (le plus critique pour sim-to-real) : `python deploy_utils/tune_camera.py`. Trackbars jusqu'à ce que la position du gripper sim et réelle coïncident → `p` pour print params → copier dans `envs/base_random_env.py` ligne ~497.
- Conditions scène : table sombre ou retrain avec `b8ada9_overlay.png` (présent dans le repo). Lumière indirecte stable, pas de soleil direct.

### Phase 1 — Dry run safe-mode
```bash
python deploy.py --checkpoint=runs/<rlpd_run>/ckpt_best.pt --env_id=SO101LiftCube-v1 --no-continuous_eval
```
Demande input avant chaque step. Vérifier que les actions sont raisonnables sur 5-10 pas avant de passer en continu.

### Phase 2 — Mesurer le taux de grasp réel
Lancer 10-20 grasps consécutifs. Repositionner le cube à la main entre épisodes. Mesurer :
- >60% → marche, passer à phase 3
- 30-60% → fragile, identifier le mode d'échec
- <30% → sim-to-real gap trop grand, retour Phase 0

### Phase 3 — Ajouter le PLACE FK/IK (À CODER, pas dans Squint)
Squint deploy.py fait tourner la policy en continu. Notre projet a besoin de :
```
Boucle main:
  1. Tant que not grasped: action = rlpd_policy(obs)        ← RL
  2. Une fois grasped:    action = scripted_place_to_bowl()  ← FK/IK
  3. Si gripper ouvert ET cube dans bowl: SUCCESS, reset
```
FK déjà dispo dans `so101_fk.py::tcp_pos(q)`. IK numérique via `nudge_arm_joints(q, delta_xyz)` (déjà codé, Newton 4-itérations). Bowl xyz : pour la première démo, passer en CLI (`--bowl_xyz=x,y,z`) ; vision-based plus tard.

Ce wrapper `deploy_rlpd_then_place.py` est ~150-200 lignes. À écrire quand on a un checkpoint RLPD qui marche en sim.

---

## 8. Tokens et secrets

- HF token : stocké côté Polytechnique sur l'autre machine (`C:\Users\user\Desktop\MA2\tokens.txt`). **Ne jamais commit, paste, ou echo ce token.** Si nécessaire sur cette machine Linux, `huggingface-cli login` interactif.
- Pas de wandb token tracké dans le repo — `wandb login` interactif.

---

## 9. Ressources externes

- **Repo principal du projet** : `Rsebti/robot-learning-project3` (GitHub privé). Contient le `CLAUDE.md` historique avec tout le contexte projet 3, mais **la section "stratégie" parle d'Isaac Lab — c'est dépassé**. Le pivot vers Squint+RLPD documenté ici override.
- **Dataset démos v1** : `Rsebti/projet3_demos_v1` (HF, 100 MB data + 200 MB video).
- **Repo Squint upstream** : https://github.com/fedecomi04/squint (référence). Notre fork est local.
- **Paper RLPD** : Ball et al., "Efficient Online RL with Offline Data", ICML 2023.

---

## 10. Premier message à envoyer à l'utilisateur

Après avoir lu ce fichier, dis-lui :
> "J'ai lu MEMORY.md. Je suis prêt à attaquer l'étape A (setup conda env). Tu veux que je commence ?"

Et **attends sa confirmation** avant de lancer `conda env create` (5-10 min d'install, bouffe de la bande passante). Pour les étapes B et après, propose mais n'attends pas la confirmation si l'utilisateur est en mode auto.
