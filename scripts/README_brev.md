# Autonomous Brev/MassedCompute L40S training

3-stage training chain (eval1 → eval2, eval3 warm-started from eval1) running
unattended on a single L40S 48 GB VM. Each stage: 20 M steps @ 2048 envs,
episode length 150 (5 s at 30 Hz). Total wall-time ~5–9 h, cost ~$5–10.

## One-time setup (on your laptop)
You need a wandb account and your API key from <https://wandb.ai/authorize>.

## Launching the run (on the VM)

SSH into the freshly-provisioned L40S, then:

```bash
curl -L https://raw.githubusercontent.com/fedecomi04/squint/master/scripts/brev_run_all.sh -o run_all.sh

export WANDB_API_KEY=...        # required
export WANDB_ENTITY=fedecomi04  # or your wandb username/team

tmux new -s squint
bash run_all.sh                 # provisions + runs all 3 stages
# Ctrl-b d to detach. SSH out. Reconnect later with `tmux attach -t squint`.
```

When all 3 stages finish, the script prints a "you can now delete the VM"
banner. Checkpoints live in wandb under `$WANDB_ENTITY/maniskill-so101` as
artifacts `model_eval1_SO101PlaceCube-v1_1`, `..._eval2_..._2`,
`..._eval3_..._3`. Pull them later with `python -c "import wandb; ..."` or
the helper `scripts/brev_pull_ckpt.sh eval1`.

## Knobs

All overridable via env vars before `bash run_all.sh`:

| Var | Default | Notes |
|---|---|---|
| `ENV_ID` | `SO101PlaceCube-v1` | swap to `SO101LiftCube-v1`, etc. |
| `TOTAL_TIMESTEPS` | `20000000` | per stage |
| `EP_STEPS` | `150` | 5 s @ 30 Hz (100 = 3.3 s, too short for pick+place) |
| `NUM_ENVS` | `2048` | drop to 1024 if you hit PhysX broad-phase OOM |
| `WANDB_PROJECT` | `maniskill-so101` | |

## Recovery

If the VM is destroyed mid-chain (e.g. after eval1 but before eval2),
spin up a new VM, re-export the env vars, then:

```bash
curl -L https://raw.githubusercontent.com/fedecomi04/squint/master/scripts/brev_run_all.sh -o run_all.sh
bash scripts/brev_setup.sh                    # re-provision
bash scripts/brev_pull_ckpt.sh eval1          # re-download eval1 ckpt from wandb
bash scripts/brev_run_evals.sh eval2 eval3    # run only the warm-start stages
```

## What each script does

- `brev_setup.sh` — installs miniforge, clones repo, creates the `squint`
  conda env, logs into wandb (+ HF if `HF_TOKEN` set). Idempotent.
- `brev_run_evals.sh` — runs one or more of `eval1 eval2 eval3` sequentially.
  Each stage's last checkpoint uploads to wandb at end-of-run.
- `brev_run_all.sh` — wraps `setup + run_evals` in a single command.
- `brev_pull_ckpt.sh <exp>` — downloads any past stage's wandb artifact to
  `runs/<exp>/ckpt.pt`. Use for resumption.
