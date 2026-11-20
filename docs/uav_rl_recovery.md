# UAV connectivity recovery with PPO

This experiment starts with an eight-UAV network split into two measured components. Source UAV 0 and destination UAV 4 stay fixed. A centralized policy chooses normalized 3D velocities for the six relays once per simulated second. ns-3 applies the same speed, travel, boundary, and swept-separation filter to PPO and all baselines. The controller receives global state over an ideal separate control channel; it does not solve control delivery across a partitioned data network. Motion is kinematic, without inertia, wind, obstacles, or a battery power model.

## Build and dependencies

Use Python with `gymnasium`, `stable-baselines3`, `torch`, `numpy`, and `matplotlib` installed, for example `rtk proxy py -m pip install -r requirements-rl.txt`. The recovery build lives at `build/recovery/uav-olsr-dataset`; the existing default simulator is left intact. On Windows, build with `rtk proxy powershell -File scripts/utils/build_recovery_wsl.ps1 -Ns3Prefix /path/to/ns3/build` and pass `--wsl Ubuntu` to the commands below. On Linux, build with `NS3_PREFIX=/opt/ns3 rtk proxy bash scripts/utils/build_recovery.sh`, use `python` instead of `py`, and omit `--wsl`. Both builders write source/header/binary hashes; the training and evaluation commands stop if any declared hash is stale. An ns-3 installation or build with the required Wi-Fi, OLSR, applications, mobility, netanim and flow-monitor modules must already exist.

`RecoveryConfig.steps` counts commanded seconds after the initial observation. The bridge requests `steps+1` ns-3 snapshots, so the default provides exactly 120 actions after observing the initial partition, rather than 119. Real packet-flow metrics cover that controlled interval, starting 0.01 seconds after the initial snapshot.

## Short functional smoke run

From the repository root on Windows:

```powershell
rtk proxy py -m scripts.train.train_recovery_ppo --binary build/recovery/uav-olsr-dataset --wsl Ubuntu --output reports/uav_rl_recovery_smoke/train_seed1 --total-timesteps 1024 --n-steps 64 --batch-size 32 --validation-every 1024 --validation-scenarios 1 --steps 32 --seed 1
rtk proxy py -m scripts.analysis.evaluate_recovery --binary build/recovery/uav-olsr-dataset --wsl Ubuntu --output reports/uav_rl_recovery_smoke/evaluation --model reports/uav_rl_recovery_smoke/train_seed1/best_model.zip --scenarios 3 --steps 32
```

This checks that interaction, parameter updates, checkpoint loading, and paired baseline evaluation work. It is a single-policy-seed exploratory run, not evidence of convergence or superiority over the geometric relay chain. Training writes `run_config.json`, `monitor.csv`, `validation_history.csv`, `model.zip` (final weights), and `best_model.zip`. Checkpoints are ranked by validation recovery rate, then lower failure-capped recovery time, then lower total travel, then higher return. A failed case is charged the full configured horizon for selection; its recorded recovery time remains null. Validation also runs after the last PPO update so final weights can be selected. `best_checkpoint.json` records the winning metrics and scenario table. The manifest records seeds, settings, dependency versions, repository dirty state, hashes of the Python control and CLI sources, simulator source/binary hashes, model hashes, training time, actual interaction steps, and how many policy weights changed. New evaluation summaries carry their own simulator, model, source, and dependency provenance.

## Training schedule and diagnostics

Failed training episodes continue to the full horizon after exhausting the travel budget. ns-3 prevents further movement, while time penalties continue. Success may end a training episode early; validation and testing always continue to the full horizon to measure connectivity durability and packet delivery.

Curriculum is enabled by default. At each episode reset, the completed transition count selects a separation of 1.65 times communication range during the first quarter of the requested training budget, 2.0 times range during the second quarter, and nominal simulator sampling during the final half. Rotation, relay jitter, radio physics, movement constraints, and success criteria remain active. Every initial state must still be measured as disconnected. The stage changes only at reset, so an episode can cross a schedule boundary; `curriculum_transitions.json` records the actual changes. `--no-curriculum` runs nominal layouts throughout for an ablation. Validation and testing always use nominal layouts and reject curriculum overrides. An easier initial separation is a training hypothesis; its benefit must be measured.

`training_episodes.csv` streams completed episode outcomes, stages, seeds, components, travel, budget exhaustion and rejected actions. An unfinished final episode is retained in transition logs but excluded from completed-episode statistics. Validation writes per-scenario `episodes.csv` tables and aggregate success, capped time, travel, exhaustion, rejection and component statistics. SB3 `progress.csv` includes optimizer diagnostics, including the final update. Training outputs must be new or empty to prevent mixing runs.

The small MLP runs on CPU. `--torch-threads` controls its CPU thread count (default 2); ns-3 remains a separate CPU simulator. Throughput with and without validation overhead is recorded. A GPU is not required by this configuration. The default eight nominal validation scenarios provide broader checkpoint coverage than a one-case smoke check, while remaining separate from the testing pool.

## Research run

Use the default 120-second horizon, at least five independent policy seeds, and at least 20 unseen test scenarios. Repeat training with different `--seed` values and separate output directories. The default PPO settings are two 128-unit MLP layers, 1024 rollout steps, batch 64, 10 epochs, learning rate 0.0003, discount 0.99, GAE 0.95, clip 0.2, and entropy coefficient 0.01 on CPU. The CLI records overrides to rollout, batch, epochs, training steps, and validation interval. For example:

This is a substantial compute budget. Benchmark a short run on the target machine and publish the actual step count and wall time for each policy seed before interpreting a larger study.

```powershell
rtk proxy py -m scripts.train.train_recovery_ppo --binary build/recovery/uav-olsr-dataset --wsl Ubuntu --output reports/uav_rl_recovery/train_seed1 --total-timesteps 204800 --seed 1
rtk proxy py -m scripts.analysis.evaluate_recovery --binary build/recovery/uav-olsr-dataset --wsl Ubuntu --output reports/uav_rl_recovery/heldout --model reports/uav_rl_recovery/train_seed1/best_model.zip --model reports/uav_rl_recovery/train_seed2/best_model.zip --model reports/uav_rl_recovery/train_seed3/best_model.zip --model reports/uav_rl_recovery/train_seed4/best_model.zip --model reports/uav_rl_recovery/train_seed5/best_model.zip --scenarios 20
```

Train, validation, and test layout/radio seeds occupy disjoint reserved ranges. Evaluation records both requested and actual reset seeds and rejects a changed seed. A checkpoint is selected with validation episodes, never with held-out test outcomes. Evaluation runs every method on the same held-out layout/radio pairs, uses a deterministic PPO action, and continues through the full horizon after first recovery. The baselines are stationary relays, seeded uniform random velocity, and a geometry-based chain ordered by relay projection onto the source–destination axis. The random baseline's generator is derived from both layout and radio seeds with NumPy `SeedSequence`. All actions go through ns-3's motion filter.

Evaluation writes `episodes.csv` and `episodes.json` with one row per policy/scenario, `summary.json`, per-episode `trajectory.jsonl`, and `learning_recovery.png`. Technical simulator errors are separate from policy failures. Time to recovery is null for unsuccessful episodes (right censored); the report includes the count of censored episodes and labels any mean recovery time as success-only. The summary averages scenarios within each policy seed, then policy seeds equally. A two-level bootstrap interval is reported only with multiple policy seeds and a complete paired scenario grid; the same scenario IDs are resampled across policy seeds. If a technical error or missing/duplicate case breaks pairing, `summary.json` marks the paired comparison invalid and omits intervals. PDR and UDP delay come from ns-3's final `closed_loop_metrics.csv` after normal process completion, with transmitted, received, and offered packet counts retained; no snapshot-derived PDR substitutes for it.

The experiment may show PPO underperforming the geometric chain. Report measured recovery, time, connectivity durability, travel, rejection count, PDR, delay, inference time, and training budget together. A short smoke run must retain its exploratory label even if its reward is higher.
