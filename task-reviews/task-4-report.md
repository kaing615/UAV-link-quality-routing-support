# Task 4: development and engineering pilot

Status: evidence complete; independent specification and quality reviews and parent adoption are pending. This report does not authorize formal preparation or execution.

## Scope and fixed identities

Task 4 collected 96 training, 48 calibration, 32 heldout development and 48 corrected engineering pilot cells. All 224 cells passed strict collection validation. The model was fitted once; development and pilot inspection did not change its thresholds or parameters.

| Item | Raw SHA256 |
| --- | --- |
| Adopted design | `ce4ac80694795fd1c58125eb930be98d998bd148ea004cda1a9c169e43debcc2` |
| Fitted predictor | `719816f80b9df51c5544422522265e50bd454768869523036a8b8f4b1c657c10` |
| Corrected pilot freeze | `7d17ed501f996d94aa62cc7b7fb4ff86b14cbe7981b19262a277a837c67ed77e` |
| Development diagnostic manifest | `27a2acbfad316b645176e6ad9b169006b43abe5648932dff606677e1e1f05be4` |
| Engineering v003 manifest | `ac19d1acbb166717d349677978706ac3130bc0b72e7ee5ace1a5ea5d47099bbf` |
| Corrected audit helper | `83747b84e9f852c8879028b290096349f6ed6bc489090228b980f0ee1a741f19` |

The remote study root is `/home/sac/uav-service/2026-11-12-service-prevention-v2`. Pilot used `source-runtime-v003`, frozen alias `pilot-integration-v003`, and evidence suffix `v004`. The simulator, existing reference controller/model, radio settings and historical reports remain unchanged. Task 3's reviewed adapter repair adds collector aliases while retaining native decisions and measured displacement.

## Inventory and process evidence

| Phase | Planned cells | Strict-valid cells |
| --- | ---: | ---: |
| Train | 96 | 96 |
| Calibration | 48 | 48 |
| Development | 32 | 32 |
| Corrected pilot | 48 | 48 |

The pilot has six independent scenario seed pairs, eight roles per pair, 48 attempt-0 results with status `ok` and direct exit 0. Every corrected pilot episode has 240 decisions and 241 frames. No identical-cell retry was needed in this package. The complete cell, attempt, seed, counter, movement, endpoint and hash inventories are in `engineering.json`, `engineering-audit.json`, the strict validation and raw run artifacts.

The direct parent recorders report exit 0 for pilot preparation, queue execution and audit v003. Their wall times are 5.2552 s, 670.0388 s and 417.8561 s respectively. `completed.json` alone only records artifact generation; it does not establish a direct process exit or adopt a gate. Queue concurrency is bounded to four simulations.

The original asynchronous `development-diagnostic-v001` audit did not retain its direct process exit. Its completion marker and verified manifest establish artifact completion, and its bound validations establish the development collection results. This report does not assign exit 0 to that original audit. No simulation or fit was repeated to replace this missing process record.

The first failed pilot remains separate: four legacy successes, four v2 collector errors (`KeyError: 'intervention_started'`) and 40 unstarted cells. The four v2 errors are retained as technical failures. This adds eight simulator attempts to the 608 planned valid episodes without adding seed IDs. The repaired pilot used a new package and output root.

## Predictor diagnostics

The development model gates passed at H10 and H20, including class support, convergence, finiteness, split and calibration provenance. The heldout HOLD diagnostic retains 3,019 eligible rows, eight seed pairs and horizon censoring. Calibration geometry was diagnostic only.

| Horizon | Censored rows | Positives | Brier | PR-AUC | Recall at fixed threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 s | 93 | 156 | 0.0389 | 0.4241 | 2/156 = 0.0128 |
| 20 s | 204 | 306 | 0.0484 | 0.7873 | 99/306 = 0.3235 |

H10 recall is weak. The warning analysis records 12 detected events out of 27 events in its event risk set, 18 evaluable warnings, five censored warnings and three false warnings. Those diagnostics do not estimate the effect of moving UAVs.

## Engineering checks

Audit v003 examined all 48 cells and reports zero issues and no processing failures. All 48 exact-prefix replays passed; all six v1 reference cells passed factory parity. Sink and FlowMonitor terminal received counters match in all 48 cells; application PDR still uses sink arrivals divided by offered packets.

The largest position prediction residual is `9.999894245993346e-10` m and the largest travel-versus-position residual is `1.4141986064980465e-09` m. No simulator safety rejection occurred. Maximum used per-relay budget fraction across roles is 0.98705. The bounded v2 roles have bursts of at most three command intervals. Audit checks retain cooldown, full-target budget, boundary benefit, bounds/separation, nested risk, causal prefix, path mismatch and minimax/per-hop evidence. There are no forecast starts before arming, and the main v2 policy has no measured movement before arming. Forecast and restoration starts remain distinguished in the raw audit.

All six HOLD fixtures first armed at step 7, before the required step 20. The offered count is 15,000 in each fixture; received counts in stable order are 15,000, 13,216, 13,858, 15,000, 11,874 and 13,855. These fixtures establish pilot readiness and do not filter formal seeds.

Audit v002 is preserved with 7,652 `stored_action_or_decision_replay_mismatch` issues. It compared Python tuple values with JSON lists before normalization. Scoped review accepted the v003 finite JSON comparison and retained a regression that rejects real field changes. V003 reprocessed the same pilot raw data without simulation reruns, refitting or research-source changes.

## Pilot outcomes

Each gradual mean below uses only two scenario pairs per regime. They are engineering descriptions; formal inference remains pending.

| Policy | n4 PDR | n4 total relay m | n5 PDR | n5 total relay m |
| --- | ---: | ---: | ---: | ---: |
| HOLD | 0.90247 | 0.00 | 0.85763 | 0.00 |
| Reactive | 0.88490 | 141.71 | 0.85443 | 190.15 |
| Geometry | 0.90187 | 144.82 | 0.90667 | 191.14 |
| v1 reference | 0.87567 | 721.88 | 0.94273 | 809.61 |
| v2 | 0.92840 | 140.00 | 0.95890 | 85.11 |
| v2 without forecast | 0.91050 | 97.00 | 0.95713 | 39.11 |
| v2 without online service state | 0.91357 | 108.00 | 0.93300 | 85.11 |
| v2 without motion gate | 0.88313 | 142.40 | 0.90440 | 191.05 |

For the prespecified main comparison, v2 minus v2 without forecast, the equal-weight pooled PDR difference is 0.009833 (0.9833 percentage points), with 44.50 m additional total relay movement. The pilot block bootstrap interval is [0.001367, 0.018300] for PDR and [22.00, 67.00] m for movement. Only four gradual scenario pairs contribute. In one n5 pair the PDR difference is negative (-0.005067); v2 also falls below HOLD in one n4 pair (-0.003533). These negative points remain in the analysis. Both static main-v2 cells have PDR 1.0 and zero relay movement.

Shared scenario seeds do not imply identical packet-level randomness after policy actions diverge. The paired analysis resamples scenario blocks and keeps all eight role outcomes together. Movement is distance, not measured energy or endurance.

## Timing and interpretation

Across the stored decision durations, median is 0.007152 s, p95 is 0.020254 s and maximum is 0.095783 s. The v2 component timing field has 5,760 entries: median decision 0.013442 s, p95 0.023615 s and maximum 0.095718 s. These populations differ and are retained separately. Per-cell process times are in the full inventory.

The synchronous pipe bridge blocks simulator progression during centralized decisions. Processing duration therefore does not measure radio/control delay, distributed deployment latency or real UAV response time. The 3 s routing allowance remains a planning assumption. No novelty, guaranteed superiority, latency, energy saving or deployment claim follows from this pilot.

## Retained packages and formal boundary

All archive hashes below match the remote and local bytes.

| Archive in `remote-evidence/` | SHA256 |
| --- | --- |
| `development-complete-v003.tar.gz` | `d7c31a8d633be792a7c09e8d8761830dc929d185c6ee1cfd2ccfcde768a0b36a` |
| `pilot-engineering-v003.tar.gz` | `68a067314f854d27b61faee9baea3eb50a97789cc2c9300655d7fdc0d0e28170` |
| `pilot-integration-v003-complete.tar.gz` | `d9872bbff422e690f7e5d00b2ab0f9c86bb9496be67392705f3f02468a297433` |
| `pilot-recorders-v001.tar.gz` | `6044a150ed30fa4033ba118c3d641442c348218e107fe99c2bf23d54485260d4` |
| `pilot-frozen-v001.tar.gz` | `3f5592900448122335f97382f716e0337e4048a333c00b6656f9eedb273c3473` |

Local evidence lives under `reports/uav_service_prevention_v2_2026-11-12/`; the corrected engineering package is `remote-evidence/pilot-engineering-v003/`. The failed pilot and false v002 audit retain their original directories. Draft reports v001/v002 remain historical drafts.

The engineering candidate reports complete development/model/engineering checks, all six HOLD fixtures armed by20, `formal_outcomes_read=false`, `formal_authorized=false` and `parent_adoption=false`. Two accepted independent Task 4 reviews and a separate parent adoption record must precede formal preparation. The prepared formal package then requires its own hash-only prelaunch review and exact-package parent launch approval. No formal outcome has been used in this report.
