# Task 4 independent specification review

Date: 2026-11-12.

## Verdict

**PASS — 0 Critical, 0 Important, 0 Minor unresolved specification findings.**

The reviewed evidence satisfies the bounded development/model and engineering-pilot requirements of Task 4. This verdict is confined to Task 4 specification compliance. Parent adoption, formal preparation review and exact-package prelaunch approval remain separate gates; this review does not authorize formal collection or establish efficacy.

Reviewed final `task-4-report.md` raw SHA256: `afe6d9b7e1117a9b31df7dbdfda92fb2bf67d1d5bfda5449091b0663b4cb1bed`.

## Basis and review method

Read `task-4-brief.md` and `task-4-extracted.md` first, followed by the adopted numerical design, final Task 4 report, scoped collector/layout/comparator reviews, quality review, corrected engineering v003 package, development diagnostic package and complete development collection, pilot freeze/validation/raw attempts, parent recorder results and retained seed-audit allocation. The governing design hash is `ce4ac80694795fd1c58125eb930be98d998bd148ea004cda1a9c169e43debcc2`.

This is an independent specification verdict. The quality review supplied useful evidence locations; its PASS was not used as a substitute for checking the requirements. Independent local inspection included raw SHA256 verification, JSON inventory checks, all pilot decision/frame counts, counter arithmetic, forecast-start eligibility/urgency, benefit/full-target arithmetic and burst/cooldown checks. The existing audit helper was read to establish what its retained replay and engineering results mean. It was not executed.

No simulations, fitting, tests, SSH, source changes, commits, deletion or evidence overwrite were performed. The historical 17,008-file seed scan was not repeated. Its retained literal audit was inspected, including overlap arithmetic against the retained reserved IDs. Only this review file was created.

## Requirement assessment

### 1. Bounded inventory, pairing and preservation — satisfied

- Separate frozen train/calibration/development protocols and strict validations contain exactly 96/48/32 valid cells. Only HOLD and geometry occur, equally represented. Each regime has the specified 12/6/4 distinct seed pairs, reused only across the declared static/gradual contexts and two policies within its split.
- The corrected pilot has exactly six scenario pairs and eight roles per pair: two gradual plus one separate static pair per regime, 48 valid attempt-0 cells. Each role appears six times. All raw pilot results are `ok`, with direct simulator exit 0, completed process evidence and captured stdout/stderr. Independently checked all 960 listed raw artifact hashes and both captured streams per cell.
- The retained audited inventory allocates 24 train, 12 calibration, eight development, six pilot and 48 formal independent pairs. Neither layout nor radio IDs overlap across phases or the retained historical reserved IDs. The historical audit records 17,008 files, 469 reserved layout IDs and 249 reserved radio IDs. No historical rescan or fresh seed search was conducted during this review.
- The failed first pilot remains disclosed separately: four legacy successes, four v2 collector errors and 40 unstarted cells. The eight attempted simulations are additional preserved integration evidence, not additional seed allocations or successful completion of that package. The reviewed collector repair returned to Task 3, and the complete pilot uses a fresh freeze/output root. The repair review justifies retaining the original 176 development episodes and one model fit because their HOLD/geometry branches and fitting inputs were unchanged.
- Parent recorder results retained in `remote-evidence/pilot-recorders-v001.tar.gz` independently report direct exit 0 for corrected pilot preparation, execution and v003 audit, with wall durations 5.2552084923, 670.0388004780 and 417.8560652733 seconds. Queue/global concurrency evidence is retained and reviewed separately from artifact completion; no second worker pool is claimed.

### 2. Frozen identities and fixed scenario — satisfied

Independently verified all 135 pilot freeze file hashes and all 127 executed-source hashes against the frozen copies. The engineering manifest's 11 file entries match actual bytes. Its bound identities match the pilot package:

| Identity | Raw SHA256 |
| --- | --- |
| Pilot freeze | `7d17ed501f996d94aa62cc7b7fb4ff86b14cbe7981b19262a277a837c67ed77e` |
| Pilot protocol | `4c6fd1f6c91d06f68b43d2fb8929bffa8ca9d30ade5cd9708fee5f088c4cab0c` |
| Fitted service model | `719816f80b9df51c5544422522265e50bd454768869523036a8b8f4b1c657c10` |
| Exact reference model | `3b74301ccb0e659ba6d2b593250decabd0f317386adfa4cc5d5b4e5174bacae8` |
| Method | `c977477ffe7f6704a289ebd0cd050e836c796d130f2a4232d7996ba806765a1f` |
| Pilot source manifest | `a5680954a0d5cd83aa864a7d951a786d4f19ae0c4b2cb3a588841f6abe3cb35b` |

Frozen scenario configurations retain 240 m range, n4/.15 and n5/.10, warmup 10 s, 240 intervals, the declared bounds/altitude, source/destination, 8 m/s simulator limit, 400 m relay budget and 5 m separation. Mission start is 30 for gradual and 1000 for static, with mission speed 1. Runtime evidence identifies the retained ns-3.45 binary (`fda432c0fd7af79e474b565a7639f92da75b446f283e4c38b9cf9532b3528347`), Python/environment and existing dependencies. Historical 243 m results are not substituted for new 240 m fixtures.

Comparing development and corrected pilot executed-source manifests shows only `src/control/service_v2/study.py` changed, matching the separately reviewed collector adapter. Model bytes remain identical. Features, labels, fitting, controller, method, simulator and old reference bytes did not change between those packages. No numerical retuning is evidenced.

### 3. One fitted model and development diagnostics — satisfied

The model retains the exact 25-feature contract, H10/H20 heads, adopted target, training-only normalization, deterministic Newton solver settings, L2 .01/.001, unpenalized intercept, independent calibration HOLD mapping and monotone horizon projection. The diagnostic binds complete training/calibration trace/source metadata, reproduces normalization exactly and reports strict loading success. Calibration geometry remains diagnostic only. No development or pilot refit occurred.

| Horizon | Train positive / negative rows | Calibration HOLD positive / negative rows | Train groups per class | Calibration groups per class |
| --- | ---: | ---: | ---: | ---: |
| H10 | 794 / 17,040 | 187 / 4,364 | 24 / 24 | 12 / 12 |
| H20 | 1,661 / 15,564 | 428 / 3,970 | 24 / 24 | 12 / 12 |

These exceed the fixed 50-row/six-group train and 20-row/three-group calibration gates. Both train and calibration fits converge at both horizons with finite reported results. Eligibility, right censoring and zero-eligible episode lists remain explicit; rows are not presented as independent experimental replicates.

Heldout HOLD diagnosis preserves Brier/log loss, prevalence, reliability/PR diagnostics and warning/event accounting. The report visibly retains weak fixed-threshold H10 recall (2/156), H20 recall (99/306), 12 detected events among 27 eligible events, three false warnings and five censored warnings. Weak accuracy and negative efficacy are correctly not converted into engineering rejection or permission to tune.

The original asynchronous development diagnostic lacks its direct process exit. The final report states that limitation and distinguishes completion/manifest evidence from exit 0. Its bound collection validations and retained model diagnostics remain reviewable; no missing audit exit is silently manufactured or replaced by another fit/run.

### 4. Accounting, causal execution and motion — satisfied

Independent raw checks count exactly 11,520 pilot decisions and 11,568 frames (240/241 per cell). For all 48 cells, cumulative-minus-prior equals each service window count; terminal-minus-last-live equals recorded late counts. Application PDR equals terminal sink arrivals/offers. Initialization is preserved but does not become a full traffic window. The terminal boundary remains 251 s in the retained accounting, with no artificial drain. Sink and FlowMonitor streams stay distinct; all actual pilot differences happen to be zero.

The corrected retained audit reports zero engineering issues/processing failures, 48/48 exact prefix replays and 6/6 exact v1 factory parity checks. Source inspection confirms replay consumes current/past frames before examining the following execution frame, and reference parity constructs the accepted controller/default method directly. V003 removes only top-level timing from decision comparison and normalizes finite JSON representation; action equality remains exact. Preserved v002 evidence still has its false gate and 7,652 mismatches. The scoped comparator correction does not rewrite it.

Independent decision checks find all 5,760 v2 decisions free of logged observation/feature/prediction errors; all 5,640 populated risk pairs are finite, bounded and nested. All 53 forecast starts satisfy their role's arming/eligibility support and H <= tau + 3 + 2 urgency. No-forecast does not start forecasts; the online-state ablation is assessed under its explicitly different support rule.

All 96 gated starts independently pass positive proposed motion, final same-chain maximum-hop gain >=2 m, efficiency >=.10 and full-target affordability. Hop maxima/gain arithmetic agrees. All 96 retain a hop lengthening by more than 1 m; this is allowed by the adopted minimax rule and shows that the discarded per-hop veto was not reintroduced. The controller tolerance remains 1e-8, distinct from the 1e-5 m serialized displacement evidence tolerance.

Independent nonzero-action sequences have maximum gated burst length three and minimum intervening zero-action gap five. The removed gates are correctly not imposed on the no-motion-gate ablation. The retained audit checks fixed chain/targets, measured budgets, bounds/separation, preceding safety and actual compensation, including command 239. Maximum position residual is `9.999894245993346e-10` m; travel/position residual is `1.4141986064980465e-09` m. Maximum relay travel is 394.819576761 m. No simulator safety rejection is recorded. Main v2 has zero measured movement or forecast starts before arming.

The analysis retains 420 mechanism traces with planned/observed paths, per-hop/rollout diagnostics, reasons and following five-second evidence. Planned geometry is not called an imposed OLSR route or a guaranteed service improvement.

### 5. Healthy-start fixture and reported interpretation — satisfied

All six 240 m HOLD fixtures first arm at step 7, satisfying the step-20 gate. Each offers 15,000 packets; received counts are 15,000, 13,216, 13,858, 15,000, 11,874 and 13,855 in the report's stable order. This establishes the pilot fixture and does not become a formal-seed exclusion rule.

The principal pilot contrast is v2 minus no-forecast, with two gradual pairs per regime, all eight roles kept together, 10,000 bootstrap replicates and RNG seed 20261112. Equal-weight pooled difference is 0.0098333333 PDR and 44.49999999 additional relay metres. The negative n5 principal pair and negative n4 versus HOLD pair remain visible. Static main-v2 cases retain PDR 1 and zero motion. Engineering descriptions are not promoted to formal inference, packet-level counterfactual pairing or universal superiority.

Feature/predictor/planner/decision median/p95/max and per-cell process times are retained. The report separates 11,520 controller timings from 5,760 v2 component records and states that synchronous pipe decisions block simulator progression. The 3 s routing allowance remains a planning hypothesis. No measured radio/control delay, distributed implementation, energy or endurance claim is made.

### 6. Formal boundary — satisfied; authorization remains pending

The candidate retains `formal_outcomes_read=false`, `formal_authorized=false`, `parent_adoption=false` and no efficacy requirement. Local `runs/formal`, `frozen/formal` and `formal-analysis` do not exist. Formal seed inventory was inspected only as audited allocation, not outcome data. No formal outcome was read in this review.

The independent review fields in the immutable candidate remain historical `pending` values. This review and the separate quality review can be incorporated through a fresh parent adoption record binding their exact bytes and accepted identities. They do not mutate that candidate or bypass the subsequent formal-package/prelaunch gates.
