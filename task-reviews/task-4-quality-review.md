# Task 4 independent quality and reproducibility review

Date: 2026-11-12.

## Verdict and findings

**PASS — 0 Critical, 0 Important, 0 Minor unresolved findings.**

The corrected 48-cell engineering pilot has reviewable raw attempts, matching identities, retained process evidence and consistent accounting/motion diagnostics. The Task 4 report accurately distinguishes engineering evidence, descriptive pilot outcomes and pending formal work. This is the quality/reproducibility review; independent specification review and parent adoption remain separate. This review does not authorize formal preparation or launch.

Reviewed final `task-4-report.md` raw SHA256: `afe6d9b7e1117a9b31df7dbdfda92fb2bf67d1d5bfda5449091b0663b4cb1bed`.

One documentation omission was returned during review and corrected before this verdict: the report now explicitly states that the original asynchronous development audit's direct exit was not retained. Completion and verified outputs are not called exit 0. The report preserves this limitation without a simulation rerun or refit.

## Evidence and identity checks

Reviewed the Task 4 brief/design, prior integration/helper/comparator reviews, final Task 4 report, corrected engineering v003 package, frozen `pilot-integration-v003` package, strict validation, raw pilot attempts, queue revisions, three direct parent recorder results, development/model diagnostics, and retained failed-pilot/v002 evidence. Source inspection was confined to the existing reviewed audit logic; no research code was changed or executed.

Independently hashed the local archives and verified the report's values:

| Archive | Raw SHA256 |
| --- | --- |
| `development-complete-v003.tar.gz` | `d7c31a8d633be792a7c09e8d8761830dc929d185c6ee1cfd2ccfcde768a0b36a` |
| `pilot-engineering-v003.tar.gz` | `68a067314f854d27b61faee9baea3eb50a97789cc2c9300655d7fdc0d0e28170` |
| `pilot-integration-v003-complete.tar.gz` | `d9872bbff422e690f7e5d00b2ab0f9c86bb9496be67392705f3f02468a297433` |
| `pilot-recorders-v001.tar.gz` | `6044a150ed30fa4033ba118c3d641442c348218e107fe99c2bf23d54485260d4` |
| `pilot-frozen-v001.tar.gz` | `3f5592900448122335f97382f716e0337e4048a333c00b6656f9eedb273c3473` |

The remote/local transfer comparison is retained parent evidence; this reviewer independently verified local bytes without SSH.

- All 135 entries in the pilot freeze file inventory match their raw hashes; all 127 executed-source hashes in `source-manifest.json` match the frozen source copies.
- The engineering input identities match actual pilot freeze, protocol, source manifest, method, fitted model, reference model, adopted design and collection bytes. In particular model `719816f80b9df51c5544422522265e50bd454768869523036a8b8f4b1c657c10`, reference `3b74301ccb0e659ba6d2b593250decabd0f317386adfa4cc5d5b4e5174bacae8`, design `ce4ac80694795fd1c58125eb930be98d998bd148ea004cda1a9c169e43debcc2`, method `c977477ffe7f6704a289ebd0cd050e836c796d130f2a4232d7996ba806765a1f` and freeze `7d17ed501f996d94aa62cc7b7fb4ff86b14cbe7981b19262a277a837c67ed77e` agree.
- All 27 manifest entries across corrected engineering/development diagnostic packages, all 96 pilot attempt result/artifact identity references, all 960 raw attempt artifact hashes and all 96 captured simulator streams match. The corrected engineering manifest itself hashes to `ac19d1acbb166717d349677978706ac3130bc0b72e7ee5ace1a5ea5d47099bbf`.

## Inventory, execution and accounting

Strict pilot validation is complete: 48 planned/48 valid cells, one attempt-0 per cell, six scenario seed pairs with eight roles each. Raw attempts contain exactly 11,520 decisions and 11,568 frames: 240 decisions/241 frames per cell. Every raw simulator process record reports started/completed, direct exit 0, no error, and captured stdout/stderr with matching hashes. All 48 queue wrapper exits are separately recorded as 0. Across 97 retained queue status revisions, maximum recorded running count is four.

Parent recorders independently retain direct exit 0 for pilot prepare, execute and corrected audit, with respective wall durations 5.2552084923 s, 670.0388004780 s and 417.8560652733 s. The audit invocation binds reviewed helper hash `83747b84e9f852c8879028b290096349f6ed6bc489090228b980f0ee1a741f19`, the intended Python/environment, `source-runtime-v003`, unchanged pilot alias and fresh output root. Artifact completion is kept distinct from process exit and parent adoption.

Independent raw checks find exact cumulative-minus-prior equals window counts for each service counter, and exact terminal-minus-last-frame equals the recorded late counts. Terminal FlowMonitor reception agrees with each terminal CSV. All 48 actual sink-minus-FlowMonitor differences are zero; these streams are still recorded separately. Application PDR equals terminal sink arrivals/offers. No artificial flush or packet-level pairing is claimed.

Development diagnostic evidence binds 96 train/48 calibration/32 development cells and the unchanged fitted model. Both horizons report converged train/calibration fits, exact training-only normalization, and class/group gates. H10 train positive/negative counts are 794/17,040 and calibration HOLD 187/4,364; H20 counts are 1,661/15,564 and 428/3,970. Each class has 24 training and 12 calibration seed groups. The report retains weak H10 recall and the original development-audit direct-exit limitation.

## Motion, causality and mechanism evidence

The corrected audit reports zero engineering/processing issues, 48/48 exact-prefix replays and 6/6 exact v1 factory parity checks. Its reviewed finite JSON comparator preserves exact actions and rejects genuine semantic changes; the prior regression is retained.

Independent arithmetic checks of all raw frames/decisions confirm:

- No measured bounds/separation/budget violation. Maximum relay travel is 394.819576761 m; minimum sampled pair separation is 9.2156856324 m. No simulator safety rejection is recorded. Maximum measured one-interval v2 displacement is 2.00000000393 m, within the declared evidence tolerance.
- The maximum stored prediction/execution residual is `9.999894245993346e-10` m and travel/position residual `1.4141986064980465e-09` m, both far below the unchanged `1e-5` m evidence tolerance. Controller acceptance tolerance remains `1e-8`.
- Gated v2 moving runs have maximum length three; minimum intervening complete zero intervals is five. Chain/targets stay fixed during each burst. The no-motion-gate ablation is correctly excluded from these removed gates.
- All 5,760 v2 decisions are free of logged observation/feature/prediction errors. All 5,640 populated risk pairs are finite, bounded and nested H10 <= H20. All 53 forecast starts satisfy their role's causal arming/history/eligibility support. The prefix replay consumes current/past frames; following frames are used for execution verification after the decision. Main v2 has zero measured movement or forecast starts before arming.
- All 96 gated starts pass positive motion, final maximum-hop gain >=2 m, efficiency >=.10 and full-target affordability. Independently checked hop maxima, signed hop gains and final gain arithmetic agree. All 96 include a hop lengthening by more than 1 m, retained as diagnostics under the adopted minimax rule. No discarded 1 m veto is reintroduced.

The analysis retains 420 mechanism traces, including 379 v2 starts with observed routes, planned chains, per-hop/rollout data and following five-second frames/decisions. It records 349 v2 planned/observed-route mismatches; it does not present a geometric plan as an imposed OLSR route or guaranteed delivery benefit. These are descriptive mechanism traces.

All six HOLD cases arm at step 7 with the stated received/offered counts. The fixture is an engineering gate and is not used to exclude formal seeds.

## Report support, timing and preservation

The report's gradual policy means, movement totals, main pooled difference/intervals and explicit negative paired points match the analysis. Its two gradual pairs per regime and descriptive pilot status are clear. The principal pooled PDR change is 0.0098333333 with 44.49999999 m additional movement; the negative n5 main-comparison point and negative n4 versus HOLD point remain visible. Static main-v2 cases have PDR 1 and zero motion. Bootstrap blocks retain all eight roles; the report does not claim packet independence, energy, guaranteed superiority or formal efficacy.

Stored timing populations remain separate: 11,520 controller durations versus 5,760 v2 component records, plus 48 process times. Median/p95/max figures in the report match. The synchronous blocked-simulator assumption and heuristic 3 s allowance are explicit. The independently generated phase analysis and engineering analysis differ in exactly 40 scalar leaves, all newly measured shadow-decision timing fields; scientific outputs agree.

The failed first pilot remains technical/integration evidence, with eight attempted cells disclosed separately. V002 remains unchanged: its 11 manifest entries verify, its bound pilot inputs equal v003's, and its issue inventory contains exactly 7,652 replay mismatches with a false gate. V003 reran only the audit on the same raw pilot; it does not rewrite the old result.

The final candidate still states parent adoption false, formal authorization false and independent reviews pending. This quality PASS can be incorporated only through the separate parent-owned adoption procedure.

No simulations, fitting, SSH, code edits, commits, deletions or evidence overwrites were performed. Independent verification used read-only local hashing, JSON inspection and arithmetic; the only new output is this review.
