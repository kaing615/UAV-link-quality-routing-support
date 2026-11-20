# Task 5 formal prelaunch independent review

Date: 2026-11-12.

## Verdict

**PASS — 0 Critical, 0 Important, 0 Minor unresolved findings.**

This accepts the identities and planned inventory of the actual prepared formal package. It is separate from parent launch approval. It does not authorize queue execution, assess formal outcomes or establish efficacy. No formal outcomes were read; the local `runs/formal` directory does not exist.

## Scope and governing gates

Reviewed the actual `frozen/formal/` package, parent adoption and both accepted Task 4 records, `evidence/formal-prepare-v004/`, the direct preparation recorder, and `evidence/queue-dry-run-formal-v004/queue-plan.json`. Applied `task-4-helper-fix-1-evidence/GATE-SCHEMAS.md`, with the separately reviewed v004 runtime/path relocation. The relocated formal seed input is the exact path/hash named by actual parent adoption and preparation, under `source-runtime-v003`; the schema document's historical v002/v003 invocation examples do not override that accepted relocation.

Verification was limited to local read-only hashes, package/record contents and planned allocation/argv comparisons. No simulation, test, fitting, SSH, code edit or outcome inspection was performed. No historical seed scan was repeated. Only this review file was created.

## Exact formal identity and adoption binding

The following is the actual formal identity object for the accepted prelaunch record's `bindings.formal` field:

```json
{
  "phase": "formal",
  "protocol_sha256": "99590e2d9f2fe945d04af941071c7b28d491f611e1f79c54b31e3d348489676b",
  "freeze_sha256": "193a9984a836a04f07422d0f3657e380f3e8fbc1fef65ceae91219bc83f5c5e6",
  "source_manifest_sha256": "a5680954a0d5cd83aa864a7d951a786d4f19ae0c4b2cb3a588841f6abe3cb35b",
  "spec_sha256": "ce4ac80694795fd1c58125eb930be98d998bd148ea004cda1a9c169e43debcc2",
  "method_sha256": "c977477ffe7f6704a289ebd0cd050e836c796d130f2a4232d7996ba806765a1f",
  "new_model_sha256": "719816f80b9df51c5544422522265e50bd454768869523036a8b8f4b1c657c10",
  "reference_model_sha256": "3b74301ccb0e659ba6d2b593250decabd0f317386adfa4cc5d5b4e5174bacae8",
  "seed_plan_sha256": "fdfaee568be703ffe0f5bfe8fbb75724cea599caa7069e80acc9082e7a98748f",
  "inherited_pilot_freeze_sha256": "7d17ed501f996d94aa62cc7b7fb4ff86b14cbe7981b19262a277a837c67ed77e"
}
```

Required `bindings.adoption_sha256` is `497711997c61a94d8590e8dbbde2e4931b24f0af50acc137fc192e0b83385679`, the raw hash of `engineering/task4-parent-adoption-v001.json`.

### Package and inheritance checks

- All 135 formal freeze entries match raw local files. All 127 executed-source hashes match formal frozen source copies.
- Formal source manifest, design, method, fitted model, reference model and runtime files are byte-identical to the adopted pilot package. Formal inheritance equals the adopted pilot freeze. The newly computed formal protocol/freeze identities match `verified-formal-inheritance.json` exactly.
- `runtime.json` raw SHA256 is `cbe86f177b354001d174f78d4d3ffc2e45723e8184d92731de82b218aa835767`, identical to pilot. Its Python executable/environment, package identities, ns-3.45 launcher and binary hash remain unchanged. The retained actual frozen-runtime probe ran from formal frozen source with the declared thread/library environment and reports direct exit 0. No fresh runtime was substituted during this review.
- All 174 regular files in `remote-evidence/formal-prelaunch-v001.tar.gz` match their extracted local bytes. Archive SHA256 is `70dc03de0253fd9b6b69ef1ba613fc72a8b1d062cf0daa01e2080563e4e91427`.

### Parent adoption and review checks

- Adoption has the exact required top-level fields, `adopted=true`, `adopted_by=parent`, and matching pilot identity/artifact bindings.
- All ten adoption references match the raw local counterpart hashes: pilot protocol, development/engineering manifests, four prior validation files, both acceptance records and the formal input seed plan. Previous validation files were hash-checked only for this review; no prior or formal outcome analysis was reopened.
- Specification acceptance record SHA256 is `04d47b6e5aa964e1eec900abd3936bd3039ac16d2a7a0c34779c85a49ff94bac`; quality acceptance record SHA256 is `3c04d6b2b1931b5d9b6de3d0e98d95fe3d06534c6b9a116e13b0f74defc49766`. Both are accepted Task 4 records with zero Critical/Important findings and bindings equal to adoption.
- Their raw accepted review-report hashes match the actual reports: specification `d64e787cf3a97b57e8838fb92a55c24bab78909279a533aaf4b2c01060ee1683`, quality `974d26898aedf9d5b658e975127a74c40d0e7b7a2c246bd98d54083a9eed7062`.
- Preparation's `verified-task4-adoption.json` binds this exact adoption hash, references, pilot identities and 96/48/32/48 prerequisite counts. A completion marker is not used as adoption or process-exit evidence.

## Formal allocation and queue checks

The frozen `seed-plan.json` is byte-identical to the adopted relocated formal plan. Its literal formal records exactly generate the protocol's scenario/seed tuples. There are 48 independent pairs, 384 unique cells and exactly eight roles per scenario:

| Regime | Gradual pairs / cells | Separate static pairs / cells |
| --- | ---: | ---: |
| n4 | 20 / 160 | 4 / 32 |
| n5 | 20 / 160 | 4 / 32 |
| Total | 40 / 320 | 8 / 64 |

Each role appears 48 times. Formal layout and radio IDs independently have no overlap with consumed train/calibration/development/pilot IDs or the retained historical reserved sets. Static/gradual and regime allocations use distinct pairs. This check used the previously audited literal inventory, with no new seed search or history rescan.

The fixed 240 m scenarios, policy ordering and source/config identities remain those of the adopted method. The protocol permits only one identical-cell technical/infrastructure retry, with separately named attempts; it does not authorize seed replacement.

The dry-run queue has exactly 384 commands with unique declared cell IDs. Every command exactly instantiates the frozen `run_argv` template using attempt 0 and the exclusive `runs/formal/<cell-id>/attempt-0` destination. It names formal frozen source as working directory and the exact formal protocol hash. `dry_run=true`, `max_workers=4`, `automatic_retry=false`, and `stop_pending_on_wrapper_error=true`. The retained dry-run invocation includes the single parent global-simulations lock and exits 0. These are planned argv arrays, not evidence of executed episodes.

Queue-plan raw SHA256 is `18260ff23133a8cb350129d6ef2b2888eabda8f016624c465d72be58d7b0b17b`. Direct formal preparation result SHA256 is `c281bbe12c6405c0cd6578e8fb378cba8aa3e189c1a897f1ab5a615da6c02c80`; it records direct exit 0 and wall duration 13.7872345448 s. Its child freeze, actual frozen-runtime probe and queue dry-run also report direct exit 0 in the adopted environment.

## Remaining parent gate

Parent may publish acceptance of this prelaunch review using scope `formal-prelaunch`, review kind `prelaunch`, the exact formal identity object above and the exact adoption hash. A distinct `formal-launch-approval-v001.json` is still required. Immediately before dispatch, the parent phase helper must re-read adoption, the actual strict-loaded package, accepted prelaunch record and exact-package launch approval and retain `verified-formal-launch.json`. This PASS does not replace that step.
