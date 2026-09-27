# Interim S2/S3 acceptance update — 27 September 2026

This is **not a completion certificate**. It preserves all 71 requirements from the prepared matrix and changes only five statuses to `proven_with_scope`: `s2-01`, `s2-03`, `s2-04`, `s3-01`, `s3-02`. S4 is running at the root task observation; its changing outputs were not read. The final PDF, publication and terminal audit remain pending. S5 is excluded. Push-T E2 remains a valid negative stop, with conditional E3–E5 unactivated.

## Completed evidence

| Requirement | Accepted evidence and scope |
|---|---|
| S2 recipe/completion | 214,780 updates = ten epochs of 21,478; 18,034,978 parameters; 4,544 training and 93 validation source episodes. Existing final CPU audit checks exports and recovery/config/data/split/source coherence. No new tensor audit here. |
| S2 durable checkpoint | Private final revision `9e05478658484d0f26ea740fead9e0ab7625367e`, exact final tag and 22 non-receipt payload files verified by the existing audit. Remote content was not downloaded/restored. |
| S2 logging/figures | 10,739 saved training records; 2,139 recovered W&B records matched; eleven validation steps present and final prediction loss 0.0039440737455151975 matched. Actual PNG/PDF training curves are hash-bound, with recorded root visual review. They do not invent a local validation time-series. |
| S2 timing | Recovered report interval 10,231.99154 s; managed attempt 10,247.25198 s; original-start-to-recovery-finish calendar span 62,245.25885 s (17.29035 h). At least 2,400 logged updates repeated. These are different scopes; none is total project GPU cost. |
| S3 qualification | Current-model development gate passed for **health only** on 192 paired branches, 64 roots and 32 source episodes. MLP R²: height 0.985428, pitch 0.924221, speed 0.907845. Speed nevertheless failed rule specificity (15/32 = 0.46875 versus 0.8). Health recall 67/71 and specificity 107/121 pass. |
| S3 durable probes/figures | Private revision `cb1560d7b02b4fbe6585b3c0c325a78d56b337e6`, nine payload files; actual R²/horizon figures copy exact saved gate/report data. Source-byte hashes agree, with provenance limits below. |

## Required terminal work

1. Audit **terminal S4**, binding the workflow/config, health-only model/probe/data identities, output hashes, rows and unchanged gate. A valid development stop means `go=false`, `diagnostic_stop` and `final_test_evaluated=false`; it does not complete acquisition. A complete branch requires both budgets (128/512), all three seeds (0/1/2), both arms, **12** checkpoint receipts, baseline/no-update exports, paired truth/censor identities, charged costs, retention/error summaries and honest undefined/interval semantics.
2. Verify the actual private S4 bank upload and all activated model uploads with the prepared remote checker, after the local evidence audit. Later results/figures are not automatically covered by runtime uploads.
3. Produce actual terminal S4 presentation; complete and independently inspect the non-draft PDF at A4 size with exact charts/media and numerical/provenance citations.
4. Preserve the actual versions of used helpers and their results, then reconcile source/config/audit/model/data/report/media inventories. Resolve the **existing exact E2 archive approval** before that payload upload; no retry or alternate route was attempted here. Complete and verify approved GitHub/HF publication.
5. Write a fresh final requirement verdict and delivery index only after activated experiments, presentation and publication are all resolved.

## Limits that remain

The fixed 128-root Push-T test design has no recorded pre-test variance-based sizing rationale; no retrospective power or resizing claim is justified. Historical S0 source/seeds/splits/weights and parts of S1 source provenance remain unestablished. S1 alignment is post hoc and bounded. E2 fallbacks, undefined FSA, differing acceptance, incomplete goal retention and unknown historical failed-run cost remain explicit.

S3 lacks per-example probe predictions and full branch tapes/truth needed to recompute R², horizon errors or physical replay; its 19,200 branch steps are count-derived. Current/reviewed source hashes do **not** replace a missing per-run executed S3 source snapshot. W&B is mutable; tags/privacy are observation-time metadata. S2 final report and later S2/S3 figures require separate archive coverage. This update does not rerun models, simulation, tensors, datasets, tests or network checks, and does not claim final A4 review of S3.

## Traceability

The JSON carries all 71 original requirements, exact plan references, five status transitions, updated row-specific evidence hashes, unchanged historical requirements, detailed terminal branch requirements and a before/after hash index of bounded read inputs. Its original `evidence_index` is explicitly the prior snapshot, not a new verification of old files.

Base matrix SHA256: `76dc57c0d725429d0780631291da470bce5b1f91942f47b30afc1f5405d915ac`.

Updated JSON SHA256: `0f32e1a66fb0f66628e42e3e7a6c86ef64d80658af71f31942885d749578c072`.

Status counts: `{"conditional_not_activated": 3, "excluded_by_user": 1, "historical_diagnostic_only": 1, "implemented_not_executed": 4, "pending_execution_or_final_audit": 13, "proven_with_scope": 44, "provenance_limited": 3, "valid_negative_stop": 2}`.

New authoritative evidence:

- `docs/mainPlan/results/continuation-v2/walker_s2_final_local_audit_20260927.json` — `93e1b378570b8670499b76316d747f30da769ca3ca8ec42f10fe7f2c765924de`.
- `docs/mainPlan/results/continuation-v2/walker_checkpoint_214780_upload.json` — `21a73425d840487f0491517dad0dece0d8760f23af9c7fb617b2b32d49b21b74`.
- `docs/mainPlan/results/continuation-v2/walker_s2_final_wandb_audit_20260927.json` — `af188000839dc37675ef259d5e183030b96453d8e04bda5a30b0f25a604c130c`.
- `docs/mainPlan/results/continuation-v2/walker_s2_final_timing_20260927.json` — `c07777cd00ae952d66069899275adaa839c56e40548d08b7e802416949b11313`.
- `docs/mainPlan/results/s2/walker2d-lewm-a-recovery-20260927-1-summary/manifest.json` — `3532654496300fd7561c249f3c4dee64164430f842535e7c72223adb411338de`.
- `docs/mainPlan/results/continuation-v2/walker_s2_learning_curves_visual_audit_20260927.json` — `2d22bab4c5ebe9c3e551c51c4757a152b5e897a1484ee8593d541ab111f95c7f`.
- `docs/mainPlan/results/continuation-v2/walker_s3_final_local_audit_20260927.json` — `636a617b5e8131d5674c617e02fb5a16ffb274b2b8a0ef0429982ce44872080a`.
- `docs/mainPlan/results/continuation-v2/walker_s3_final_remote_audit_20260927.json` — `a988a8af499900a53b7fa66dfc786865735aa5b3f18f9eee52f314732c314da9`.
- `docs/mainPlan/results/s3/walker2d-probes-recovery-20260927-1/gate.json` — `b85c71467edb371eb13e795a19318ed5c350fc186aa3ca04acbe0f8b92cae2e6`.
- `docs/mainPlan/results/s3/walker2d-probes-recovery-20260927-1-presentation/manifest.json` — `933a1a98780097fc67548262a24e11fb4db959fbc8cba773f243e454bf1e818d`.
