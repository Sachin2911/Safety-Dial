# Evidence guide

[Dataset home and videos](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction) · [All scientific setups](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/blob/main/reports/reproduction-20261104/setups.md)

The 36 original run folders support one scientific experiment. Use the report and galleries for
reading; these folders retain exact configurations, source snapshots, action traces and provenance.
The links below use the original verified revisions. Failed and stopped jobs remain part of the record.

| Inside a run | Purpose |
| --- | --- |
| `protocol.md`, `config.json`, `source.tar.gz` | Prospective protocol, settings and executed source snapshot |
| `artifacts/` | Cases, provenance, runtime records and generated artifacts |
| `metrics/` | Episode outcomes, action traces and panel summaries, when produced |
| `logs/`, `result.json` | Execution logs and process completion state |
| `manifest.json` | File sizes and SHA-256 checksums for the preserved run |

A process state is not a scientific score. The stopped Reacher job contains the completed primary
panel; only its next, zero-action construction stall was continued. The media-06 package is the final
reviewed gallery; media-01 through media-04 failed and media-05 was superseded after visual review.

## Scientific evaluation

| Run | Process status | Role |
| --- | --- | --- |
| [repro-20261104-eval-cube-seed42-cem10](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/e3aa7017aa421ff7518af893a64ccb325bf9fb6b/runs/repro-20261104-eval-cube-seed42-cem10) | completed | 1 audited scientific panel |
| [repro-20261104-eval-cube-seed42-cem30](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/90b379572054bd9e6384b88cabea784e4875cc4b/runs/repro-20261104-eval-cube-seed42-cem30) | completed | 1 audited scientific panel |
| [repro-20261104-eval-cube-seed43-cem10](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/ce1c2de4313478255dcd509a55e867b1ce041804/runs/repro-20261104-eval-cube-seed43-cem10) | completed | 1 audited scientific panel |
| [repro-20261104-eval-cube-seed44-cem10](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/f6fe3a6ba339c5eebaefea53db43d8d32bee52a0/runs/repro-20261104-eval-cube-seed44-cem10) | completed | 1 audited scientific panel |
| [repro-20261104-eval-pusht-seed42-cem30](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/90afabbf3ad488e5005eab885879d8ac2ec186ba/runs/repro-20261104-eval-pusht-seed42-cem30) | completed | 1 audited scientific panel |
| [repro-20261104-eval-pusht-seed43-cem30](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/21c73823e35673c703d518e435a6741a92429ffa/runs/repro-20261104-eval-pusht-seed43-cem30) | completed | 1 audited scientific panel |
| [repro-20261104-eval-pusht-seed44-cem30](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/c5cb75fb30b13ba27cfcd963f75c6fd1ea4c31f9/runs/repro-20261104-eval-pusht-seed44-cem30) | completed | 1 audited scientific panel |
| [repro-20261104-eval-reacher](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/b7d0c0cbe56d0f454a9bff712d0bc2da2652d8c7/runs/repro-20261104-eval-reacher) | stopped | 1 audited scientific panel |
| [repro-20261104-eval-reacher-seed42-cem30](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/c8de2b7b13f5d304c173e4d7688c23ae381cce42/runs/repro-20261104-eval-reacher-seed42-cem30) | completed | 1 audited scientific panel |
| [repro-20261104-eval-reacher-seed43-cem10](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/7342ebd59b2f4e1cb5aa3423cef9535dffdbaae0/runs/repro-20261104-eval-reacher-seed43-cem10) | completed | 1 audited scientific panel |
| [repro-20261104-eval-reacher-seed44-cem10](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/d98e1a2254a4e1e944a2ed541027fd4048a264ac/runs/repro-20261104-eval-reacher-seed44-cem10) | completed | 1 audited scientific panel |
| [repro-20261104-eval-tworoom](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/1400c508aa326eb18bcda9ea7fea2b8050b80e41/runs/repro-20261104-eval-tworoom) | completed | 4 audited scientific panels |

## Engineering checks

| Run | Process status | Role |
| --- | --- | --- |
| [repro-20261104-engineering-cube-01](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/f8cf9c06e06aaa5f6e71dcfd37ca059d089946c7/runs/repro-20261104-engineering-cube-01) | completed | Supporting evidence |
| [repro-20261104-engineering-cube-02](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/f3e6297e49342be3e359044191e0dbbe1662e2e6/runs/repro-20261104-engineering-cube-02) | completed | Supporting evidence |
| [repro-20261104-engineering-pusht-01](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/4830ff2817aea14fd35e4c90399cb81b12b96dda/runs/repro-20261104-engineering-pusht-01) | completed | Supporting evidence |
| [repro-20261104-engineering-reacher-01](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/bd1332b1a2ffe681345c62b729986ae0a669e4ed/runs/repro-20261104-engineering-reacher-01) | failed | Supporting evidence |
| [repro-20261104-engineering-reacher-02](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/6b3d78c4b38787dfedee186c6ca735becd435be4/runs/repro-20261104-engineering-reacher-02) | completed | Supporting evidence |
| [repro-20261104-engineering-reacher-03](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/bd424a616d94d950a830c69a12e25b63b5717d9a/runs/repro-20261104-engineering-reacher-03) | completed | Supporting evidence |
| [repro-20261104-engineering-reacher-04](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/5d5c6af19aabab65ab987b3a3691343c16b3b01c/runs/repro-20261104-engineering-reacher-04) | completed | Supporting evidence |
| [repro-20261104-engineering-tworoom-01](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/fe196dc06e48ea68bf25a8b9f8076a111aafc33b/runs/repro-20261104-engineering-tworoom-01) | completed | Supporting evidence |

## Dataset and checkpoint preparation

| Run | Process status | Role |
| --- | --- | --- |
| [repro-20261104-prepare-cube](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/022c6eaac688d928e1279d41bfa03b318a397a92/runs/repro-20261104-prepare-cube) | completed | Supporting evidence |
| [repro-20261104-prepare-pusht](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/374075b8b4fa2fcc2621d0b14e069ee9d79b4691/runs/repro-20261104-prepare-pusht) | completed | Supporting evidence |
| [repro-20261104-prepare-reacher](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/fbf42be5acffeb9915a613530f12dd62d43940f9/runs/repro-20261104-prepare-reacher) | completed | Supporting evidence |
| [repro-20261104-prepare-tworoom](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/a0bd93793f434a478c9a94ea03f706aef52027ce/runs/repro-20261104-prepare-tworoom) | completed | Supporting evidence |

## Runtime diagnostics

| Run | Process status | Role |
| --- | --- | --- |
| [repro-20261104-reacher-render-mujoco311](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/ed28f9b67f3738d65b6dcdc7e8908c98f8c01e06/runs/repro-20261104-reacher-render-mujoco311) | failed | Supporting evidence |
| [repro-20261104-reacher-render-mujoco311-dm1044](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/6f5abd485c228ccb4668cd5a01826ea75d8c596b/runs/repro-20261104-reacher-render-mujoco311-dm1044) | completed | Supporting evidence |
| [repro-20261104-reacher-render-osmesa](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/07cf42ec8dc894b8549692920447089255fb285b/runs/repro-20261104-reacher-render-osmesa) | completed | Supporting evidence |
| [repro-20261104-runtime-mujoco311](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/21fefdad8060d831c0ee7c22148b3dff67ff7ed9/runs/repro-20261104-runtime-mujoco311) | completed | Supporting evidence |
| [repro-20261104-runtime-mujoco311-dm1044](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/68b537fed3682c84052a040a193fbe762b3d6b3c/runs/repro-20261104-runtime-mujoco311-dm1044) | completed | Supporting evidence |

## Final scientific audit

| Run | Process status | Role |
| --- | --- | --- |
| [repro-20261104-final-audit](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/2835f451107dc4eabe564d50c700698969d09618/runs/repro-20261104-final-audit) | completed | Supporting evidence |

## Media preparation

| Run | Process status | Role |
| --- | --- | --- |
| [repro-20261104-media-01](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/bc0a3d927e5c065dcbc761d2389119b521b6c4b8/runs/repro-20261104-media-01) | failed | Failed attempt; retained |
| [repro-20261104-media-02](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/22dab59c308c9c6077491e49cac56be0d6ea7a43/runs/repro-20261104-media-02) | failed | Failed attempt; retained |
| [repro-20261104-media-03](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/2a1fa210b4452c2194ca33e76d97f8b447876dd5/runs/repro-20261104-media-03) | failed | Failed attempt; retained |
| [repro-20261104-media-04](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/0bdb225849c68156fba01a7ecb9a49f6776fba1f/runs/repro-20261104-media-04) | failed | Failed attempt; retained |
| [repro-20261104-media-05](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/7448fb8ae31bb14615beac06a5ef9713e3af5318/runs/repro-20261104-media-05) | completed | Superseded |
| [repro-20261104-media-06](https://huggingface.co/datasets/Sachioster/LeWMExp1_Baseline_Reproduction/tree/a00abce4219395d4f8fbdd1d7fbcd28b211c05ab/runs/repro-20261104-media-06) | completed | Final gallery |
