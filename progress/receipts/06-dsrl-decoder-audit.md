# Task 6 offline decoder audit

The formal authority gate is invalid and learning remains unadmitted. All 128 probe queries were used. This audit reads the saved files only and makes no new model, actor, gradient or simulator call.

Four of sixteen native replay comparisons pass the frozen 1e-5 absolute/relative tolerance. These are the two repeats of each factual snapshot collected on GPU lane 0. The twelve comparisons against factual outputs from lanes 1 to 3 fail; the largest absolute difference is 0.0031306744. All sixteen preserve the converted gripper commands, both over the official first eight actions and over the full sixteen-action chunk. Both repeats are exactly equal for every one of the eight saved inputs.

Inference: the retained GPU-0 policy server is consistent with itself on these inputs, while factual outputs collected on other lanes vary numerically from its replays. This pattern does not identify a hardware, compilation or kernel cause. The identity criterion remains failed.

Descriptively, one of eight snapshots meets the original bounded-candidate gripper margins: E2:q629, task 10, prefix positions 6 and 7. Its candidate gripper range is -0.1278138 to 1.0690833. The other seven saved candidate sets do not meet those margins. This is below the declared two-snapshot, two-task minimum, but the invalid identity gate prevents treating it as a valid authority screen or an absence result. The JSON records all fourteen original vectors individually, margin witnesses, action RMS differences, query references and file hashes.

Accounting is 1,140 base-policy queries: 1,000 native rollout queries, 12 startup queries and 128 probe queries. There were four simulator starts, 8,000 policy controls, 40 warm-up controls, eight charged engineering updates and no development learning updates or remote actor calls. No replacement was used. All 128 probe artifacts and eight factual artifacts matched their recorded SHA256 values and saved observations/actions.

Binding: source v4 `3ef7306af4645aefb9d7d1863758481adf0d11dc97ede90f44d73d8d5cca58fc`; authority artifact SHA256 `042579d2a264ce75becc4ce62792dcdc2185cb82e065ffc6156e05b6ebc1189b`. The machine-readable companion is `06-dsrl-decoder-audit.json`. Raw and frozen source files were not edited.
