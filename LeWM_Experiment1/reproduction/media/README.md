# Reproduction videos and trajectories

The [private Hugging Face results page](https://huggingface.co/datasets/Sachioster/lewm-experiment1/blob/main/reports/reproduction-20261104/README.md)
collects the updated one-page PDF, full results, audits and eight examples.
Each environment has one recorded success and one failure, selected as the first
case of each outcome in the primary seed-42 panel. These examples illustrate
outcomes; they are not a representative performance sample.

Each MP4 reconstructs the 50 recorded post-action states at 10 displayed actions
per second. Each trajectory image includes states after actions 1, 25 and 50.
The clips are labelled as reconstructions, not original recordings or real-time
playback. The science used no video capture and has not been rerun.

TwoRoom, Reacher and Cube use recorded positions directly. PushT combines saved
block poses with its native kinematic agent update applied to the saved actions.
All 100 reconstructed PushT rewards match exactly, and every termination flag
matches. The selected Reacher success has no retained goal image and is labelled
accordingly. The selected Cube success met the native condition at action 1;
its nearly stationary cube is shown at a physical plot scale.

The final package is `repro-20261104-media-06`. Four failed media attempts and the
superseded pre-visual-review package remain preserved. Corrections addressed
Pymunk body restoration order, native arithmetic precision, byte-string metadata,
cached drawing geometry and stationary-trajectory plot scaling. The final package
reuses verified TwoRoom and Reacher artifacts from attempt 02.

- [Protocol](protocol.md) and [selected cases](config.json).
- [Cloud and independent-download receipt](receipt.json), including every attempt.
- [Visual review](visual-review.json), including file hashes.
- [Report publication receipt](publication.json), pinning the PDF and report revision.
- [Rendering source](render.py), executed only on the rented host.

All attempts together reserved 24 environment setups, used 524 setup physics steps
and made 20 reset-internal controller calls. These are separate from the original
scientific interaction ledger, which is unchanged. There were zero new evaluation
actions and zero model queries. This completed media protocol authorizes no further
rendering attempts. Instance 59111700 was subsequently destroyed at Sachin's
explicit direction after backup verification; see the
[shutdown receipt](../records/rental-shutdown.json).
