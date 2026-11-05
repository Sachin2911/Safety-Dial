# Reproduction media, 4 November 2026

Sachin requested videos and trajectory pictures, selected one success and one
failure per environment, and explicitly requested Hugging Face publication.
This is presentation of completed evidence, not an extension of the completed
scientific protocol. The existing instance is retained at his direction.

Use the first successful and first unsuccessful case, by saved case order, in
each primary seed-42 main panel. `config.json` freezes the eight case identities
and source Hugging Face revisions. These examples illustrate outcomes and are
not a representative performance sample. Keep original success labels, including
any early success followed by later movement. Do not choose more dramatic cases.

Render the 50 recorded post-action states at 10 displayed actions per second.
Label the videos as reconstructions, not original recordings or wall-clock
playback. TwoRoom, Reacher and Cube use saved positions directly. PushT saved
block poses but not the agent position: reconstruct its kinematic agent from
the saved actions and initial state, with the native PD update equations.
Require reconstructed native rewards and termination labels to match the saved
trace before publication. Do not run CEM or advance an evaluation episode.

Restore available start/goal inputs from the already verified caches. Reacher's
cache was deleted with approval: use its recorded joint states and the saved
case-zero goal image where available. Label the selected success example's
missing initial/goal image explicitly; never invent a target. Save the exact
reconstruction inputs and per-frame positions with the media for recovery.

All simulator construction, restoration and rendering run on the existing rented
host. Set up at most 24 environments across rendering attempts, with at most
2,000 setup physics steps. Count setup resets, native controller calls and physics
steps separately from the unchanged scientific ledger. Allow no new evaluation
actions or model queries. Stop if a reconstruction cannot be validated; retain
the failure and report the limitation. Rendering may be retried within this
bounded presentation allowance; no scientific outcome is rescored.

Use at most 60 minutes of the existing session, within its original six-hour
deadline and $8 estimate, and at most 100 MB of new published media. No new rental
or public dataset download is needed. Native source and scientific snapshots stay
unchanged. Keep a separate media audit and source hashes. Verify MP4 decoding,
frame counts, image dimensions, trajectory consistency and cloud readback.
Publish eight videos, eight trajectory images, their reconstruction records and
the updated one-page PDF with the existing results in the private dataset
`Sachioster/lewm-experiment1`. Rental destruction remains manual.
