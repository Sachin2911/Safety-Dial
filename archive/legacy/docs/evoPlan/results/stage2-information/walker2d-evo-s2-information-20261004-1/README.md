# Preserved training run with failed validation setup

All 27,000 optimiser updates completed and 23 candidate checkpoints were saved.
Validation setup then failed when the descriptive progress-floor string was
converted to a float. No candidate was selected or evaluated on development here.

The recorded validation replay spent 6,400 simulator steps; history encoding and
render verification used 256 renders and 192 encodes. Those costs remain charged.
The continuation walker2d-evo-s2-information-resume-20261004-1 corrected only the
intended numeric ratio to 0.5 and reused the hash-verified weights without retraining.
Its report contains the completed validation and development results.

This bundle preserves the original failed configuration, checkpoint pool, training
history, cached state features, validation roots, failure record, source and logs.
The manifest was assembled after failure; protocol commit bb73e80 identifies the
executed source. Wall time was not captured by the failed runner and is unreported.
