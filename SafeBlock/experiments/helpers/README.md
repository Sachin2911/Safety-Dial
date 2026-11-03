# Frozen-system runtime helpers

These six UNISafe runtime helpers were moved from the root `experiments/helpers/`
on 3 November 2026. SafeBlock now owns its runtime code, settings and local assets.
`unisafe09_source.ROOT` resolves to `SafeBlock/`; its relative configuration, data
and upstream paths therefore stay together without root-level compatibility links.

The upstream checkout remains pinned at `ab0fdf87f38edefa48b54a291ec5b7156c17b6e0`
under ignored `SafeBlock/third_party/unisafe09/UNISafe/`. Existing weights and runtime
files are under ignored `SafeBlock/data/unisafe09/`. Source preparation and simulator
work remain restricted to rented hosts under a newly applicable protocol.

Frozen protocol JSON retains its original bytes. The working entry points use
`project_paths.project_path` to resolve the old asset and output prefixes into
SafeBlock. This relocation does not revive deleted output evidence or authorize a run.
