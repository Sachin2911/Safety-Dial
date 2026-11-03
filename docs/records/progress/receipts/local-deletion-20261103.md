# Local archive and output deletion, 3 November 2026

After the first SafeLeWM cloud migration, Sachin requested deletion of all three
remaining cleanup groups. He confirmed "I think all 3" after clarification that
the groups included active work and evidence not yet verified in cloud storage.
This supersedes the earlier preservation requirement for these local paths only:

- `archive/`: the older root archive, including historical code and document bundles.
- `outputs/`: historical local outputs.
- `SafeLeWM/outputs/`: remaining LeWM local outputs and transfer files.
- `SafeBlock/outputs/`: Block Plucking local outputs.

The [machine-readable receipt](local-deletion-20261103.json) records completion,
per-folder file counts, byte counts and deletion times. The compressed
[path inventory](local-deletion-20261103.paths.jsonl.gz) records names, types,
sizes, permissions and timestamps. It contains no file contents and is not a backup.
Symlinks are removed without following their targets.

Deletion completed for all four folders: 761,325 regular files containing
82,409,928,177 logical bytes. Their pre-deletion allocated footprint was about
79.61 GB (74.14 GiB); logical size includes repeated hard-linked file paths.

No new cloud backup or full backup-coverage check was performed for this deletion.
Existing cloud copies and Git history may cover some removed files; this record
does not claim that all deleted content can be recovered. Historical preservation
receipts remain statements about their original verification dates. Historical
scripts and notes can still reference the removed paths; recover their inputs from
an available independent copy before attempting those workflows.

The previously verified sixteen SafeLeWM archives remain in the private
[Hugging Face evidence dataset](https://huggingface.co/datasets/Sachioster/safety-dial-evidence).
Their [local index, receipt and recovery instructions](../../SafeLeWM/archive/README.md)
remain available. The cloud migration's disposable preparation directory was part
of `SafeLeWM/outputs/` and is included in this deletion.

Current source code outside the root archive, protocols, result reports, manuscripts
and the Python environment remain in the project. This local deletion changes no
remote storage, running experiment or rental. It authorizes no new experiment.
