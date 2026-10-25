# Task 6 rental and budget controls

25 October 2026. Sachin authorized implementation, Vast rental and SSH execution
within the reviewed task 6 protocol and its hard $15 total cap. No task 6 rental
or simulator work had occurred when these workstation controls were implemented.

`manage_dsrl06.py` requests a fresh quote for four RTX 4090 GPUs with driver 580.x,
at least 250,000 MB allocated RAM, 48 effective CPU cores and 200 GB selected disk.
The combined quoted compute/disk rate cannot exceed $2.20/hour. The transfer check
prices 100 GB incoming and 8 GB outgoing and rejects offers above $1.80 for those
amounts, leaving part of the $2 reserve for other storage. Exact rates and the
physical machine are recorded in a fresh rental file rather than inherited from
an older experiment. Comparisons remain on one physical host.

A locked financial book reserves the quoted rate before the create request.
Creation acknowledgement loss triggers reconciliation by a unique rental label,
never a repeated create request. A rejected request is released only after a
validated provider list confirms absence. If the provider unexpectedly returns
multiple matching allocations, every instance is recorded, charged and stopped
immediately. Their combined hours and rates remain in the ledger.

The independent workstation watchdog confirms a live heartbeat before the purchase.
The instance watchdog confirms its instance-scoped credentials and a live process
before setup begins. Credentials stay in the provider environment. CLI errors are
captured and their raw text withheld because the Vast CLI can include an API key
in exception URLs. No workstation secret is bundled or passed to SSH commands.

Both guards begin stop requests five minutes before the earlier aggregate six-hour
or $12.90 active-spend deadline. The original $0.10 rounding/shutdown margin stays
inside the $13 active allowance. The instance guard also observes transfer bytes;
unexpected transfer costs reduce remaining compute time. Admission and preservation
use the earlier effective deadline. A provider process reported as exited still
counts as active until actual, intended and current provider states all confirm
stopped, or a validated complete provider list proves absence.

Setup replays the pinned environment through separate intended interpreters.
Its only build adjustment is targeting the RTX 4090's CUDA architecture 8.9 in a
generated copy of the previous cuRobo build script, with `MAX_JOBS=16`. The original
scripts remain unchanged. Known task 5 package repairs are reapplied, the guarded
DSRL recorder patch is installed, full source revisions are checked, and environment
freezes, patch hashes and downloaded-asset provenance are preserved. The learner has
its own fully pinned Python environment. The root legacy environment is never synced
into the template or the simulator environment.

`make_dsrl06_freeze.py` records the approved job list, caps, complete learner settings,
static latent audit, probe specification, CPU actor topology and source hashes before
renting. It imports the two already consumed workstation gradient attempts. The
bootstrap archive contains only explicitly inventoried source and preflight evidence.
Its bytes are checked after SSH transfer. Continuous local copying starts during setup.
Final preservation uses a consistent SQLite backup and pinned private uploads.
The destroy command requires the complete local inventory and its unchanged manifest,
verified private revisions, and confirmed compute stop before deleting the disk.

Workstation regressions cover uncertain/rejected creates, duplicate allocations,
conservative failed-host charges, concurrent ledger readers, inherited counters,
provider errors that do not prove absence, secret-safe CLI failures and destruction
before preservation. They use a mocked provider and make no rental requests.

The final combined workstation pass reported 137 tests passed, zero failures and
zero skips, including the localhost learner and actor service checks. Ruff check,
format verification, both setup scripts' shell syntax and `git diff --check` passed.
The raw report and its hashes are recorded in the stage A verification section.
Independent source review found no remaining blocking integration defect. Native
decoder replay, gripper authority, GPU memory, real rollout topology and throughput
remain engineering gates; these workstation checks do not claim they passed.
