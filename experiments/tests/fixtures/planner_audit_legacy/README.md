# Legacy planner compatibility fixtures

These are exact, unmodified source bytes from commit
`0d625b87fa04b80bfc24f08a9a575111efd75c17`, before the base planner acquired
its `_audit` hook. They preserve the legacy side of the audit-wrapper
compatibility tests after the current planner changes are committed.

The tests load these local files and verify their SHA-256 hashes. No Git
executable, repository history, network access, or old commit is required at
test runtime, so this coverage also works in shallow clones and source archives.
The `.py.txt` suffix keeps these historical sources out of module discovery.
Do not reformat or update the fixture source bytes.

| Fixture | Original path | SHA-256 |
|---|---|---|
| `imagination.py.txt` | `experiments/helpers/imagination.py` | `ce28b16299b2d4d9d8f84642db34c25ecf714736d41ab34a437070e1f3d5eef6` |
| `safeCemT.py.txt` | `experiments/helpers/safeCemT.py` | `0c7ac2793b94325fcf0d7cec43a962a822f1c881244d7de55d4e6430f786e76a` |

The fixture files were extracted with `git show <revision>:<original-path>`
without decoding, newline normalization, or other source changes.
