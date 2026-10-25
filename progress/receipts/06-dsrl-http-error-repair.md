# Task 6 HTTP failure receipt repair

25 October 2026. This observational source repair follows the first remote
engineering profile's HTTP 409 response. The original worker exit receipt records
completion at 23:50:26.765139 UTC, but its traceback retained only the HTTP status.
The response JSON explaining the source learner's exception was not saved. The
coordinator became pending and did not retry the operation. Two on-host source
parity update attempts remained charged, in addition to two workstation updates;
the scheduled 128 timed profile updates had not begun.

[run_dsrl06.py](../../experiments/scripts/run_dsrl06.py) now catches an HTTP error
only to preserve its status and response body before raising the same failure.
For `http/<operation>.response.json`, the separate receipt is
`http/<operation>.error.json`. It records the parsed JSON body, or a marked UTF-8
text fallback, the response-body SHA256 and the immutable request-file SHA256.
It saves no request headers or environment variables. A flushed temporary file is
linked atomically into a new receipt path. An identical existing receipt is left
unchanged; a different one cannot replace it.

The error file is never a success response. No retry, learner update, model query,
simulator start, controller change, numerical tolerance change or new admission is
introduced. Historical response and exit files remain unchanged. An explicit
further engineering diagnostic remains a distinct operation whose update attempts
must be charged under the existing caps.

Profile recovery now also requires the source event's `request_sha256` to match
SHA256 of `json.dumps(request_body, sort_keys=True)` with default separators, exactly
matching the learner's hashing rule. An explicit `operation_id` in a repaired request
therefore distinguishes it from an otherwise identical earlier request. Missing or
mismatched fingerprints, mismatched timed-update/parity fields and any known HTTP
error receipt remain pending. An unrelated zero-update diagnostic cannot satisfy a
128-update profile request. A preserved failure never becomes a recovered success.

`./.venv/bin/python -m pytest -q experiments/tests/test_dsrl06_queue.py`: 21 passed.
Ruff check and format pass. New checks use an in-memory mocked HTTP error, with no
network or ML calls. They cover structured HTTP 409 preservation, exact request
attribution, immutable receipt behavior, unchanged failure propagation, no success
file, and a non-JSON HTTP 503 response. Existing recovery and no-duplicate queue
checks continue to pass. Additional recovery checks distinguish missing identity,
different operation IDs, zero-update diagnostics, exact matching completed profiles
and a known failure with a superficially matching completion event.
