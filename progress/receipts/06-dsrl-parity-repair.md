# Task 6 engineering reference-parity repair

25 October 2026. The first GPU engineering profile returned HTTP 409 rather than
remaining in compilation. The subsequent explicit, separately counted diagnostic
at `2026-10-24T23:58:32.947406+00:00` retained this error:

```text
Bridge differs from direct author update: 3.1837262213230133e-06
```

Its request lasted 62.046 seconds. The unchanged parity tolerance is 1e-6. Both
requests reserved their reference and bridge updates before dispatch and failed
before any timed profiling updates. The learner reached six charged engineering
updates: two prior workstation updates and four GPU parity updates. Root saved
complete before/after diagnostic checkpoints and retained the failed requests;
the repair does not erase those charges or replace the frozen receipts.

The previous check constructed a new author learner, then deserialized dynamic
state using that new learner's template. Flax TrainState `apply_fn` and Optax `tx`
are static PyTree fields. Deserialization preserves the template's newly created
function closures, so numerically identical dynamic values can reach separate
JIT specializations and independently selected GPU kernels. Initial reference
and bridge compilation intervals were approximately 66 and 54 seconds; the next
fresh reference again required approximately 31 seconds. Different static
closures are established by the implementation; the exact numerical contribution
of GPU kernel selection versus reduction nondeterminism is not proven by this
record.

`AuthorBackend.reference_parity` now uses `copy.copy(self.agent)`. The author's
update rebinds immutable actor, critic and temperature TrainStates, target
parameters and RNG arrays. Optax transformations are functional and the source
JIT declares no donated arguments. The shallow agent copy therefore starts with
the exact same static functions, dynamic input buffers, dtypes and RNG, while
the two update calls rebind separate agent attributes. No new learner or
optimizer is constructed for the comparison. This removes the unnecessary
compilation difference without changing the author update equations, critic
settings, optimizers, augmentation, precision defaults or 1e-6 criterion.

Parity diagnostics now include the state tree path holding the largest numeric
difference. Mismatched PyTree structures and nonfinite or mismatched arrays
remain errors. The separate highest-precision actor-inference scope is unchanged
and does not enclose source SAC updates.

Profile results and their durable `profile_completed` events also carry
`request_sha256=json_hash(request)`. This hashes all request metadata, including
an optional operation identity, so recovery can match an exact completed request
and cannot substitute an old failed profile's intent. This field changes neither
the batch nor update count. Root will use a fresh operation identity for the
explicit repaired profile and will retain its full HTTP response.

Seventeen learner tests pass, including two mocked reference-parity cases. They
verify shared immutable initial state and static closures, independent reference
and bridge attribute rebinding, two reservations, identical-result acceptance,
and rejection of a synthetic approximately 3e-6 difference with its state path.
The profile test verifies that changing operation identity changes the recorded
request hash. Both localhost HTTP tests pass. Ruff, formatting and diff checks
pass. These tests use mock updates and make no real gradient, actor, VLA or
simulator call.

Root owns the immutable v4 source amendment, checkpointed service restart and
the next explicit profile. Its intended two parity and 128 timed source updates
must all be charged when dispatched. This receipt claims no repaired GPU parity
result; the runtime result remains a separate engineering gate.
