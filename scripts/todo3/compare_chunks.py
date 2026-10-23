#!/usr/bin/env python3
"""Identity check for todo 3 item 3: compare the policy chunks of two recorded episodes.

Both episodes are the same task and seed, recorded with scripts/todo2/record_rollout.py,
one straight to the policy server and one through supervisor_proxy.py --mode off. Pass if
every chunk up to and including the first finger contact is byte-identical (the physics is
not bit-reproducible after contact, so later chunks may differ).

    python -I scripts/todo3/compare_chunks.py <direct.npz> <proxied.npz>
"""
from __future__ import annotations

import json
import sys

import numpy as np


def load(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return z, meta


def main(a, b):
    za, ma = load(a)
    zb, mb = load(b)
    for k in ("scene", "episode_seed"):
        if ma.get(k) != mb.get(k):
            print(f"FAIL: {k} differs: {ma.get(k)!r} vs {mb.get(k)!r}"); return 1
    ca, cb = za["proposal_chunk"], zb["proposal_chunk"]
    sa, sb = za["proposal_step"], zb["proposal_step"]
    n = min(len(ca), len(cb))
    first_diff = next((i for i in range(n) if ca[i].tobytes() != cb[i].tobytes()), None)
    contact = za["jar_finger_contact"].astype(bool)
    contact_step = int(np.argmax(contact)) if contact.any() else None
    contact_query = None
    if contact_step is not None:
        contact_query = int(np.searchsorted(sa, contact_step, side="right"))  # chunks issued at or before contact
    print(f"chunks: {len(ca)} direct, {len(cb)} proxied; first differing chunk: {first_diff}; "
          f"first finger contact at step {contact_step} (chunk index {contact_query})")
    print(f"first chunk identical: {ca[0].tobytes() == cb[0].tobytes()}")
    if first_diff is None:
        print("PASS: every chunk byte-identical"); return 0
    if contact_query is not None and first_diff >= contact_query:
        print("PASS: identical up to and including the first contact chunk"); return 0
    print("FAIL: chunks differ before the first finger contact"); return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
