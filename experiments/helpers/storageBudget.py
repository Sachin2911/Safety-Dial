"""Read-only Hub storage accounting and conservative checkpoint-history projections.

Published allowances: https://huggingface.co/docs/hub/storage-limits (checked 2026-09-26).
Git history retains prior checkpoint versions; projected bytes deliberately assume no
cross-version deduplication. These estimates do not purchase storage or alter repositories.
"""

from __future__ import annotations

import math

STORAGE_LIMITS_URL = "https://huggingface.co/docs/hub/storage-limits"


def private_storage_profile(store, *, quota_bytes=None) -> dict:
    profile = store.api.whoami()
    own_namespace = store.namespace == profile["name"]
    if quota_bytes is None:
        if not own_namespace:
            raise ValueError("An organization quota must be supplied explicitly")
        quota_bytes = 1_000_000_000_000 if profile.get("isPro") else 100_000_000_000
    totals, incomplete = {}, []
    for kind, listing in (("model", store.api.list_models),
                          ("dataset", store.api.list_datasets), ("space", store.api.list_spaces)):
        count, used = 0, 0
        for repository in listing(author=store.namespace):
            if not repository.private:
                continue
            info = store.api.repo_info(repository.id, repo_type=kind)
            size = getattr(info, "used_storage", None)
            count += 1
            if size is None:
                incomplete.append(kind)
            else:
                used += int(size)
        totals[kind] = {"private_repositories": count, "used_bytes": used}
    buckets = list(store.api.list_buckets(namespace=store.namespace))
    totals["bucket"] = {"private_repositories": sum(bool(b.private) for b in buckets),
                        "used_bytes": sum(int(b.size) for b in buckets if b.private)}
    used = sum(item["used_bytes"] for item in totals.values())
    return {"quota_bytes": int(quota_bytes), "used_bytes": used,
            "available_bytes": max(0, int(quota_bytes) - used), "by_type": totals,
            "complete": not incomplete, "missing_usage_types": sorted(set(incomplete)),
            "quota_source": STORAGE_LIMITS_URL, "namespace_is_token_owner": own_namespace,
            "is_pro": bool(profile.get("isPro")), "can_pay": bool(profile.get("canPay"))}


def checkpoint_storage_budget(profile: dict, *, total_steps: int, push_every: int,
                              checkpoint_bytes: int, model_runs: int = 1,
                              reserve_bytes: int = 10_000_000_000,
                              overhead_fraction: float = 0.15) -> dict:
    if min(total_steps, push_every, checkpoint_bytes, model_runs) <= 0:
        raise ValueError("Checkpoint projection counts and sizes must be positive")
    if reserve_bytes < 0 or overhead_fraction < 0:
        raise ValueError("Storage reserve and overhead must be nonnegative")
    versions = math.ceil(total_steps / push_every)
    raw = versions * checkpoint_bytes * model_runs
    projected = math.ceil(raw * (1 + overhead_fraction))
    remaining = profile["quota_bytes"] - profile["used_bytes"] - projected - reserve_bytes
    return {"passes": bool(profile.get("complete") and remaining >= 0),
            "versions_per_model": versions, "model_runs": model_runs,
            "checkpoint_bytes": checkpoint_bytes, "raw_history_bytes": raw,
            "projected_history_bytes": projected, "reserve_bytes": reserve_bytes,
            "overhead_fraction": overhead_fraction, "remaining_after_reserve_bytes": remaining,
            "assumes_deduplication": False, "profile": profile}


def require_storage_budget(budget: dict) -> dict:
    if not budget["passes"]:
        raise RuntimeError("Checkpoint storage budget is insufficient or usage is incomplete: "
                           f"remaining after reserve={budget['remaining_after_reserve_bytes']} bytes")
    return budget
