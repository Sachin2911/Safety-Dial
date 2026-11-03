"""Deterministic reservations for a strict simulator-action allowance."""


def allocation(remaining, waiting, workers=8, candidate_maximum=1600):
    if remaining < 0 or waiting < 0:
        raise ValueError("Budgets and candidate counts cannot be negative")
    if not remaining or not waiting:
        return []
    width = min(workers, waiting, max(1, remaining // candidate_maximum))
    quotas = [min(candidate_maximum, remaining)] if width == 1 else [candidate_maximum] * width
    assert sum(quotas) <= remaining
    return quotas


def eligible(results, panel_size=8):
    return len(results) == panel_size and all(r["status"] == "complete" for r in results)


def episode_status(actions, terminal, action_limit, episode_cap=200):
    if not 0 < action_limit <= episode_cap or not 0 <= actions <= action_limit:
        raise ValueError("Invalid episode action accounting")
    return "complete" if terminal or actions == episode_cap else "budget_exhausted"
