"""Hard per-episode counter ceilings, reserved in full before each episode."""

from collections import defaultdict


def ceilings(env):
    work = 50000 if env == "tworoom" else 15000
    return {
        "environment_actions": 50,
        "executed_physics_steps": 50 if env == "tworoom" else 100,
        "predictor_steps": 10 * (work + 5),
        "analytic_path_steps" if env == "tworoom" else "checker_physics_steps": (
            10 * (work * 5 + 2 * 25) + 50
        )
        * (1 if env == "tworoom" else 2),
    }


class BudgetCounter(defaultdict):
    def __init__(self, env, native_parity=False):
        super().__init__(int)
        self.limits = ceilings(env)
        if native_parity:
            self.limits["predictor_steps"] += 50000 if env == "tworoom" else 15000

    def __setitem__(self, key, value):
        if key in self.limits and value > self.limits[key]:
            raise RuntimeError(f"Resource ceiling exceeded before work: {key}")
        super().__setitem__(key, value)
