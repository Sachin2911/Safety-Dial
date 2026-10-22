"""CPU contract for a prospective closure-to-native feasibility reference.

This selector has no simulator, decoder, policy client or policy RNG handle.
Privileged monitor gating is an engineering reference, not a learned actor.
"""

import hashlib
import json

import numpy as np

from helpers.maniguard_latent import CODE_DIM, FEATURE_DIM, LatentController, finite

MODES = {"native_only", "closure_only", "privileged_lid_handoff", "clock_handoff"}


class HandoffReference:
    """Debounce at factual control steps and switch only at native query boundaries.

    The privileged mode requires consecutive closed/supported post-control
    observations without any preceding violation. Loss of that condition before
    the switch cancels qualification. Once switched, native remains latched even
    if closure is later lost. Violation logging never stops an episode here.
    Every new episode requires reset; interrupted physical episodes cannot be
    resumed from this object. Clock/native/closure modes consume no monitor labels.
    """

    def __init__(self, closure_weights, *, mode, debounce=3, clock_switch_step=488):
        if (
            mode not in MODES
            or type(debounce) is not int
            or debounce < 1
            or type(clock_switch_step) is not int
            or clock_switch_step < 0
            or clock_switch_step % 8
        ):
            raise ValueError("Invalid prospective handoff contract")
        self.closure = LatentController(closure_weights)
        if self.closure.identity:
            raise ValueError("Declare a nonidentity closure controller")
        self.closure.weights.setflags(write=False)
        self.native = LatentController(np.zeros((CODE_DIM, FEATURE_DIM), dtype=np.float32))
        self.mode, self.debounce, self.clock_switch_step = mode, debounce, clock_switch_step
        self.contract = {
            "schema": "maniguard-handoff-reference-v1",
            "mode": mode,
            "debounce": debounce,
            "clock_switch_step": clock_switch_step,
            "native_horizon": 16,
            "model_action_dim": 32,
            "executed_prefix": 8,
            "closure_controller_sha256": self.closure.sha256(),
            "native_controller_sha256": self.native.sha256(),
            "arm_scale": self.closure.spec.alpha,
            "basis_seed": self.closure.spec.basis_seed,
            "information_endowment": (
                "privileged factual monitor"
                if self.privileged
                else "causal features and query clock only"
            ),
            "partial_episode_resume": False,
        }
        self.contract_sha256 = hashlib.sha256(
            json.dumps(self.contract, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        self.reset()

    @property
    def privileged(self):
        return self.mode == "privileged_lid_handoff"

    def reset(self):
        self.last_observed, self.last_query = -1, None
        self.consecutive = 0
        self.qualified_step = self.first_debounce_step = None
        self.ever_violated = self.native_latched = False
        self.switch_query_step = self.switch_qualified_step = None

    def observe(self, step, *, closed, supported, violated):
        if not self.privileged:
            raise ValueError("This reference does not consume monitor labels")
        if (
            type(step) is not int
            or step != self.last_observed + 1
            or any(type(v) is not bool for v in (closed, supported, violated))
        ):
            raise ValueError("Require consecutive factual observations and Boolean labels")
        if step and (self.last_query is None or step > self.last_query + 8):
            raise ValueError("A native query boundary was skipped")
        self.last_observed = step
        self.ever_violated |= violated
        if step > 0 and closed and supported and not self.ever_violated:
            self.consecutive += 1
            if self.consecutive == self.debounce:
                self.qualified_step = step
                if self.first_debounce_step is None:
                    self.first_debounce_step = step
        else:
            self.consecutive, self.qualified_step = 0, None

    def state(self):
        return {
            "last_observed": self.last_observed,
            "last_query": self.last_query,
            "consecutive": self.consecutive,
            "qualified_step": self.qualified_step,
            "first_debounce_step": self.first_debounce_step,
            "ever_violated": self.ever_violated,
            "native_latched": self.native_latched,
            "switch_query_step": self.switch_query_step,
            "switch_qualified_step": self.switch_qualified_step,
        }

    def select(self, step, actor_features):
        expected = 0 if self.last_query is None else self.last_query + 8
        if type(step) is not int or step != expected:
            raise ValueError("Selection requires the next exact eight-control query boundary")
        features = finite(actor_features, (FEATURE_DIM,), "causal actor features")
        if np.any(np.abs(features) > 1) or features[0] != 1:
            raise ValueError("Require bounded causal features with the native bias coordinate")
        if self.privileged and self.last_observed != step:
            raise ValueError("Observe this query's actual state before selecting")
        native = (
            self.mode == "native_only"
            or self.native_latched
            or (self.mode == "clock_handoff" and step >= self.clock_switch_step)
            or (self.privileged and self.qualified_step is not None)
        )
        code = np.zeros(CODE_DIM, dtype=np.float32) if native else self.closure.code(features)
        offset = None if native else self.closure.offset(features, 16, 32)
        if native and not self.native_latched and self.mode != "native_only":
            self.switch_query_step = step
            self.switch_qualified_step = self.qualified_step if self.privileged else None
        self.native_latched, self.last_query = native, step
        return {
            "branch": "native" if native else "closure",
            "query_step": step,
            "controller_sha256": self.native.sha256() if native else self.closure.sha256(),
            "code": code,
            "noise_offset": offset,
            "call_native_infer_without_explicit_noise": native,
            "requests_policy_rng_reset": False,
            "policy_rng_continuity_verified": False,
            "privileged_actor_inputs": self.privileged,
            "state": self.state(),
            "contract_sha256": self.contract_sha256,
            "dispatch_authorized": False,
            "physical_fitness": None,
            "new_vla_requests": 0,
            "new_physical_controls": 0,
        }


def replay_factual_prefix(rows, reference, *, controls):
    """Stop before treating any changed controller's future as factual evidence.

    The original full-episode verifier remains the source-integrity authority.
    This check validates observation/query order and numerical closure codes on
    its consumed prefix. It makes no decoder, dynamics or completion prediction.
    Only the privileged reference receives monitor predicates as actor inputs.
    """
    if type(controls) is not int or controls < 1:
        raise ValueError("Declare the factual full-episode control count")
    reference.reset()
    last_observed, last_query, matched = -1, None, 0
    decision = None
    for row in rows:
        kind, step = row["type"], row.get("step")
        if kind == "observed":
            if type(step) is not int or step != last_observed + 1 or step > controls:
                raise ValueError("Incomplete or unordered factual observations")
            if step and (last_query is None or step > last_query + 8):
                raise ValueError("A factual query was skipped")
            last_observed = step
            if reference.privileged:
                monitor = row["monitor"]
                if monitor.get("valid") is not True:
                    raise ValueError("Invalid factual monitor")
                observation = monitor["observation"]
                reference.observe(
                    step,
                    closed=observation["ap"]["jar_closed"],
                    supported=observation["ap"]["jar_on_support"],
                    violated=observation["doomed"],
                )
        elif kind == "latent_features":
            expected = 0 if last_query is None else last_query + 8
            if (
                type(step) is not int
                or step != expected
                or step >= controls
                or last_observed != step
            ):
                raise ValueError("Invalid factual query/observation boundary")
            decision = reference.select(step, row["features"])
            last_query = step
            if decision["branch"] == "native":
                break  # Do not parse or score the closure trajectory after a changed choice.
            actual = finite(row["code"], (CODE_DIM,), "recorded closure code")
            if not np.allclose(actual, decision["code"], atol=1e-6, rtol=0):
                raise ValueError("The declared closure controller differs from the factual code")
            matched += 1
    if decision is None:
        raise ValueError("The factual prefix has no valid query")
    changed = decision["branch"] == "native"
    if not changed and (last_observed != controls or last_query != 8 * ((controls - 1) // 8)):
        raise ValueError("Incomplete factual closure-only episode")
    return {
        "mode": reference.mode,
        "contract": reference.contract,
        "contract_sha256": reference.contract_sha256,
        "first_changed_query_step": last_query if changed else None,
        "last_factual_observation_consumed": last_observed,
        "factual_closure_codes_compared": matched,
        "cpu_controller_code_evaluations": matched + int(changed),
        "monitor_observations_consumed_for_gating": last_observed + 1
        if reference.privileged
        else 0,
        "stopped_at_first_changed_choice": changed,
        "later_original_rows_used_as_hybrid_outcomes": False,
        "state": reference.state(),
        "hybrid_physical_outcome": None,
        "hybrid_grasp_prediction": None,
        "dispatch_authorized": False,
        "policy_rng_continuity_verified": False,
        "new_physical_controls": 0,
        "new_vla_requests": 0,
        "new_encoder_images": 0,
        "new_learned_model_training_or_queries": 0,
    }
