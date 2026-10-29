"""Task-9 proposal adapters. No access to object geometry or outcome labels."""


class Adapter:
    def __init__(self, condition):
        self.condition = dict(condition)
        self.kind = condition["adapter_kind"]
        if self.kind not in {"grasp_release_delay", "release_delay", "grasp_latch"}:
            raise ValueError(self.kind)
        self.hold_steps = condition.get("hold_steps", 0)
        self.min_close_steps = condition["min_close_steps"]
        self.min_aperture = condition.get("min_aperture_m", 0.005)
        self.scale = condition["translation_scale"]
        if not isinstance(self.hold_steps, int) or not 0 <= self.hold_steps <= 120:
            raise ValueError("Invalid delay")
        if not 1 <= self.min_close_steps <= 60 or not 0 <= self.min_aperture < 0.025:
            raise ValueError("Invalid grasp parameters")
        if not 0 < self.scale <= 1:
            raise ValueError("Invalid translation scale")
        self.reset()

    def reset(self):
        self.closed_run = self.remaining = self.events = 0
        self.previous_aperture = None
        self.latched = False

    def __call__(self, proposal, obs, feature):
        del feature
        if proposal.shape != (1, 7):
            raise ValueError("Task 9 requires one seven-coordinate action")
        evidence = False
        if self.kind != "release_delay":
            aperture = obs["gripper_pos"][0].abs()
            contact = bool(((aperture >= self.min_aperture) & (aperture <= 0.025)).all())
            stable = self.previous_aperture is not None and bool(
                (aperture - self.previous_aperture).abs().max() < 0.002
            )
            evidence = contact and stable
            self.previous_aperture = aperture.clone()
            if not evidence:
                self.closed_run = 0
        opening = proposal[0, -1].item() >= 0
        trigger = (
            opening
            and self.remaining == 0
            and self.closed_run >= self.min_close_steps
            and (self.hold_steps > 0 or self.kind == "grasp_latch")
            and not self.latched
        )
        if trigger:
            self.events += 1
            if self.kind == "grasp_latch":
                self.latched = True
            else:
                self.remaining = self.hold_steps
        override = self.latched or self.remaining > 0
        scaling = evidence and self.scale != 1.0
        changed = proposal.clone() if override or scaling else proposal
        if override:
            changed[:, -1] = -1.0
            self.remaining = max(0, self.remaining - 1)
        if scaling:
            changed[:, :3] *= self.scale
        # History is always the ORIGINAL task command, even during an override.
        self.closed_run = 0 if opening else self.closed_run + 1
        if self.kind != "release_delay" and not evidence:
            self.closed_run = 0
        return changed, {
            "adapter": self.kind,
            "adapter_gripper_override": override,
            "adapter_trigger": trigger,
            "adapter_remaining_steps": self.remaining,
            "adapter_events": self.events,
            "adapter_grasp_evidence": evidence,
            "adapter_translation_active": scaling,
            "adapter_latched": self.latched,
        }
