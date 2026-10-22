"""Small task-policy adapters using command history, without object-state access."""


class ReleaseDelay:
    """Delay an opening request after sustained policy closing, then return control."""

    def __init__(self, hold_steps=0, min_close_steps=3):
        if not isinstance(hold_steps, int) or not 0 <= hold_steps <= 120:
            raise ValueError("hold_steps must be an integer in [0, 120]")
        if not isinstance(min_close_steps, int) or not 1 <= min_close_steps <= 60:
            raise ValueError("min_close_steps must be an integer in [1, 60]")
        self.hold_steps = hold_steps
        self.min_close_steps = min_close_steps
        self.reset()

    def reset(self):
        self.closed_run = 0
        self.remaining = 0
        self.events = 0

    def __call__(self, proposal, obs, feature):
        del obs, feature  # Interface permits observed features; this adapter uses history only.
        opening = proposal[0, -1].item() >= 0
        trigger = (
            opening and self.remaining == 0 and self.closed_run >= self.min_close_steps
            and self.hold_steps > 0
        )
        if trigger:
            self.remaining = self.hold_steps
            self.events += 1
        override = self.remaining > 0
        if override:
            proposal = proposal.clone()
            proposal[:, -1] = -1.0
            self.remaining -= 1
        self.closed_run = 0 if opening else self.closed_run + 1
        return proposal, {
            "adapter": "release_delay",
            "adapter_gripper_override": override,
            "adapter_trigger": trigger,
            "adapter_remaining_steps": self.remaining,
            "adapter_events": self.events,
        }


class GraspReleaseDelay(ReleaseDelay):
    """Delay release only after persistent finger-aperture evidence of a grasp."""

    def __init__(self, hold_steps=0, min_close_steps=3, min_aperture=0.005):
        if not 0 <= min_aperture < 0.025:
            raise ValueError("min_aperture must be in [0, 0.025) metres per finger")
        self.min_aperture = min_aperture
        super().__init__(hold_steps, min_close_steps)

    def reset(self):
        super().reset()
        self.previous_aperture = None

    def __call__(self, proposal, obs, feature):
        if "gripper_pos" not in obs:
            raise ValueError("Grasp-aware adaptation requires measured finger joint positions")
        aperture = obs["gripper_pos"][0].abs()
        contact = bool(((aperture >= self.min_aperture) & (aperture <= 0.025)).all())
        stable = self.previous_aperture is not None and bool(
            (aperture - self.previous_aperture).abs().max() < 0.002
        )
        if not contact or not stable:
            self.closed_run = 0
        changed, log = super().__call__(proposal, obs, feature)
        if not contact or not stable:
            self.closed_run = 0
        self.previous_aperture = aperture.clone()
        log.update(adapter="grasp_release_delay", adapter_grasp_evidence=contact and stable)
        return changed, log
