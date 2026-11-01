"""Frozen LeWM rollout with a frozen decoded-current-target terminal task metric.

The wrapped get_cost performs the sole imagination call. This wrapper reuses its
saved endpoint, projects endpoint and CURRENT planning-target embeddings through
the qualified linear readout, and replaces only the task score. Safety, candidate
projection and CEM remain outside this wrapper and retain their existing authority.
"""

import math

import torch


ORIGINAL_BACKEND = "original-latent-v1"
DECODED_BACKEND = "decoded-current-target-v1"


class DecodedCurrentTargetCost(torch.nn.Module):
    def __init__(self, frozen, readout, counters):
        super().__init__()
        self.frozen, self.readout, self.counters = frozen, readout, counters
        self.history = []

    def encode(self, info):
        return self.frozen.encode(info)

    def _publish(self):
        if hasattr(self.counters, "publish"):
            self.counters.publish()

    def _project(self, embedding, role):
        rows = math.prod(embedding.shape[:-1])
        c = self.counters
        c["decoded_task_readout_call_attempts"] += 1
        c["decoded_task_readout_rows_attempted"] += rows
        c[f"decoded_task_{role}_rows_attempted"] += rows
        self._publish()
        try:
            value = self.readout(embedding)
            if value.dtype != torch.float64 or value.shape != (*embedding.shape[:-1], 2):
                raise ValueError("Decoded terminal scoring requires the qualified float64 XY map")
            if not torch.isfinite(value).all():
                raise ValueError("Nonfinite decoded terminal XY")
            c["decoded_task_readout_call_successes"] += 1
            c["decoded_task_readout_rows_successful"] += rows
            c[f"decoded_task_{role}_rows_successful"] += rows
            return value
        except BaseException:
            c["decoded_task_readout_call_failures"] += 1
            raise
        finally:
            self._publish()

    def get_cost(self, info, candidates):
        c = self.counters
        rows = int(candidates.shape[0] * candidates.shape[1])
        c["decoded_task_cost_call_attempts"] += 1
        c["decoded_task_candidate_rows_attempted"] += rows
        self._publish()
        try:
            # Exactly one original get_cost, which populates predicted_emb in place.
            original = self.frozen.get_cost(info, candidates)
            endpoint = info["predicted_emb"][..., -1, :]
            target = info["goal_emb"][..., -1, :]
            while target.ndim < endpoint.ndim:
                target = target.unsqueeze(-2)
            predicted_xy = self._project(endpoint, "endpoint")
            target_xy = self._project(target, "target")
            squared = (predicted_xy - target_xy).square().sum(-1)
            if squared.shape != original.shape or not torch.isfinite(squared).all():
                raise ValueError("Decoded task score must match the original candidate cost shape")
            result = squared.to(dtype=original.dtype, device=original.device)
            if not torch.isfinite(result).all():
                raise ValueError("Decoded task score overflowed the planner dtype")
            c["decoded_task_cost_call_successes"] += 1
            c["decoded_task_candidate_rows_successful"] += rows
            self.history.append(
                {
                    "call": len(self.history),
                    "candidate_rows": rows,
                    "endpoint_shape": list(endpoint.shape),
                    "target_shape": list(target.shape),
                    "original_task_cost": original.detach().cpu().tolist(),
                    "decoded_task_cost": result.detach().cpu().tolist(),
                    "predicted_terminal_xy": predicted_xy.detach().cpu().tolist(),
                    "current_target_xy": target_xy.detach().cpu().tolist(),
                    "score_dtype": str(result.dtype),
                    "projection_dtype": str(predicted_xy.dtype),
                    "goal_readout_sha256": self.readout.fingerprint,
                }
            )
            return result
        except BaseException:
            c["decoded_task_cost_call_failures"] += 1
            raise
        finally:
            self._publish()

    def artifact(self):
        return {
            "schema": "lewm-decoded-current-target-cost-v1",
            "backend": DECODED_BACKEND,
            "formula": "sum((readout(predicted_emb[..., -1, :])-readout(goal_emb[..., -1, :]))**2, axis=-1)",
            "current_target": "The target embedding passed by the planner, including a selected real observed successor; never silently replaced with the original final goal",
            "goal_readout_sha256": self.readout.fingerprint,
            "projection_dtype": "torch.float64",
            "returned_score_dtype": "Exact dtype of the wrapped original task score",
            "extra_encoder_predictor_queries": 0,
            "history": self.history,
        }
