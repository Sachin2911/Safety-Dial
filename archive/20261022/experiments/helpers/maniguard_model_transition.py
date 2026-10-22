"""Small CPU transition ensemble with separately audited task-state readouts.

The model predicts one known executed prefix. It never fabricates future camera
observations or asks a frozen VLA to act on an imagined compact state.
"""

import json
from pathlib import Path

import numpy as np

from helpers.maniguard_model_data import STAGE_NAMES


def numeric(value, dimensions, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != dimensions or not np.isfinite(value).all() or not len(value):
        raise ValueError("Invalid " + name)
    return value


class _RidgeEnsemble:
    """Episode-bootstrap residual regressors with frozen random nonlinear features."""

    def _features(self, x):
        z = (x - self.mean) / self.scale
        nonlinear = np.tanh(z @ self.projection + self.bias)
        return np.concatenate([np.ones((*x.shape[:-1], 1)), z, nonlinear], axis=-1)

    @classmethod
    def fit(cls, x, y, episodes, spec, seed, ledger, request_id):
        x, y = numeric(x, 2, "training inputs"), numeric(y, 2, "training targets")
        episodes = np.asarray(episodes)
        if len(x) != len(y) or episodes.shape != (len(x),):
            raise ValueError("Training inputs, targets and episode groups differ")
        if (type(spec["members"]) is not int or spec["members"] < 2
                or type(spec["random_features"]) is not int or spec["random_features"] < 0
                or not np.isfinite(spec["ridge"]) or spec["ridge"] <= 0
                or not np.isfinite(spec["scale_floor"]) or spec["scale_floor"] <= 0):
            raise ValueError("Invalid fixed ensemble specification")
        model, rng = cls(), np.random.default_rng(seed)
        model.mean = x.mean(axis=0)
        model.scale = np.maximum(x.std(axis=0), spec["scale_floor"])
        model.projection = rng.normal(size=(x.shape[1], spec["random_features"])) / np.sqrt(
            x.shape[1])
        model.bias = rng.uniform(-1, 1, spec["random_features"])
        design = model._features(x)
        unique = np.unique(episodes)
        selections = [np.concatenate([np.flatnonzero(episodes == episode)
                                      for episode in rng.choice(unique, len(unique), replace=True)])
                      for _ in range(spec["members"])]
        ledger.admit(request_id, "training_row_uses", sum(len(s) for s in selections))
        try:
            penalty = np.eye(design.shape[1]) * spec["ridge"]
            penalty[0, 0] = 0
            coefficients = []
            for indices in selections:
                values = design[indices]
                coefficients.append(np.linalg.solve(values.T @ values + penalty,
                                                    values.T @ y[indices]))
            model.coefficients = np.stack(coefficients)
        except Exception:
            ledger.finish(request_id, "failed")
            raise
        ledger.finish(request_id, "returned")
        return model

    def _predict(self, x):
        return np.einsum("nf,mfo->mno", self._features(x), self.coefficients)

    def _predict_paired(self, x):
        return np.einsum("mnf,mfo->mno", self._features(x), self.coefficients)


class TransitionModel:
    """Action-conditioned dynamics, prefix risk and a distinct observation readout.

    Bootstrap spread and calibrated scores are empirical diagnostics. They are
    not certified probabilities for counterfactual actions or physical guarantees.
    """

    @classmethod
    def fit(cls, data, spec, ledger, request_id):
        if (type(spec["support_rows"]) is not int or spec["support_rows"] < 1
                or not np.isfinite(spec["support_radius"]) or spec["support_radius"] <= 0):
            raise ValueError("Invalid state/action support specification")
        if not np.any(data.risk_eligible):
            raise ValueError("No pre-violation prefixes for fitting the risk head")
        model = cls()
        model.spec = dict(spec)
        model.horizon = data.actions.shape[1]
        x = data.inputs()
        model.dynamics = _RidgeEnsemble.fit(x, data.following - data.current,
                                           data.episode, spec, spec["seed"], ledger,
                                           request_id + "/dynamics")
        eligible = data.risk_eligible.astype(bool)
        model.risk = _RidgeEnsemble.fit(x[eligible], data.risk[eligible, None],
                                       data.episode[eligible], spec, spec["seed"] + 1,
                                       ledger, request_id + "/risk")
        model.readout = _RidgeEnsemble.fit(data.following, data.stages, data.episode,
                                          spec, spec["seed"] + 2, ledger,
                                          request_id + "/readout")
        indices = np.linspace(0, eligible.sum() - 1,
                              min(spec["support_rows"], int(eligible.sum())), dtype=int)
        model.support = ((x[eligible] - model.risk.mean) / model.risk.scale)[indices]
        model.calibration = {"ready": False, "offset": 1.0,
                             "reason": "No held-out factual calibration"}
        model.fit_stage_positive_episodes = {
            name: len(np.unique(data.episode[data.stages[:, i] == 1]))
            for i, name in enumerate(STAGE_NAMES)}
        return model

    def _inputs(self, current, previous, history_valid, actions):
        current = numeric(current, 2, "current features")
        previous = numeric(previous, 2, "causal previous features")
        actions = numeric(actions, 3, "known candidate action prefix")
        history_valid = np.asarray(history_valid, dtype=float)
        n = len(current)
        if (current.shape != (n, 44) or previous.shape != current.shape
                or actions.shape != (n, self.horizon, 8) or history_valid.shape != (n,)
                or not np.isin(history_valid, [0, 1]).all()):
            raise ValueError("Model inputs violate the frozen causal feature/action contract")
        return current, np.concatenate([current, previous, history_valid[:, None],
                                         actions.reshape(n, -1)], axis=1)

    def predict(self, current, previous, history_valid, actions, ledger, request_id):
        current, x = self._inputs(current, previous, history_valid, actions)
        ledger.admit(request_id, "model_head_row_queries", len(x) * self.spec["members"] * 3)
        try:
            following = np.clip(current[None] + self.dynamics._predict(x), -1, 1)
            risk = np.clip(self.risk._predict(x)[..., 0], 0, 1)
            stages = np.clip(self.readout._predict_paired(following), 0, 1)
            normalized = (x - self.risk.mean) / self.risk.scale
            distances = []
            for start in range(0, len(x), 64):
                block = normalized[start:start + 64]
                square = (np.sum(block**2, axis=1)[:, None]
                          + np.sum(self.support**2, axis=1)[None]
                          - 2 * block @ self.support.T)
                distances.extend(np.sqrt(np.maximum(0, square.min(axis=1)) / x.shape[1]))
            result = {"next_mean": following.mean(axis=0), "next_std": following.std(axis=0),
                      "risk_mean": risk.mean(axis=0), "risk_std": risk.std(axis=0),
                      "stage_mean": stages.mean(axis=0), "stage_std": stages.std(axis=0),
                      "support_distance": np.asarray(distances)}
        except Exception:
            ledger.finish(request_id, "failed")
            raise
        ledger.finish(request_id, "returned")
        return result

    def actual_readout(self, features, ledger, request_id):
        features = numeric(features, 2, "actual future feature readout")
        ledger.admit(request_id, "model_head_row_queries", len(features) * self.spec["members"])
        try:
            result = np.clip(self.readout._predict(features), 0, 1).mean(axis=0)
        except Exception:
            ledger.finish(request_id, "failed")
            raise
        ledger.finish(request_id, "returned")
        return result

    def risk_upper_score(self, prediction, gate):
        if not self.calibration["ready"]:
            return np.ones(len(prediction["risk_mean"]))
        return np.clip(prediction["risk_mean"]
                       + gate["risk_std_multiplier"] * prediction["risk_std"]
                       + self.calibration["offset"], 0, 1)

    def calibrate(self, data, gate, ledger, request_id):
        prediction = self.predict(data.current, data.previous, data.history_valid, data.actions,
                                  ledger, request_id)
        eligible = data.risk_eligible.astype(bool)
        scores = []
        for episode in np.unique(data.episode[eligible]):
            indices = eligible & (data.episode == episode)
            residual = (data.risk[indices] - prediction["risk_mean"][indices]
                        - gate["risk_std_multiplier"] * prediction["risk_std"][indices])
            scores.append(max(0., float(residual.max())))
        n = len(scores)
        order = int(np.ceil((n + 1) * (1 - gate["alpha"])))
        ready = n >= gate["minimum_calibration_episodes"] and 1 <= order <= n
        self.calibration = {"ready": ready, "episodes": n, "order_statistic": order,
                            "offset": float(sorted(scores)[order - 1]) if ready else 1.0,
                            "reason": "Empirical episode-max residual calibration" if ready
                            else "Insufficient independent episode calibration coverage",
                            "scope": "Exchangeability and counterfactual coverage are unverified."}
        return dict(self.calibration)

    def audit(self, data, gate, ledger, request_id, *, safe_completions):
        prediction = self.predict(data.current, data.previous, data.history_valid, data.actions,
                                  ledger, request_id + "/transition")
        actual = self.actual_readout(data.following, ledger, request_id + "/actual-readout")
        failures, blocks = [], {}
        for name, columns in {"robot_history": slice(1, 20), "overview": slice(20, 28),
                              "wrist": slice(28, 36)}.items():
            error = float(np.sqrt(np.mean((prediction["next_mean"][:, columns]
                                           - data.following[:, columns])**2)))
            persistence = float(np.sqrt(np.mean((data.current[:, columns]
                                                 - data.following[:, columns])**2)))
            passed = error <= gate["maximum_dynamics_to_persistence_rmse"] * persistence
            blocks[name] = {"prediction_rmse": error, "persistence_rmse": persistence,
                            "ratio": error / persistence if persistence else None, "pass": passed}
            if not passed:
                failures.append("Dynamics do not beat persistence in " + name)
        stage_mae = np.mean(np.abs(actual - data.stages), axis=0)
        stage_positive = {name: len(np.unique(data.episode[data.stages[:, i] == 1]))
                          for i, name in enumerate(STAGE_NAMES)}
        for i, name in enumerate(STAGE_NAMES):
            if stage_mae[i] > gate["maximum_actual_state_readout_mae"]:
                failures.append("Actual-state readout error: " + name)
        for name in ("prerequisite", "held", "in_goal"):
            if min(self.fit_stage_positive_episodes[name], stage_positive[name]) < gate[
                    "minimum_positive_stage_episodes"]:
                failures.append("Insufficient positive episode coverage: " + name)
        upper = self.risk_upper_score(prediction, gate)
        eligible = data.risk_eligible.astype(bool)
        positive, negative = eligible & (data.risk == 1), eligible & (data.risk == 0)
        alarm = upper > gate["risk_score_limit"]
        fn = float(np.mean(~alarm[positive])) if positive.any() else None
        fp = float(np.mean(alarm[negative])) if negative.any() else None
        if not self.calibration["ready"]:
            failures.append(self.calibration["reason"])
        if fn is None or fn > gate["maximum_false_negative_rate"]:
            failures.append("Missing positive risk coverage or excessive factual false negatives")
        if fp is None or fp > gate["maximum_false_positive_rate"]:
            failures.append("Missing negative risk coverage or excessive factual false positives")
        if safe_completions < gate["minimum_diagnostic_safe_completions"]:
            failures.append("Insufficient demonstrated safe task completion")
        unsupported = int(np.sum(prediction["support_distance"] > self.spec["support_radius"]))
        if unsupported:
            failures.append("Diagnostic state/action prefixes outside fitted support")
        return {"prediction_gate_pass": not failures, "failures": failures,
                "dynamics_blocks": blocks,
                "actual_state_readout_mae": dict(zip(STAGE_NAMES, stage_mae.tolist(), strict=True)),
                "fit_stage_positive_episodes": self.fit_stage_positive_episodes,
                "diagnostic_stage_positive_episodes": stage_positive,
                "diagnostic_safe_completions": safe_completions,
                "risk_positive_prefixes": int(positive.sum()),
                "risk_negative_prefixes": int(negative.sum()), "false_negative_rate": fn,
                "false_positive_rate": fp, "unsupported_prefixes": unsupported,
                "calibration": self.calibration,
                "scope": ("Factual prototype gate on previously exposed tasks. "
                          "Not a physical or counterfactual safety certificate.")}

    def save(self, path, *, corpus_sha256):
        metadata = {"spec": self.spec, "horizon": self.horizon,
                    "calibration": self.calibration, "corpus_sha256": corpus_sha256,
                    "fit_stage_positive_episodes": self.fit_stage_positive_episodes,
                    "physical_execution_enabled": False}
        arrays = {"metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
                  "support": self.support}
        for name in ("dynamics", "risk", "readout"):
            head = getattr(self, name)
            for key in ("mean", "scale", "projection", "bias", "coefficients"):
                arrays[name + "_" + key] = getattr(head, key)
        with Path(path).open("xb") as stream:
            np.savez_compressed(stream, **arrays)

    @classmethod
    def load(cls, path):
        model = cls()
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            if metadata["physical_execution_enabled"] is not False:
                raise ValueError("Unexpected deployment-enabled prototype artifact")
            model.spec, model.horizon = metadata["spec"], metadata["horizon"]
            model.calibration = metadata["calibration"]
            model.fit_stage_positive_episodes = metadata["fit_stage_positive_episodes"]
            model.support = saved["support"].copy()
            for name in ("dynamics", "risk", "readout"):
                head = _RidgeEnsemble()
                for key in ("mean", "scale", "projection", "bias", "coefficients"):
                    setattr(head, key, saved[name + "_" + key].copy())
                setattr(model, name, head)
        return model
