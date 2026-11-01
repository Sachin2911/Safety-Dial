"""Durable common-panel CMA, random search and categorical REINFORCE training.

All execution belongs on the declared rented GPU. A completed checkpoint can resume;
a started-only or failed episode cannot be silently replayed. Budgeting uses actual
reported actions for finalized episodes and a labelled conservative charge for an
unfinalized attempt. Equal allowances do not imply equal actual consumption.
"""

from dataclasses import asdict, dataclass, field, fields
import datetime as dt
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import pickle
import traceback

import numpy as np


class IncompleteJobError(RuntimeError):
    pass


class BudgetExhausted(RuntimeError):
    pass


class TechnicalFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    directory: Path
    case_id: str
    case: dict
    environment_seed: int
    policy_seed: int
    parameters: tuple
    mode: str
    max_actions: int
    temperature: float
    method: str
    repeat_id: int
    stage: str
    generation: int
    candidate_index: int

    @property
    def folder(self):
        return self.directory

    @property
    def seed(self):
        return self.environment_seed

    @property
    def weights(self):
        return self.parameters


@dataclass
class EpisodeOutcome:
    safe_completion: bool
    violation: bool
    unfinished: bool
    engaged: bool
    progress: float
    environment_actions: int
    resources: dict = field(default_factory=dict)
    decisions: list = field(default_factory=list)
    error: str | None = None
    extra: dict = field(default_factory=dict)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path, value, *, compact=False):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=None if compact else 2, allow_nan=False) + "\n")
    temporary.replace(path)


def panel_summary(rows):
    if not rows:
        raise ValueError("A candidate requires its entire nonempty frozen panel")
    for row in rows:
        if row.get("error") or not 0 <= row["progress"] <= 1:
            raise ValueError("Invalid/technical episodes cannot supply optimizer fitness")
    return {
        "episodes": len(rows),
        "safe_completion": sum(r["safe_completion"] for r in rows),
        "violation": sum(r["violation"] for r in rows),
        "unfinished": sum(r["unfinished"] for r in rows),
        "engaged": sum(r["engaged"] for r in rows),
        "progress_sum": float(sum(r["progress"] for r in rows)),
        "environment_actions": sum(r["environment_actions"] for r in rows),
    }


def panel_fitness(rows):
    summary = panel_summary(rows)
    return (summary["safe_completion"], -summary["violation"], summary["progress_sum"])


def lexicographic_episode_reward(row, panel_size):
    """Order embedding, not a tuned safety penalty: panel sums preserve the tuple rank.

    Progress sum/(N+1) is less than one. One violation therefore dominates all
    progress, and one safe completion (N+1) dominates N violations plus progress.
    """
    if panel_size < 1 or row.get("error") or not 0 <= row["progress"] <= 1:
        raise ValueError("Invalid shared finite-panel reward")
    return (
        (panel_size + 1) * int(row["safe_completion"])
        - int(row["violation"])
        + row["progress"] / (panel_size + 1)
    )


class CampaignTraining:
    """One frozen optimizer/repeat, one actual-action allowance, one durable run tree.

    evaluate(JobSpec, report_progress) returns EpisodeOutcome (or its dict form).
    report_progress(actions, resources) uses cumulative episode counts and writes
    only a compact per-job live snapshot. REINFORCE traces stay in compressed NPZ;
    campaign state stores hashes and episode references, never full feature tables.
    """

    def __init__(self, protocol, output, evaluate, *, policy_log_probability=None, resume=False):
        import torch

        self.protocol, self.root, self.evaluate = protocol, Path(output), evaluate
        self.log_probability = policy_log_probability
        if protocol.get("schema") != "lewm-campaign-training-v1":
            raise ValueError("Explicit training protocol schema required")
        if os.environ.get("RUNPOD_POD_ID") not in protocol["allowed_pod_ids"]:
            raise RuntimeError("Training is restricted to the declared rented GPU")
        if not torch.cuda.is_available():
            raise RuntimeError("The declared rented GPU must be available")
        self.method = protocol["method"]
        if self.method not in ("cma", "random", "reinforce"):
            raise ValueError("Unknown frozen optimizer")
        self.initial = np.asarray(protocol["initial_parameters"], dtype=np.float64)
        if self.initial.ndim != 1 or not np.isfinite(self.initial).all():
            raise ValueError("Finite persistent parameters required")
        if not protocol["training_panel"] or protocol["max_episode_actions"] < 1:
            raise ValueError("A frozen nonempty panel and positive episode cap are required")
        if protocol["max_environment_actions"] < 1 or protocol["max_candidate_panels"] < 1:
            raise ValueError("Positive action/candidate limits required")
        if protocol["population"] < 2 or protocol["sigma"] <= 0:
            raise ValueError("A nondegenerate candidate distribution is required")
        low, high = [
            np.broadcast_to(np.asarray(v, dtype=float), self.initial.shape)
            for v in protocol["parameter_bounds"]
        ]
        if not np.isfinite(low).all() or not np.isfinite(high).all() or not (low < high).all():
            raise ValueError("Finite ordered parameter bounds required")
        if not ((low <= self.initial) & (self.initial <= high)).all():
            raise ValueError("The shared initial anchor lies outside parameter bounds")
        learner = protocol["reinforce"]
        if (
            not all(
                math.isfinite(learner[k]) and learner[k] > 0
                for k in ("learning_rate", "temperature", "max_grad_norm")
            )
            or not 0 <= learner["baseline_decay"] < 1
        ):
            raise ValueError("Invalid frozen categorical learner hyperparameters")
        self.source_identity = {
            "harness_sha256": _digest(Path(__file__)),
            "declared_source_ids": protocol["source_ids"],
        }
        if policy_log_probability is not None:
            source = Path(inspect.getfile(policy_log_probability))
            self.source_identity["policy_log_probability_source_sha256"] = _digest(source)
            self.source_identity["policy_log_probability_name"] = (
                policy_log_probability.__qualname__
            )
        self.identity = _hash({"protocol": protocol, "sources": self.source_identity})
        self.checkpoint_path = self.root / "checkpoint.pkl"
        self.es, self.parameter, self.adam = None, None, None
        self.random = np.random.default_rng(protocol["optimizer_seed"])
        self.cma_global_state = np.random.RandomState(protocol["optimizer_seed"]).get_state()
        if resume:
            with self.checkpoint_path.open("rb") as stream:
                saved = pickle.load(stream)  # Own trusted run-tree checkpoint only.
            if saved["identity"] != self.identity:
                raise ValueError("Frozen training protocol/source identity changed")
            self.state, self.es = saved["state"], saved["es"]
            self.random.bit_generator.state = saved["random_state"]
            self.cma_global_state = saved["cma_global_state"]
            if self.method == "reinforce":
                self._actor(saved["parameter"], saved["adam"])
        else:
            if self.root.exists():
                raise FileExistsError("Use a fresh run tree or an explicit verified resume")
            self.root.mkdir(parents=True)
            (self.root / "jobs").mkdir()
            self.state = {
                "identity": self.identity,
                "method": self.method,
                "repeat_id": protocol["repeat_id"],
                "phase": "training",
                "active": None,
                "anchor": None,
                "best_trained": None,
                "pending": None,
                "history": [],
                "optimizer_updates": 0,
                "candidate_panels_completed": 0,
                "training_panels_completed": 0,
                "generation": 0,
                "baseline_by_case": {},
                "actual_environment_actions": 0,
                "charged_environment_actions": 0,
                "resource_totals": {},
                "episode_starts": 0,
                "completed_job_ids": [],
                "unfinalized_job_ids": [],
                "action_accounting_exact": True,
            }
            _json(
                self.root / "source-freeze.json",
                {
                    "protocol": protocol,
                    "protocol_sha256": _hash(protocol),
                    "sources": self.source_identity,
                    "identity": self.identity,
                },
            )
            if self.method == "cma":
                import cma

                rng = np.random.RandomState(protocol["optimizer_seed"])
                outer = np.random.get_state()
                try:
                    self.es = cma.CMAEvolutionStrategy(
                        self.initial,
                        protocol["sigma"],
                        {
                            "seed": protocol["optimizer_seed"],
                            "randn": rng.randn,
                            "popsize": protocol["population"],
                            "CMA_diagonal": True,
                            "bounds": protocol["parameter_bounds"],
                            "verbose": -9,
                            "verb_disp": 0,
                            "verb_log": 0,
                        },
                    )
                finally:
                    np.random.set_state(outer)
            if self.method == "reinforce":
                self._actor(self.initial, None)
        self._job_ledger = {}
        self._reconcile()
        self._save()

    def _actor(self, values, adam_state):
        import torch

        if self.log_probability is None:
            raise ValueError("REINFORCE requires the exact controller log-probability function")
        self.parameter = torch.nn.Parameter(
            torch.as_tensor(values, dtype=torch.float32, device="cuda")
        )
        self.adam = torch.optim.Adam(
            [self.parameter], lr=self.protocol["reinforce"]["learning_rate"]
        )
        if adam_state is not None:
            self.adam.load_state_dict(adam_state)

    def _reconcile(self):
        self.state.update(
            actual_environment_actions=0,
            charged_environment_actions=0,
            resource_totals={},
            episode_starts=0,
            completed_job_ids=[],
            unfinalized_job_ids=[],
            action_accounting_exact=True,
        )
        for folder in sorted((self.root / "jobs").iterdir()):
            started_path = folder / "started.json"
            if not started_path.exists():
                continue
            started = json.loads(started_path.read_text())
            if started["identity"] != self.identity:
                raise ValueError("Job belongs to another frozen training run")
            self.state["episode_starts"] += 1
            result = folder / "result.json"
            live = folder / "live.json"
            if result.exists():
                row = json.loads(result.read_text())
                progress = {
                    "environment_actions": row["environment_actions"],
                    "resources": row["resources"],
                }
                finalized = row.get("accounting_finalized", False)
                if row.get("status") == "complete":
                    self.state["completed_job_ids"].append(folder.name)
                else:
                    self.state["unfinalized_job_ids"].append(folder.name)
            else:
                progress = (
                    json.loads(live.read_text())
                    if live.exists()
                    else {"environment_actions": 0, "resources": {}}
                )
                finalized = False
                self.state["unfinalized_job_ids"].append(folder.name)
            observed = int(progress["environment_actions"])
            charge = observed if finalized else max(observed, started["job"]["max_actions"])
            self._job_ledger[folder.name] = {**progress, "charge": charge, "finalized": finalized}
            self.state["actual_environment_actions"] += observed
            self.state["charged_environment_actions"] += charge
            self.state["action_accounting_exact"] &= finalized
            for key, value in progress["resources"].items():
                self.state["resource_totals"][key] = (
                    self.state["resource_totals"].get(key, 0) + value
                )
        if self.state["active"] in self.state["completed_job_ids"]:
            self.state["active"] = None

    @staticmethod
    def _cpu(value):
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {k: CampaignTraining._cpu(v) for k, v in value.items()}
        if isinstance(value, list):
            return [CampaignTraining._cpu(v) for v in value]
        if isinstance(value, tuple):
            return tuple(CampaignTraining._cpu(v) for v in value)
        return value

    def _save(self):
        self.state["remaining_action_allowance"] = max(
            0, self.protocol["max_environment_actions"] - self.state["charged_environment_actions"]
        )
        self.state["allowance_is_not_equal_actual_consumption"] = True
        temporary = self.checkpoint_path.with_suffix(".part")
        payload = {
            "identity": self.identity,
            "state": self.state,
            "es": self.es,
            "random_state": self.random.bit_generator.state,
            "cma_global_state": self.cma_global_state,
            "parameter": None if self.parameter is None else self.parameter.detach().cpu().numpy(),
            "adam": None if self.adam is None else self._cpu(self.adam.state_dict()),
        }
        with temporary.open("wb") as stream:
            pickle.dump(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.checkpoint_path)
        _json(self.root / "state.json", self.state)

    def _reserve(self, jobs):
        if self.state["unfinalized_job_ids"]:
            raise IncompleteJobError(
                "No replay of failed/incomplete jobs: "
                + ",".join(self.state["unfinalized_job_ids"])
            )
        deadline = dt.datetime.fromisoformat(self.protocol["deadline_utc"])
        reserve = self.protocol.get("preservation_reserve_seconds", 600)
        if dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=reserve) >= deadline:
            raise BudgetExhausted("deadline_preservation_reserve")
        if self.state["episode_starts"] + jobs > self.protocol["max_episode_starts"]:
            raise BudgetExhausted("episode_start_allowance")
        required = jobs * self.protocol["max_episode_actions"]
        if (
            self.state["charged_environment_actions"] + required
            > self.protocol["max_environment_actions"]
        ):
            raise BudgetExhausted("insufficient_actions_for_complete_candidate_panel")

    def _report(self, job, actions, resources=None, *, finalized=False):
        if int(actions) != actions or actions < 0:
            raise ValueError("Cumulative action count must be a nonnegative integer")
        actions = int(actions)
        resources = {} if resources is None else dict(resources)
        resources["environment_actions"] = actions
        resources = {
            str(k): float(v) if isinstance(v, (float, np.floating)) else int(v)
            for k, v in resources.items()
        }
        if any(not math.isfinite(v) or v < 0 for v in resources.values()):
            raise ValueError("Resource counters must be finite and nonnegative")
        previous = self._job_ledger.get(
            job.job_id, {"environment_actions": 0, "resources": {}, "charge": 0}
        )
        if actions < previous["environment_actions"] or any(
            value < previous["resources"].get(key, 0) for key, value in resources.items()
        ):
            raise ValueError("Cumulative resource counters regressed")
        merged = {**previous["resources"], **resources}
        self.state["actual_environment_actions"] += actions - previous["environment_actions"]
        self.state["charged_environment_actions"] += actions - previous["charge"]
        for key, value in merged.items():
            self.state["resource_totals"][key] = (
                self.state["resource_totals"].get(key, 0)
                + value
                - previous["resources"].get(key, 0)
            )
        progress = {
            "environment_actions": actions,
            "resources": merged,
            "charge": actions,
            "finalized": finalized,
        }
        self._job_ledger[job.job_id] = progress
        _json(
            job.directory / "live.json",
            {k: v for k, v in progress.items() if k != "charge"},
            compact=True,
        )
        if (
            actions > job.max_actions
            or self.state["charged_environment_actions"] > self.protocol["max_environment_actions"]
        ):
            raise TechnicalFailure(
                "Evaluator exceeded the hard actual-action cap; all excess was recorded"
            )

    def _job(self, job):
        spec = asdict(job)
        spec["directory"] = str(job.directory)
        identity = _hash(spec)
        result_path = job.directory / "result.json"
        if result_path.exists():
            row = json.loads(result_path.read_text())
            if row["input_sha256"] != identity or row["identity"] != self.identity:
                raise ValueError("Saved job inputs changed")
            if row["status"] != "complete":
                raise IncompleteJobError("Failed job is retained, never replayed: " + job.job_id)
            return row
        if job.directory.exists():
            raise IncompleteJobError("Started-only job is retained, never replayed: " + job.job_id)
        self._reserve(1)
        job.directory.mkdir()
        _json(
            job.directory / "started.json",
            {"identity": self.identity, "input_sha256": identity, "job": spec},
        )
        self.state["episode_starts"] += 1
        self.state["active"] = job.job_id
        self._save()
        finalized = False
        try:
            outcome = self.evaluate(
                job, lambda actions, resources=None: self._report(job, actions, resources)
            )
            if isinstance(outcome, EpisodeOutcome):
                row = {
                    f.name: getattr(outcome, f.name)
                    for f in fields(outcome)
                    if f.name != "decisions"
                }
                decisions = outcome.decisions
            else:
                row = dict(outcome)
                decisions = row.pop("decisions", [])
            self._report(job, row["environment_actions"], row.get("resources", {}), finalized=True)
            finalized = True
            row["resources"] = self._job_ledger[job.job_id]["resources"]
            for field_name in ("safe_completion", "violation", "unfinished", "engaged"):
                if not isinstance(row[field_name], (bool, np.bool_)):
                    raise ValueError("Episode outcome flags must be booleans")
                row[field_name] = bool(row[field_name])
            if not math.isfinite(row["progress"]) or not 0 <= row["progress"] <= 1:
                raise ValueError("Original-task progress must be explicitly bounded in [0,1]")
            if row["safe_completion"] and (row["violation"] or row["unfinished"]):
                raise ValueError("A safe completion cannot be violating or unfinished")
            trace = self._save_decisions(job, decisions) if decisions else None
            row.update(
                identity=self.identity,
                input_sha256=identity,
                job_id=job.job_id,
                case_id=job.case_id,
                decision_trace=trace,
                accounting_finalized=True,
                status="failed" if row.get("error") else "complete",
            )
            _json(result_path, row)
            if row.get("error"):
                raise TechnicalFailure("Technical episode retained: " + job.job_id)
            self.state["completed_job_ids"].append(job.job_id)
            self.state["active"] = None
            self._save()
            return row
        except BaseException:
            progress = self._job_ledger.get(
                job.job_id, {"environment_actions": 0, "resources": {}, "charge": 0}
            )
            if not finalized:
                charge = max(progress["environment_actions"], job.max_actions)
                self.state["charged_environment_actions"] += charge - progress["charge"]
                progress.update(charge=charge, finalized=False)
                self._job_ledger[job.job_id] = progress
                self.state["action_accounting_exact"] = False
            if job.job_id not in self.state["unfinalized_job_ids"]:
                self.state["unfinalized_job_ids"].append(job.job_id)
            _json(
                job.directory / "failure.json",
                {
                    "error": traceback.format_exc(),
                    "resources": progress["resources"],
                    "observed_environment_actions": progress["environment_actions"],
                    "budget_charge": progress["charge"],
                    "accounting_finalized": finalized,
                    "unfinalized_charge_is_conservative_not_exact_actual_work": not finalized,
                },
            )
            self._save()
            raise

    def _save_decisions(self, job, decisions):
        import torch

        features, masks, selected, temperatures, behavior_logps = [], [], [], [], []
        for decision in decisions:
            if isinstance(decision, dict):
                f, m, i = decision["features"], decision["available"], decision["chosen_index"]
                temperature = decision.get("temperature", job.temperature)
            else:
                f, m, i = decision
                temperature = job.temperature
            f, m = np.asarray(f, dtype=np.float32), np.asarray(m, dtype=bool)
            if f.ndim != 2 or m.shape != (len(f),) or not np.isfinite(f).all():
                raise ValueError("Malformed categorical target-decision trace")
            if (
                not 0 <= int(i) < len(f)
                or not m[int(i)]
                or not math.isfinite(temperature)
                or temperature <= 0
            ):
                raise ValueError("Chosen target was unavailable or temperature invalid")
            features.append(f)
            masks.append(m)
            selected.append(int(i))
            temperatures.append(float(temperature))
            if self.log_probability is None:
                raise ValueError(
                    "A behavior trace requires the exact categorical log-probability law"
                )
            with torch.no_grad():
                value = self.log_probability(
                    torch.as_tensor(f, device="cuda"),
                    torch.as_tensor(m, device="cuda"),
                    torch.as_tensor(job.parameters, device="cuda", dtype=torch.float32),
                    int(i),
                    float(temperature),
                )
            if not torch.isfinite(value):
                raise ValueError("Chosen categorical behavior log probability is nonfinite")
            behavior_logps.append(float(value))
        if len({f.shape for f in features}) != 1:
            raise ValueError("Frozen candidate-bank shape changed within an episode")
        path = job.directory / "decisions.npz"
        np.savez_compressed(
            path,
            features=np.stack(features),
            available=np.stack(masks),
            selected=selected,
            temperature=temperatures,
            behavior_log_probability=behavior_logps,
        )
        return {
            "path": str(path.relative_to(job.directory.parent.parent)),
            "sha256": _digest(path),
            "decisions": len(selected),
            "feature_shape": list(features[0].shape),
        }

    def _panel(self, label, parameters, mode, *, generation, candidate_index):
        specs = []
        for index, case in enumerate(self.protocol["training_panel"]):
            key = f"{label}_p{index:03}"
            policy_seed = int.from_bytes(
                hashlib.sha256(f"{self.protocol['optimizer_seed']}/{key}".encode()).digest()[:4],
                "big",
            ) % (2**31 - 1)
            specs.append(
                JobSpec(
                    key,
                    self.root / "jobs" / key,
                    case["case_id"],
                    case,
                    int(case["seed"]),
                    policy_seed,
                    tuple(float(x) for x in parameters),
                    mode,
                    self.protocol["max_episode_actions"],
                    self.protocol["reinforce"]["temperature"],
                    self.method,
                    self.protocol["repeat_id"],
                    label,
                    generation,
                    candidate_index,
                )
            )
        for job in specs:
            if job.directory.exists() and not (job.directory / "result.json").exists():
                raise IncompleteJobError(
                    "Incomplete frozen panel cannot be replayed: " + job.job_id
                )
        self._reserve(sum(not (job.directory / "result.json").exists() for job in specs))
        rows = [self._job(job) for job in specs]
        panel_summary(rows)
        return rows

    def _nominate(self, parameters, rows, provenance):
        candidate = {
            "parameters": list(parameters),
            "fitness": list(panel_fitness(rows)),
            "summary": panel_summary(rows),
            "job_ids": [r["job_id"] for r in rows],
            "provenance": provenance,
            "parameters_equal_initial": bool(np.array_equal(parameters, self.initial)),
        }
        if provenance["kind"] == "initial_anchor":
            self.state["anchor"] = candidate
        elif not candidate["parameters_equal_initial"]:
            best = self.state["best_trained"]
            if best is None or tuple(candidate["fitness"]) > tuple(best["fitness"]):
                self.state["best_trained"] = candidate
        return candidate

    def _cma_call(self, operation, *arguments):
        outer = np.random.get_state()
        try:
            np.random.set_state(self.cma_global_state)
            return getattr(self.es, operation)(*arguments)
        finally:
            self.cma_global_state = np.random.get_state()
            np.random.set_state(outer)

    def _population(self):
        if self.state["pending"] is None:
            population = self.protocol["population"]
            if self.method == "cma":
                candidates = np.asarray(self._cma_call("ask"))
            else:
                candidates = self.initial + self.random.normal(
                    0, self.protocol["sigma"], (population, len(self.initial))
                )
                candidates = np.clip(candidates, *self.protocol["parameter_bounds"])
            self.state["pending"] = {
                "parameters": candidates.tolist(),
                "order": self.random.permutation(population).tolist(),
                "scores": {},
            }
            self._save()
        pending = self.state["pending"]
        generation = self.state["generation"]
        for index in pending["order"]:
            if str(index) in pending["scores"]:
                continue
            if self.state["candidate_panels_completed"] >= self.protocol["max_candidate_panels"]:
                return False
            parameters = pending["parameters"][index]
            rows = self._panel(
                f"{self.method}_g{generation:03}_c{index:03}",
                parameters,
                "deterministic",
                generation=generation,
                candidate_index=index,
            )
            nominee = self._nominate(
                parameters,
                rows,
                {
                    "kind": "trained_candidate",
                    "method": self.method,
                    "repeat_id": self.protocol["repeat_id"],
                    "generation": generation,
                    "candidate_index": index,
                    "completed_updates_before_proposal": self.state["optimizer_updates"],
                },
            )
            pending["scores"][str(index)] = nominee["fitness"]
            self.state["candidate_panels_completed"] += 1
            self.state["training_panels_completed"] += 1
            _json(self.root / f"candidate-{generation:03}-{index:03}.json", nominee)
            self._save()
        scores = [tuple(pending["scores"][str(i)]) for i in range(len(pending["parameters"]))]
        ranks = [sum(other > score for other in scores) for score in scores]
        if self.method == "cma":
            self._cma_call("tell", pending["parameters"], ranks)
            self.state["optimizer_updates"] += 1
        history = {
            "generation": generation,
            "parameters": pending["parameters"],
            "fitness": [list(s) for s in scores],
            "minimization_ranks": ranks,
            "genuine_evolutionary_update": self.method == "cma",
        }
        self.state["history"].append(history)
        _json(self.root / f"generation-{generation:03}.json", history)
        self.state["generation"] += 1
        self.state["pending"] = None
        self._save()
        return True

    def _policy_update(self, rows):
        import torch

        cfg = self.protocol["reinforce"]
        self.adam.zero_grad()
        losses, baselines, rewards, advantages, returns_by_case = [], [], [], [], {}
        decision_count = 0
        for row in rows:
            reward = lexicographic_episode_reward(row, len(rows))
            baseline = self.state["baseline_by_case"].get(row["case_id"], 0.0)
            advantage = reward - baseline
            trace = row["decision_trace"]
            if trace is None:
                raise ValueError("Categorical training requires actual target-decision traces")
            path = self.root / trace["path"]
            if _digest(path) != trace["sha256"]:
                raise ValueError("Saved categorical decision trace changed")
            logps = []
            with np.load(path, allow_pickle=False) as data:
                for feature, mask, choice, temperature, behavior_logp in zip(
                    data["features"],
                    data["available"],
                    data["selected"],
                    data["temperature"],
                    data["behavior_log_probability"],
                    strict=True,
                ):
                    f = torch.as_tensor(feature, device="cuda", dtype=torch.float32)
                    m = torch.as_tensor(mask, device="cuda", dtype=torch.bool)
                    logp = self.log_probability(
                        f, m, self.parameter, int(choice), float(temperature)
                    )
                    if not torch.isfinite(logp) or not logp.requires_grad:
                        raise ValueError("Controller log probability is nonfinite or detached")
                    if abs(float(logp.detach()) - float(behavior_logp)) > 1e-5:
                        raise ValueError("Stored rollout is not on-policy for the current genotype")
                    logps.append(logp)
            if not logps:
                raise ValueError("An empty decision sequence cannot train this selector")
            losses.append(-advantage * torch.stack(logps).sum())
            decision_count += len(logps)
            returns_by_case.setdefault(row["case_id"], []).append(reward)
            rewards.append(reward)
            baselines.append(baseline)
            advantages.append(advantage)
        loss = torch.stack(losses).mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([self.parameter], cfg["max_grad_norm"])
        if not torch.isfinite(norm):
            raise ValueError("Nonfinite categorical policy gradient")
        before = self.parameter.detach().clone()
        self.adam.step()
        with torch.no_grad():
            low, high = self.protocol["parameter_bounds"]
            self.parameter.clamp_(
                min=torch.as_tensor(low, device="cuda"), max=torch.as_tensor(high, device="cuda")
            )
        decay = cfg["baseline_decay"]
        for case_id, values in returns_by_case.items():
            previous = self.state["baseline_by_case"].get(case_id, 0.0)
            self.state["baseline_by_case"][case_id] = decay * previous + (1 - decay) * float(
                np.mean(values)
            )
        return {
            "loss": float(loss.detach()),
            "gradient_norm_before_clip": float(norm),
            "parameter_change_l2": float((self.parameter - before).norm().detach()),
            "episode_rewards": rewards,
            "previous_case_baselines": baselines,
            "episode_advantages": advantages,
            "categorical_decisions": decision_count,
            "reward_rule": "(panel_size+1)*safe_completion-violation+progress/(panel_size+1)",
            "gradient_rule": "Mean over episodes of -advantage times sum of categorical target log probabilities; previous-update case EMA baseline. No parameter-space score gradient.",
        }

    def _reinforce(self):
        if self.state["pending"] is None:
            self._reserve(2 * len(self.protocol["training_panel"]))
            update = self.state["optimizer_updates"]
            self.state["pending"] = {
                "update": update,
                "phase": "sample",
                "parameters": self.parameter.detach().cpu().tolist(),
            }
            self._save()
        pending = self.state["pending"]
        update = pending["update"]
        if pending["phase"] == "sample":
            rows = self._panel(
                f"reinforce_u{update:03}_sample",
                pending["parameters"],
                "categorical",
                generation=update,
                candidate_index=0,
            )
            statistics = self._policy_update(rows)
            self.state["training_panels_completed"] += 1
            self.state["optimizer_updates"] += 1
            pending.update(
                phase="evaluate",
                parameters=self.parameter.detach().cpu().tolist(),
                statistics=statistics,
                sample_job_ids=[r["job_id"] for r in rows],
            )
            self._save()
        rows = self._panel(
            f"reinforce_u{update:03}_evaluate",
            pending["parameters"],
            "deterministic",
            generation=update,
            candidate_index=0,
        )
        nominee = self._nominate(
            pending["parameters"],
            rows,
            {
                "kind": "trained_candidate",
                "method": "reinforce",
                "repeat_id": self.protocol["repeat_id"],
                "update": update,
                "completed_updates_before_evaluation": self.state["optimizer_updates"],
            },
        )
        history = {
            "update": update,
            "nominee": nominee,
            "statistics": pending["statistics"],
            "sample_job_ids": pending["sample_job_ids"],
        }
        self.state["history"].append(history)
        self.state["candidate_panels_completed"] += 1
        self.state["training_panels_completed"] += 1
        _json(self.root / f"update-{update:03}.json", history)
        self.state["pending"] = None
        self._save()

    def run(self):
        if self.state["unfinalized_job_ids"]:
            raise IncompleteJobError(
                "Resolve retained failures; they cannot be automatically replayed"
            )
        if self.state["phase"] == "complete":
            return self.state
        try:
            if self.state["anchor"] is None:
                rows = self._panel(
                    "anchor", self.initial, "deterministic", generation=-1, candidate_index=-1
                )
                self._nominate(
                    self.initial.tolist(), rows, {"kind": "initial_anchor", "trained": False}
                )
                self._save()
            while self.state["candidate_panels_completed"] < self.protocol["max_candidate_panels"]:
                if self.method == "reinforce":
                    self._reinforce()
                elif not self._population():
                    break
            reason = "candidate_panel_allowance"
        except BudgetExhausted as exc:
            reason = str(exc)
        except KeyboardInterrupt:
            self.state["phase"] = "interrupted"
            self._save()
            raise
        except Exception:
            self.state["phase"] = "technical_failure"
            self.state["error"] = traceback.format_exc()
            self._save()
            raise
        self.state.update(phase="complete", stop_reason=reason)
        anchor, trained = self.state["anchor"], self.state["best_trained"]
        selected = anchor
        if trained is not None and (
            anchor is None or tuple(trained["fitness"]) > tuple(anchor["fitness"])
        ):
            selected = trained
        self.state["selected_training_nominee"] = selected
        self.state["selected_is_trained"] = selected is not None and selected is trained
        self.state["optimizer_advantage_established"] = False
        self.state["held_out_evaluation_performed"] = False
        self._save()
        _json(
            self.root / "nominees.json",
            {
                "anchor": anchor,
                "trained": trained,
                "selected_on_training": selected,
                "selected_is_trained": self.state["selected_is_trained"],
            },
        )
        return self.state
