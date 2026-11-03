"""Single-GPU matched PPO/CMA/random lanes, with independently frozen host budgets."""

import collections
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/ppo-20261101/run-v1"
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "experiments/scripts")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import adapter_pilot_v2 as qualified  # noqa: E402
import replicate_adapter as prior  # noqa: E402
import run_mechanism as mechanism  # noqa: E402
from helpers.mechanism import SearchMap, projection_matrix  # noqa: E402
from helpers.pilot_common import digest, save, score  # noqa: E402
from helpers.ppo_adapter import CoordinatePolicy, prepare_episodes, update  # noqa: E402
from helpers.recurrent_adapter import AdapterSpec, RecurrentActionAdapter  # noqa: E402
from helpers.replication_budget import eligible  # noqa: E402
from ppo_worker import run_episode  # noqa: E402


class SerialPool:
    """Retain logical ES reservations/order while executing on one physical GPU."""

    def __init__(self, deadline):
        import h5py
        import anchored_pipeline as ap

        self.data = h5py.File(ROOT / "data/tworoom.h5", "r")
        self.bank, _ = ap.make_bank(self.data)
        _, self.native = ap.native_model()
        self.deadline = deadline

    def episode(self, spec, parameters=None, limit=200, policy=None, sampling_seed=0):
        return run_episode(
            spec,
            parameters,
            self.native,
            self.bank,
            self.data,
            OUT,
            self.deadline,
            0,
            limit,
            policy,
            sampling_seed,
        )

    def batch(self, jobs):
        result = {}
        for job in jobs:
            quota = job.get("action_quota", 200 * len(job["episodes"]))
            spent, rows = 0, []
            for spec in job["episodes"]:
                if spent == quota:
                    break
                row = self.episode(spec, job.get("parameters"), min(200, quota - spent))
                rows.append(row)
                spent += row["resources"]["environment_actions"]
                assert spent <= quota
            result[job["id"]] = rows
        return result

    def close(self):
        self.data.close()


def freeze(config, host):
    prior.OUT = OUT
    prior.freeze()
    frozen = json.loads((OUT / "freeze.json").read_text())
    files = [
        ROOT / "MatchedPPO.md",
        Path(__file__),
        ROOT / "experiments/scripts/ppo_worker.py",
        ROOT / "experiments/helpers/ppo_adapter.py",
        ROOT / "experiments/helpers/mechanism.py",
        ROOT / "experiments/scripts/run_mechanism.py",
        ROOT / "experiments/scripts/audit_matched_ppo.py",
        ROOT / "experiments/scripts/launch_matched_ppo.py",
        ROOT / "experiments/scripts/preserve_matched_ppo.py",
        ROOT / "experiments/tests/test_ppo_adapter.py",
        *sorted((ROOT / "configs/matched-ppo").glob("*")),
    ]
    for path in files:
        name = str(path.relative_to(ROOT))
        frozen["sources"][name] = digest(path)
        target = OUT / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    frozen.pop("instance_id", None)
    frozen.pop("machine_id", None)
    frozen.update(study="matched-objective PPO", runtime_hostname=socket.gethostname(), host=host)
    save(OUT / "freeze.json", frozen)


def search_ppo(pool, repeat, mapping, config, panel, summary, record):
    key = f"{repeat['id']}-{mapping.representation}-ppo"
    folder = OUT / "searches" / key
    folder.mkdir(parents=True)
    policy = CoordinatePolicy(mapping, repeat["ppo_seed"])
    optimizer = torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad], lr=config["learning_rate"]
    )
    rng = np.random.default_rng(repeat["shuffle_seed"])
    spent, iteration, episode_index, updates = 0, 0, 0, 0
    nominees = []
    budget = config["training_actions_per_search"]
    while spent < budget:
        cid = f"{key}-{iteration:05d}"
        torch.save(policy.state_dict(), folder / f"policy-{iteration:05d}.pt")
        save(
            folder / f"iteration-{iteration:05d}-intent.json",
            dict(
                spent_before=spent,
                collection_quota=min(config["rollout_actions"], budget - spent),
                checkpoint=cid,
            ),
        )
        collection, used = [], 0
        allowance = min(config["rollout_actions"], budget - spent)
        while used < allowance:
            case = panel["training"][episode_index % 8]
            sampling_seed = repeat["ppo_seed"] * 100000 + episode_index
            spec = mechanism.spec(
                "ppo-collection",
                case,
                "ppo",
                panel["training_seed"],
                f"{cid}-collect-{episode_index:06d}",
                repeat["id"],
                cid,
                mapping.representation,
                key,
            )
            spec["sampling_seed"] = sampling_seed
            row = pool.episode(
                spec, limit=min(200, allowance - used), policy=policy, sampling_seed=sampling_seed
            )
            actions = row["resources"]["environment_actions"]
            used += actions
            spent += actions
            episode_index += 1
            collection.append(row)
            record([row])
            assert spent <= budget
        episodes = prepare_episodes(collection, OUT)
        logs = update(policy, optimizer, episodes, rng, epochs=config["epochs"]) if episodes else []
        updates += int(bool(logs))
        torch.save(
            dict(policy=policy.state_dict(), optimizer=optimizer.state_dict()),
            folder / f"updated-{iteration:05d}.pt",
        )
        parameters = policy.export()
        np.savez(
            OUT / "candidates" / (cid + ".npz"),
            genotype=policy.coordinates.detach().numpy(),
            parameters=parameters,
        )
        quota = min(1600, budget - spent)
        if quota:
            specs = [
                mechanism.spec(
                    "training",
                    case,
                    "ppo",
                    panel["training_seed"],
                    cid,
                    repeat["id"],
                    cid,
                    mapping.representation,
                    key,
                )
                for case in panel["training"]
            ]
            rows = pool.batch(
                [dict(id=cid, parameters=parameters, action_quota=quota, episodes=specs)]
            )[cid]
            actual = sum(r["resources"]["environment_actions"] for r in rows)
            spent += actual
            entry = dict(
                id=cid,
                repeat=repeat["id"],
                arm="ppo",
                optimizer="ppo",
                representation=mapping.representation,
                search=key,
                generation=iteration,
                eligible=eligible(rows),
                actions=actual,
                quota=quota,
                score=None,
                episodes=len(rows),
            )
            if entry["eligible"]:
                entry.update(score(rows))
                nominees.append(entry)
            summary["candidates"].append(entry)
            record(rows)
        else:
            actual = 0
        save(
            folder / f"iteration-{iteration:05d}.json",
            dict(
                iteration=iteration,
                collection_ids=[r["id"] for r in collection],
                collection_actions=used,
                eligible_collection_ids=[e["id"] for e in episodes],
                selection_actions=actual,
                spent=spent,
                update_performed=bool(logs),
                optimization=logs,
                updates=updates,
            ),
        )
        save(OUT / "summary.json", summary)
        print(
            json.dumps(dict(search=key, iteration=iteration, spent=spent, updates=updates)),
            flush=True,
        )
        iteration += 1
    assert spent == budget and nominees
    best = max(nominees, key=lambda c: c["score"])
    result = dict(
        repeat=repeat["id"],
        optimizer="ppo",
        arm="ppo",
        representation=mapping.representation,
        dimension=mapping.dimension,
        actions=spent,
        updates=updates,
        iterations=iteration,
        nominee=best,
        eligible=len(nominees),
        additional_log_std_parameters=4,
        critic_parameters=sum(p.numel() for p in policy.critic.parameters()),
    )
    save(folder / "complete.json", result)
    summary["searches"][key] = result
    return best


def run():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = json.loads((ROOT / "configs/matched-ppo/run.json").read_text())
    panel = json.loads((ROOT / "configs/matched-ppo/panel.json").read_text())
    host = config["hosts"][socket.gethostname()]
    repeats = [r for r in config["repeats"] if r["id"] in host["repeats"]]
    assert (OUT.parent / "software-checks-passed").exists()
    assert importlib.metadata.version("cma") == "4.4.4"
    assert torch.__version__ == "2.4.1+cu124"
    freeze(config, host)
    mechanism.OUT = OUT
    started = time.monotonic()
    deadline = started + config["run_seconds"]
    summary = dict(
        status="running",
        phase="engineering",
        host=host,
        results=[],
        candidates=[],
        searches={},
        resources={},
    )
    counts = collections.Counter()
    pool = None
    (OUT / "initialization").mkdir()
    (OUT / "candidates").mkdir()

    def record(rows):
        summary["results"].extend(rows)
        for row in rows:
            counts.update(row["resources"])
        summary["resources"] = dict(counts)
        summary["seconds"] = time.monotonic() - started
        assert counts["environment_actions"] <= host["action_cap"]
        save(OUT / "summary.json", summary)

    def evaluate(arm, rid, parameters, cid, representation, search):
        for case in panel["evaluation"]:
            for seed in panel["evaluation_seeds"]:
                spec = mechanism.spec(
                    "evaluation",
                    case,
                    arm,
                    seed,
                    f"evaluation-{rid or 'shared'}-{arm}",
                    rid,
                    cid,
                    representation,
                    search,
                )
                record([pool.episode(spec, parameters)])

    try:
        pool = SerialPool(deadline)
        actor = RecurrentActionAdapter(
            AdapterSpec(
                396,
                2,
                initialization_seed=repeats[0]["initialization_seed"],
            )
        )
        neutral = actor.parameters_vector()
        np.savez(
            OUT / "initialization/engineering.npz",
            neutral=neutral,
            scales=actor.coordinate_scales(),
        )
        for case in panel["training"]:
            pair = []
            for arm, parameters in [("native", None), ("neutral", neutral)]:
                spec = mechanism.spec(
                    "engineering", case, arm, panel["training_seed"], "parity-" + arm, "engineering"
                )
                pair.append(spec["id"])
                record([pool.episode(spec, parameters)])
            qualified.parity(*pair)
        first = panel["training"][0]
        for limit in [1, 7]:
            spec = mechanism.spec(
                "engineering", first, "native", panel["training_seed"], f"prefix-{limit}"
            )
            record([pool.episode(spec, limit=limit)])
            prior.OUT = OUT
            prior.prefix_check(
                f"parity-native_{first['id']}_{panel['training_seed']}", spec["id"], limit
            )
        mapping = SearchMap(neutral, actor.coordinate_scales(), "head")
        policy = CoordinatePolicy(mapping, repeats[0]["ppo_seed"])
        torch.save(policy.state_dict(), OUT / "initialization/engineering-policy.pt")
        for i in range(2):
            spec = mechanism.spec(
                "engineering",
                first,
                "ppo",
                panel["training_seed"],
                f"stochastic-{i}",
                "engineering",
                representation="head",
            )
            spec["sampling_seed"] = 12106
            record([pool.episode(spec, policy=policy, sampling_seed=12106)])
        for field in ["actions", "inputs", "states", "raw_logits", "log_probability", "values"]:
            with (
                np.load(
                    OUT
                    / "episodes"
                    / f"stochastic-0_{first['id']}_{panel['training_seed']}"
                    / "trace.npz"
                ) as a,
                np.load(
                    OUT
                    / "episodes"
                    / f"stochastic-1_{first['id']}_{panel['training_seed']}"
                    / "trace.npz"
                ) as b,
            ):
                np.testing.assert_array_equal(a[field], b[field])
        assert counts["environment_actions"] <= 10000
        save(
            OUT / "engineering.json",
            dict(
                passed=True, actions=counts["environment_actions"], episodes=len(summary["results"])
            ),
        )
        summary["phase"] = "waiting-for-representation"
        save(OUT / "summary.json", summary)
        selection = OUT.parent / "representation-selection.json"
        while not selection.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("Waiting deadline")
            time.sleep(5)
        choice = json.loads(selection.read_text())
        winner = choice["winner"]
        assert winner in ["full", "head", "subspace"] and choice["episodes"] == 324
        (OUT / "representation-selection.json").write_bytes(selection.read_bytes())
        summary["selected_representation"] = winner
        summary["representation_selection_sha256"] = digest(selection)
        native_done = False
        for repeat in repeats:
            rid = repeat["id"]
            actor = RecurrentActionAdapter(
                AdapterSpec(396, 2, initialization_seed=repeat["initialization_seed"])
            )
            neutral, scales = actor.parameters_vector(), actor.coordinate_scales()
            projection = projection_matrix(repeat["projection_seed"])
            np.savez(OUT / "initialization" / (rid + ".npz"), neutral=neutral, scales=scales)
            np.save(OUT / "initialization" / (rid + "-projection.npy"), projection)
            mapping = SearchMap(neutral, scales, winner, projection)
            z = np.random.default_rng(repeat["untrained_seed"]).normal(
                0, config["sigma"], mapping.dimension
            )
            untrained = mapping.decode(z)
            np.savez(
                OUT / "initialization" / (rid + "-untrained.npz"), genotype=z, parameters=untrained
            )
            nominees = {}
            for arm in repeat["order"]:
                key = f"{rid}-{winner}-{arm}"
                summary["phase"] = "training-" + key
                method = search_ppo if arm == "ppo" else mechanism.search
                args = (
                    [pool, repeat, mapping, config, panel, summary, record]
                    if arm == "ppo"
                    else [pool, repeat, arm, mapping, config, panel, summary, record]
                )
                nominees[key] = method(*args)
            save(OUT / f"selection-{rid}.json", dict(locked_at=time.time(), nominees=nominees))
            summary["phase"] = "evaluation-" + rid
            if not native_done:
                evaluate("native", None, None, None, None, None)
                native_done = True
            evaluate("untrained", rid, untrained, None, winner, None)
            for arm in ["main", "random", "ppo"]:
                key = f"{rid}-{winner}-{arm}"
                cid = nominees[key]["id"]
                with np.load(OUT / "candidates" / (cid + ".npz")) as checkpoint:
                    parameters = checkpoint["parameters"].copy()
                evaluate(arm, rid, parameters, cid, winner, key)
        assert sum(s["actions"] for s in summary["searches"].values()) == len(repeats) * 120000
        summary["status"] = "complete"
    except BaseException:
        summary["status"] = "failed"
        summary["failure"] = traceback.format_exc()
        raise
    finally:
        if pool is not None:
            pool.close()
        summary["resources"] = qualified.resource_summary()
        summary["seconds"] = time.monotonic() - started
        save(OUT / "summary.json", summary)


if __name__ == "__main__":
    run()
