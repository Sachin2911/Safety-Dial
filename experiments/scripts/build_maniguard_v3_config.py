"""Build an executable configuration from the frozen v3 protocol and actual rental."""

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def build(protocol_path, receipt_path, project, launch_stamp):
    protocol_path, receipt_path = Path(protocol_path), Path(receipt_path)
    protocol = json.loads(protocol_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    binding, training, limits = (protocol[k] for k in ("rental_binding", "training", "limits"))
    for name in ("instance_id", "label", "deadline_utc"):
        if receipt[name] != binding[name]:
            raise ValueError("Current rental differs from the prepared protocol binding")
    if receipt["offer"]["machine_id"] != binding["machine_id"]:
        raise ValueError("Current physical machine differs from the protocol")
    # Copy only the execution settings of the completed engineering predecessor.
    # No old rental limits, budgets, predecessor charges or launch instructions carry over.
    base = json.loads((ROOT / "configs/maniguard_evolution/development-v2.json").read_text())
    keys = ("maniguard_checkout", "environment", "information", "adapter", "evaluator")
    config = {key: base[key] for key in keys}
    config.update(schema="maniguard-evolution-development-v3",
                  purpose=protocol["purpose"],
                  authorization="Explicit current user goal authorizes implementation and execution",
                  protocol_binding={"path": protocol_path.relative_to(ROOT).as_posix(),
                                    "sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest()},
                  rental={"instance_id": binding["instance_id"], "label": binding["label"],
                          "machine_id": binding["machine_id"],
                          "guard_directory": "/workspace/setup-maniguard-v3-20261030"})
    config["search"] = {
        "method": "cmaes", "population": training["population"],
        "generations": training["generations"], "sigma": training["sigma"],
        "seed": training["optimizer_seed"], "initialization_seed": training["initialization_seed"],
        "tasks": training["tasks"], "policy_seeds": training["policy_seeds"],
        "objective": training["objective"], "mutation_profile": training["mutation_profile"],
        "native_anchor": True, "initial_controller": None,
        "selection": "Protocol native/elite/mean development selection; strict improvement or native",
    }
    config["selection"] = {key: protocol["selection"][key] for key in ("tasks", "policy_seeds")}
    config["validation"] = {key: protocol["validation"][key] for key in ("tasks", "policy_seeds")}
    config["limits"] = {
        "deadline_utc": receipt["deadline_utc"],
        "shutdown_reserve_seconds": limits["preservation_reserve_seconds"],
        "minimum_remaining_seconds_before_first_episode":
            limits["minimum_remaining_seconds_before_first_scored_episode"],
        "minimum_start_window_seconds": limits["minimum_episode_start_window_seconds"],
        "episode_timeout_seconds": limits["episode_timeout_seconds"],
        "max_starts": limits["max_episode_starts_including_failures"],
        "max_failed_starts": limits["maximum_additional_failed_engineering_starts"],
        "max_environment_step_attempts": limits["max_instrumented_environment_step_attempts"],
        "max_model_query_attempts": limits["max_total_model_query_attempts"],
        "max_server_preparation_model_query_attempts":
            limits["max_server_preparation_model_query_attempts"],
    }
    config["evaluator"]["name"] = "maniguard_evolution_development_v3"
    icds = ("/usr/share/vulkan/icd.d/nvidia_icd.json", "/etc/vulkan/icd.d/nvidia_icd.json")
    icd = next((path for path in icds if Path(path).is_file()), None)
    if icd is None:
        raise ValueError("NVIDIA Vulkan ICD is unavailable on this rented host")
    config["environment"]["VK_ICD_FILENAMES"] = icd
    config["server_preparation_accounting"] = str(
        Path(project) / "external-runtime/server-preparation.json")
    config["preservation"] = {
        "repo": protocol["preservation"]["private_dataset"],
        "base_prefix": "22_conservative_adapter_development/" + launch_stamp,
    }
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    value = build(args.protocol.resolve(), args.receipt, args.project, stamp)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise SystemExit("Refusing to overwrite a prepared runtime configuration")
    args.out.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"runtime_config": str(args.out), "launch_stamp": stamp,
                      "base_prefix": value["preservation"]["base_prefix"]}))
