"""Create or update the private Vast template without installing research packages."""

import argparse
import ast
import json
from pathlib import Path
import shlex
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="Update the recorded template")
    parser.add_argument("--dry-run", action="store_true", help="Print settings without API calls")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vast/evo-safety-filter.json"),
        help="Template settings, relative to the repository root unless absolute",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config_path = root / args.config
    record_path = root / "configs/vast/created-template.json"
    config = json.loads(config_path.read_text())
    onstart = config["onstart_cmd"]
    subprocess.run(["bash", "-n"], input=onstart, text=True, check=True)
    if config["private"] is not True:
        raise SystemExit("This tool only manages private templates.")
    readme = (root / config.get("readme_file", "configs/vast/README.md")).read_text()

    record = json.loads(record_path.read_text()) if args.update else {}
    if args.dry_run:
        print(json.dumps({"update": args.update, "record": record, "config": config}, indent=2))
        return

    query = (
        f"private=true id={record['id']}"
        if args.update
        else "private=true name=" + config["name"].replace(" ", "_")
    )
    existing = subprocess.run(
        ["vastai", "search", "templates", query], capture_output=True, text=True, check=True
    )
    rows = json.loads(existing.stdout)
    if isinstance(rows, dict):
        rows = rows.get("templates", [])
    if args.update:
        if len(rows) != 1 or rows[0]["name"] != config["name"]:
            raise SystemExit("Could not identify the recorded private template.")
        command = ["vastai", "update", "template", rows[0]["hash_id"]]
    else:
        if rows:
            raise SystemExit("A private template with this name exists. Use --update to change it.")
        command = ["vastai", "create", "template"]

    env = " ".join(f"-e {shlex.quote(k + '=' + v)}" for k, v in config["env"].items())
    command += [
        "--name",
        config["name"],
        "--image",
        config["image"],
        "--image_tag",
        config["image_tag"],
        "--desc",
        config["description"],
        "--env=" + env,
        "--jupyter-dir",
        config["jupyter_dir"],
        "--disk_space",
        str(config["disk_space_gb"]),
        "--search_params",
        config["search_params"],
        "--onstart-cmd",
        onstart,
        "--readme",
        readme,
    ]
    for key, flag in (
        ("ssh", "--ssh"),
        ("jupyter", "--jupyter"),
        ("direct", "--direct"),
        ("jupyter_lab", "--jupyter-lab"),
    ):
        if config[key]:
            command.append(flag)
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    prefix = "updated template: " if args.update else "New Template: "
    response = result.stdout.strip()
    if not response.startswith(prefix):
        raise SystemExit("Vast did not confirm the template change: " + response)
    payload = response.removeprefix(prefix)
    saved = json.loads(payload) if args.update else ast.literal_eval(payload)
    record = {k: saved.get(k) for k in ("id", "hash_id", "name")}
    if not all(record.values()):
        raise SystemExit(
            "Vast returned incomplete template metadata; read it back before retrying."
        )
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    print("Startup runs the base entrypoint only. No research installation or instance rental.")


if __name__ == "__main__":
    main()
