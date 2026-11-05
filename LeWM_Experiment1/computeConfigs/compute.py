"""Manage one LeWM rental and its jobs. Local Python uses only the standard library."""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import time
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from compute_common import (
    ApiError,
    ComputeError,
    JsonApi,
    STORAGE_REPOSITORIES,
    atomic_json,
    error_summary,
    fingerprint,
    load_secrets,
    locked,
    nonnegative,
    positive,
    read_json,
    redact,
    reject_secrets,
    require_secret,
    valid_id,
    validate_storage,
)


HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
PROJECT = HERE.parents[1]
WORKER_FILES = ("compute_common.py", "run_io.py", "hf_store.py", "remote_worker.py")
STATE_DIR = HERE / ".state"
SOURCE_SUFFIXES = {".py", ".sh", ".json", ".yaml", ".yml", ".toml", ".md", ".txt", ".csv"}
EXCLUDED_PARTS = {".git", ".state", "data", "cache", "checkpoints", "outputs", "runs", "wandb"}


def emit(value):
    print(json.dumps(value, indent=2, allow_nan=False))


class SSHUnavailable(ComputeError):
    pass


def driver_number(value):
    if isinstance(value, str) and "." in value:
        parts = [int(x) for x in value.split(".")]
        return parts[0] * 1000000 + parts[1] * 1000 + (parts[2] if len(parts) > 2 else 0)
    return int(value)


def offer_summary(row, template, policy, args, now=None):
    """Recheck search constraints locally; a cheap offer must not silently weaken them."""
    now = time.time() if now is None else now
    try:
        fields = dict(row)
        fields["verified"] = row.get("verified", row.get("verification") == "verified")
        fields["reliability2"] = row.get("reliability2", row.get("reliability"))
        fields["driver_version"] = driver_number(row.get("driver_vers", row.get("driver_version")))
        for name, operators in template["extra_filters"].items():
            actual = fields.get(name)
            if actual is None:
                return None
            if "eq" in operators and actual != operators["eq"]:
                return None
            if "gte" in operators and float(actual) < operators["gte"]:
                return None
        upload_speed = nonnegative(row["inet_up"])
        if upload_speed < policy["minimum_upload_mbps"]:
            return None
        hourly = positive(row["dph_total"])
        if hourly > args.max_hourly:
            return None
        if row.get("end_date") and float(row["end_date"]) < now + args.hours * 3600:
            return None
        incoming = nonnegative(row["inet_down_cost"])
        outgoing = nonnegative(row["inet_up_cost"])
        # Include the image pull and one complete read-back of uploaded artifacts.
        transfer = (policy["image_download_gb"] + args.download_gb + args.upload_gb) * incoming
        transfer += args.upload_gb * outgoing
        estimate = args.hours * hourly + transfer
        budget = getattr(args, "budget_usd", None)
        if budget is not None and estimate > budget:
            return None
        return {
            "offer_id": int(row["id"]),
            "machine_id": int(row["machine_id"]),
            "gpu": row["gpu_name"],
            "hourly_usd": hourly,
            "estimated_session_usd": round(estimate, 6),
            "estimated_transfer_usd": round(transfer, 6),
            "download_usd_per_gb": incoming,
            "upload_usd_per_gb": outgoing,
            "download_mbps": float(row["inet_down"]),
            "upload_mbps": upload_speed,
            "reliability": float(fields["reliability2"]),
            "disk_gb": template["recommended_disk_space"],
        }
    except (KeyError, TypeError, ValueError):
        return None


def endpoints(instance):
    result = []
    ports = instance.get("ports") or {}
    direct = ports.get("22/tcp", [])
    if instance.get("public_ipaddr") and direct:
        result.append({"host": instance["public_ipaddr"], "port": int(direct[0]["HostPort"])})
    if instance.get("ssh_host") and instance.get("ssh_port"):
        candidate = {"host": instance["ssh_host"], "port": int(instance["ssh_port"])}
        if candidate not in result:
            result.append(candidate)
    for endpoint in result:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", endpoint["host"]):
            raise ComputeError("Unexpected SSH host returned by provider")
        if not 0 < endpoint["port"] < 65536:
            raise ComputeError("Unexpected SSH port returned by provider")
    return result


def bundle_source(source, secrets, limit):
    source = Path(source).resolve()
    if not source.is_dir() or not source.is_relative_to(PROJECT):
        raise ComputeError("--source must be a directory inside this repository")
    relative = source.relative_to(PROJECT).as_posix()
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", relative],
        cwd=PROJECT,
        stdout=subprocess.PIPE,
        check=True,
    )
    output, total, included = io.BytesIO(), 0, []
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for item in sorted(set(result.stdout.decode().split("\0")) - {""}):
            path = PROJECT / item
            local = path.relative_to(source)
            if (
                set(local.parts) & EXCLUDED_PARTS
                or any(p.startswith(".") for p in local.parts)
                or path.suffix not in SOURCE_SUFFIXES
            ):
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(source):
                raise ComputeError("Source includes a symlink or path outside --source")
            if not path.is_file():
                continue
            data = path.read_bytes()
            reject_secrets(data, secrets.values())
            total += len(data)
            if total > limit:
                raise ComputeError(
                    "Source exceeds source_limit_bytes; choose a smaller code directory"
                )
            info = tarfile.TarInfo(local.as_posix())
            info.size, info.mtime, info.mode = len(data), 0, path.stat().st_mode & 0o755
            archive.addfile(info, io.BytesIO(data))
            included.append(local.as_posix())
    if not included:
        raise ComputeError("No deployable source files found")
    return output.getvalue(), included


BOOTSTRAP = r"""
import base64, hashlib, json, os, pathlib, tempfile
request=json.load(__import__('sys').stdin)
root=pathlib.Path('/workspace/LeWM_Experiment1')
control=root/'.control'
control.mkdir(parents=True, exist_ok=True)
session=control/'session.json'
if (session.exists()
    and json.loads(session.read_text())['session_id']!=request['state']['session_id']):
    raise RuntimeError('Remote session already belongs to another rental')
tools=control/'tools'/request['tools_hash']
tools.mkdir(parents=True, exist_ok=True)
for name, encoded in request['files'].items():
    if name not in ('compute_common.py','run_io.py','hf_store.py','remote_worker.py'):
        raise RuntimeError('Unexpected worker file')
    data=base64.b64decode(encoded, validate=True)
    path=tools/name
    if path.exists() and path.read_bytes()!=data:
        raise RuntimeError('Existing worker bundle differs')
    path.write_bytes(data)
fd,tmp=tempfile.mkstemp(dir=control)
with os.fdopen(fd,'w') as stream: json.dump(request['state'],stream)
os.replace(tmp,session)
print(json.dumps({'deployed':True}))
"""


class Controller:
    def __init__(self, *, directory=HERE, state_dir=STATE_DIR, secrets=None, api=None):
        self.directory, self.state_dir = Path(directory), Path(state_dir)
        self.policy = read_json(self.directory / "config/compute-policy.json")
        self.template = read_json(self.directory / "config/vast-template.json")
        self.storage = self.select_storage("baseline")
        if self.policy["remote_root"] != "/workspace/LeWM_Experiment1":
            raise ComputeError(
                "Experiment 1 workspace and private storage settings are inconsistent"
            )
        self.secrets = load_secrets(PROJECT) if secrets is None else secrets
        self.api = api or JsonApi(
            "https://console.vast.ai/api/v0", require_secret(self.secrets, "VAST_API_KEY")
        )
        self.state_path = self.state_dir / "instance.json"

    def select_storage(self, profile):
        if profile not in STORAGE_REPOSITORIES:
            raise ComputeError("Unknown Experiment 1 storage profile")
        filename = "storage.json" if profile == "baseline" else f"storage-{profile}.json"
        storage = validate_storage(read_json(self.directory / "config" / filename))
        if storage["repo_id"] != STORAGE_REPOSITORIES[profile]:
            raise ComputeError("Storage profile points to a different dataset")
        return storage

    def state(self):
        value = read_json(self.state_path)
        if value is None:
            raise ComputeError("No managed rental. Use search and up first.")
        # Retired baseline sessions predate the storage field. New sessions carry
        # their destination so editing a config cannot redirect an active run.
        self.storage = validate_storage(value.get("storage", self.select_storage("baseline")))
        return value

    def save(self, state):
        reject_secrets(json.dumps(state).encode(), self.secrets.values())
        atomic_json(self.state_path, state)

    def instances(self):
        return self.api.call("/instances/")["instances"]

    def instance(self, state):
        row = next(
            (item for item in self.instances() if item["id"] == state.get("instance_id")), None
        )
        if row is not None and row.get("label") != state["label"]:
            raise ComputeError("Instance label differs from saved session; refusing to operate")
        return row

    def search(self, args):
        query = dict(self.template["extra_filters"])
        query.update(
            type="on-demand",
            limit=self.policy["search_limit"],
            allocated_storage=self.template["recommended_disk_space"],
            order=[["dph_total", "asc"]],
            inet_up={"gte": self.policy["minimum_upload_mbps"]},
        )
        rows = self.api.call("/bundles/", method="POST", payload=query, retry=True)["offers"]
        # Vast's bundle-query `id` is not the returned rentable ask ID. Match the
        # requested ask locally, then apply every normal hardware and cost check.
        if getattr(args, "offer_id", None):
            rows = [row for row in rows if row.get("id") == args.offer_id]
        candidates = [offer_summary(row, self.template, self.policy, args) for row in rows]
        candidates = [item for item in candidates if item]
        return sorted(
            candidates,
            key=lambda item: (
                item["estimated_session_usd"],
                -item["reliability"],
                -item["upload_mbps"],
            ),
        )

    def registry_token(self):
        if not self.secrets.get("GHCR_READ_TOKEN"):
            try:
                result = subprocess.run(
                    ["gh", "auth", "token", "--hostname", "github.com"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=15,
                    check=True,
                )
            except (OSError, subprocess.SubprocessError):
                raise ComputeError(
                    "Set GHCR_READ_TOKEN in the ignored root .env, or authenticate GitHub CLI "
                    "with package pull access"
                ) from None
            self.secrets["GHCR_READ_TOKEN"] = result.stdout.strip()
        return require_secret(self.secrets, "GHCR_READ_TOKEN")

    def credentials_preflight(self):
        query = urlencode(
            {"select_filters": json.dumps({"hash_id": {"eq": self.policy["template_hash"]}})}
        )
        rows = self.api.call("/template/?" + query).get("templates", [])
        if len(rows) != 1 or any(
            rows[0].get(key) != self.template.get(key)
            for key in ("image", "tag", "env", "onstart", "runtype", "ssh_direct", "private")
        ):
            raise ComputeError(
                "Saved template differs from local settings; update its registration"
            )
        if self.template.get("docker_login_pass"):
            raise ComputeError("Registry passwords must stay out of tracked configuration")
        key = Path(self.policy["ssh_key"]).expanduser()
        if not key.is_file() or not Path(str(key) + ".pub").is_file():
            raise ComputeError("Configured SSH key and public key are required")
        public_key = Path(str(key) + ".pub").read_text().split()[:2]
        keys = self.api.call("/ssh/")
        rows = keys if isinstance(keys, list) else keys.get("ssh_keys", [])
        if not any(
            str(item.get("public_key", item.get("ssh_key", item.get("key", "")))).split()[:2]
            == public_key
            for item in rows
        ):
            raise ComputeError("Register the configured SSH public key in Vast before renting")
        token = self.registry_token()
        authorization = base64.b64encode(f"Sachin2911:{token}".encode()).decode()
        request = Request(
            "https://ghcr.io/token?service=ghcr.io&scope=repository:sachin2911/lewm-ap:pull",
            headers={"Authorization": "Basic " + authorization},
        )
        try:
            with urlopen(request, timeout=30) as response:
                registry_token = json.load(response)["token"]
            digest = self.template["tag"].split("@", 1)[1]
            request = Request(
                f"https://ghcr.io/v2/sachin2911/lewm-ap/manifests/{digest}",
                headers={
                    "Authorization": "Bearer " + registry_token,
                    "Accept": "application/vnd.docker.distribution.manifest.v2+json",
                },
            )
            with urlopen(request, timeout=30) as response:
                if "sha256:" + hashlib.sha256(response.read()).hexdigest() != digest:
                    raise ComputeError("Registry manifest differs from the pinned image")
        except (OSError, ValueError, KeyError):
            raise ComputeError("GHCR pull preflight failed; check GHCR_READ_TOKEN") from None
        hub = JsonApi("https://huggingface.co", require_secret(self.secrets, "HF_TOKEN"))
        info = hub.call(f"/api/datasets/{self.storage['repo_id']}")
        if info.get("private") is not True:
            raise ComputeError("Configured HF destination must exist and be private")

    def ssh_arguments(self, state, endpoint):
        known = self.state_dir / f"known-hosts-{state['session_id']}"
        return [
            "ssh",
            "-i",
            str(Path(self.policy["ssh_key"]).expanduser()),
            "-p",
            str(endpoint["port"]),
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={known}",
            f"{self.policy['ssh_user']}@{endpoint['host']}",
        ]

    def ssh_call(self, state, endpoint, command, payload=None, timeout=45):
        result = subprocess.run(
            [*self.ssh_arguments(state, endpoint), shlex.join(command)],
            input=None if payload is None else json.dumps(payload).encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        if result.returncode:
            detail = result.stdout.decode(errors="replace").strip()
            if not detail:
                detail = result.stderr.decode(errors="replace").strip()
            failure = SSHUnavailable if result.returncode == 255 else ComputeError
            raise failure("SSH operation failed: " + redact(detail[-1000:], self.secrets.values()))
        return result.stdout.decode()

    def connected(self, state):
        row = self.instance(state)
        if row is None:
            raise ComputeError("Managed instance is absent from Vast")
        choices = endpoints(row)
        if not choices:
            raise ComputeError("Vast has not assigned SSH connection details yet")
        for endpoint in choices:
            try:
                self.ssh_call(state, endpoint, ["/bin/true"], timeout=20)
                return endpoint
            except (ComputeError, subprocess.TimeoutExpired):
                continue
        raise ComputeError("Neither direct nor proxy SSH is reachable; instance remains rented")

    def worker(self, state, action, *, timeout=60, provider_instance=None, **payload):
        row = self.instance(state) if provider_instance is None else provider_instance
        if row is None:
            raise ComputeError("Managed instance is absent from Vast")
        command = [self.policy["python"], state["remote_tools"] + "/remote_worker.py"]
        request = dict(
            payload, action=action, root=self.policy["remote_root"], session_id=state["session_id"]
        )
        for endpoint in endpoints(row):
            try:
                output = self.ssh_call(state, endpoint, command, request, timeout=timeout)
                break
            except (SSHUnavailable, subprocess.TimeoutExpired):
                continue
        else:
            raise ComputeError("SSH transport failed. Use status before resubmitting a job.")
        try:
            result = json.loads(output)
        except ValueError:
            raise ComputeError(
                "Worker returned invalid JSON; inspect status before retrying"
            ) from None
        if result.get("error"):
            raise ComputeError(result["error"])
        return result

    def prepare(self, state):
        until = min(time.time() + self.policy["startup_timeout_seconds"], state["deadline_epoch"])
        last_error = "SSH not ready"
        while time.time() < until:
            try:
                endpoint = self.connected(state)
                break
            except ComputeError as error:
                last_error = str(error)
                time.sleep(5)
        else:
            state.update(phase="boot_failed", error=last_error)
            self.save(state)
            raise ComputeError("Boot wait expired. Rental remains active; use recover or destroy.")
        files = {
            name: base64.b64encode((SCRIPTS / name).read_bytes()).decode() for name in WORKER_FILES
        }
        bundle_hash = fingerprint(files)
        state["remote_tools"] = self.policy["remote_root"] + "/.control/tools/" + bundle_hash
        self.ssh_call(
            state,
            endpoint,
            [self.policy["python"], "-c", BOOTSTRAP],
            {"state": state, "files": files, "tools_hash": bundle_hash},
        )
        self.ssh_call(state, endpoint, ["/usr/bin/test", "-x", "/usr/bin/timeout"])
        readiness = self.ssh_call(
            state,
            endpoint,
            [self.policy["python"], "/opt/lewm-runtime/verify_runtime.py", "--gpu", "--render"],
            timeout=180,
        )
        state["readiness"] = json.loads(readiness)
        state["ssh"] = endpoint
        state.update(phase="ready", error=None)
        self.save(state)
        return {
            "instance_id": state["instance_id"],
            "phase": "ready",
            "ssh": state["ssh"],
            "deadline_epoch": state["deadline_epoch"],
            "hourly_usd": state["offer"]["hourly_usd"],
        }

    def up(self, args):
        previous = read_json(self.state_path)
        if previous and previous.get("phase") not in ("destroyed", "create_rejected"):
            raise ComputeError("A session is already recorded; use status, recover, or destroy")
        profile = getattr(args, "storage_profile", "baseline")
        self.storage = self.select_storage(profile)
        if args.hours * 3600 <= self.policy["final_sync_reserve_seconds"] + 60:
            raise ComputeError("Session is too short for boot and final sync reserve")
        self.credentials_preflight()
        choices = self.search(args)
        if not choices:
            raise ComputeError("No offers satisfy all hardware, transfer, and spending limits")
        offer = choices[0]
        if previous:
            archive = self.state_dir / "sessions" / f"{valid_id(previous['session_id'])}.json"
            if archive.exists() and read_json(archive) != previous:
                raise ComputeError("Retired session archive differs; preserve it before renting")
            if not archive.exists():
                atomic_json(archive, previous)
        session_id = "lewm-exp1-" + uuid.uuid4().hex[:12]
        now = time.time()
        state = {
            "schema_version": 1,
            "session_id": session_id,
            "label": session_id,
            "instance_id": None,
            "phase": "create_pending",
            "created_epoch": now,
            "deadline_epoch": now + args.hours * 3600,
            "budget_usd": args.budget_usd,
            "hours": args.hours,
            "download_gb": args.download_gb,
            "upload_gb": args.upload_gb,
            "offer": offer,
            "template_hash": self.policy["template_hash"],
            "template_sha256": fingerprint(self.template),
            "image": self.template["image"] + ":" + self.template["tag"],
            "storage_profile": profile,
            "storage": self.storage,
            "runs": [],
        }
        self.save(state)  # Persist before requesting a billable resource.
        token = require_secret(self.secrets, "GHCR_READ_TOKEN")
        payload = {
            "client_id": "me",
            "template_hash_id": state["template_hash"],
            "image": state["image"],
            "runtype": "ssh_direct",
            "disk": offer["disk_gb"],
            "label": state["label"],
            "cancel_unavail": True,
            "image_login": f"-u Sachin2911 -p {token} ghcr.io",
        }
        try:
            response = self.api.call(f"/asks/{offer['offer_id']}/", method="PUT", payload=payload)
        except ApiError as error:
            if error.status in (400, 401, 403, 404, 410, 422):
                state["phase"] = "create_rejected"
                state["create_http_status"] = error.status
                self.save(state)
                raise ComputeError(
                    f"Vast rejected the launch (HTTP {error.status}). "
                    "Refresh search before trying up again."
                )
            raise ComputeError("Creation was not confirmed. Use recover; do not retry the rental.")
        if not response.get("success") or not response.get("new_contract"):
            raise ComputeError("No instance ID was confirmed. Use recover before any new rental.")
        state.update(instance_id=int(response["new_contract"]), phase="allocated")
        self.save(state)
        print(
            f"Rented instance {state['instance_id']}; waiting for SSH.", file=sys.stderr, flush=True
        )
        try:
            return self.prepare(state)
        except Exception:
            state.update(phase="boot_failed")
            self.save(state)
            raise

    def recover(self):
        state = self.state()
        if state.get("phase") == "ready":
            return self.status()
        rows = [row for row in self.instances() if row.get("label") == state["label"]]
        if len(rows) != 1:
            raise ComputeError(
                f"Found {len(rows)} rentals for this session; no new rental requested"
            )
        if state.get("instance_id") not in (None, rows[0]["id"]):
            raise ComputeError("Session label maps to a different instance ID")
        state.update(instance_id=rows[0]["id"], phase="allocated")
        self.save(state)
        return self.prepare(state)

    def status(self):
        state = self.state()
        row = self.instance(state)
        report = {
            "instance_id": state.get("instance_id"),
            "phase": state["phase"],
            "storage_repo": self.storage["repo_id"],
            "provider_status": None if row is None else row.get("actual_status"),
            "deadline_epoch": state["deadline_epoch"],
            "budget_usd": state["budget_usd"],
            "planned_session_usd": state["offer"]["estimated_session_usd"],
            "runs": [],
            "ready_to_destroy": False,
        }
        if row:
            elapsed = max(0, time.time() - (row.get("start_date") or state["created_epoch"]))
            report.update(
                elapsed_minutes=round(elapsed / 60, 1),
                accrued_compute_estimate_usd=round(
                    elapsed / 3600 * (row.get("dph_total") or state["offer"]["hourly_usd"]), 4
                ),
                deadline_passed=time.time() >= state["deadline_epoch"],
            )
            if state.get("remote_tools"):
                try:
                    report.update(self.worker(state, "status", provider_instance=row))
                except ComputeError as error:
                    report["connection_error"] = str(error)
        elif state["phase"] != "destroyed":
            report["warning"] = (
                "Instance absent; local state retained. Pending create needs recover."
            )
        return report

    def run(self, args):
        state = self.state()
        if state["phase"] != "ready":
            raise ComputeError("Instance must pass readiness checks before a run")
        remaining = (
            state["deadline_epoch"] - time.time() - self.policy["final_sync_reserve_seconds"]
        )
        if remaining <= 0 or args.max_seconds > remaining:
            raise ComputeError("Run duration exceeds the remaining session time and sync reserve")
        command = list(args.command)
        if command[:1] == ["--"]:
            command.pop(0)
        if not command:
            raise ComputeError("Specify the experiment command after --")
        if command[0] in ("python", "python3"):
            command[0] = self.policy["python"]
        if any(re.match(r"--?(?:token|password|api[-_]key)(?:=|$)", item) for item in command):
            raise ComputeError("Credentials must not be passed in job arguments")
        protocol = Path(args.protocol).read_text()
        config = json.loads(Path(args.config).read_text())
        if not isinstance(config, dict) or not protocol.strip():
            raise ComputeError("Use a JSON object config and a nonempty prospective protocol")
        reject_secrets(json.dumps([command, protocol, config]).encode(), self.secrets.values())
        source, included = bundle_source(
            args.source, self.secrets, self.policy["source_limit_bytes"]
        )
        prior = self.worker(state, "status")["runs"]
        if any(run["supervisor_alive"] or run["job_alive"] for run in prior):
            raise ComputeError("A job is already active on this instance")
        if any(not run["ready_to_destroy"] for run in prior):
            raise ComputeError("Finalize prior runs with sync before starting another job")
        used = sum(run["upload"].get("bytes", 0) for run in prior)
        allowance = int(state["upload_gb"] * 1e9) - used
        if allowance < len(source) + len(protocol.encode()) + len(json.dumps(config)) + 65536:
            raise ComputeError("Insufficient remaining artifact transfer allowance")
        run_id = valid_id(
            args.run_id
            or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
        )
        git_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
        ).strip()
        row = self.instance(state)
        if row is None:
            raise ComputeError("Managed instance disappeared")
        spec = {
            "schema_version": 1,
            "run_id": run_id,
            "session_id": state["session_id"],
            "command": command,
            "seeds": args.seed,
            "created_epoch": time.time(),
            "job_deadline_epoch": time.time() + args.max_seconds,
            "sync_interval_seconds": self.policy["sync_interval_seconds"],
            "source_limit_bytes": self.policy["source_limit_bytes"],
            "max_artifact_bytes": allowance,
            "storage": self.storage,
            "git_revision": git_revision,
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "source_files": included,
            "config_sha256": hashlib.sha256(
                (json.dumps(config, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
            ).hexdigest(),
            "protocol_sha256": hashlib.sha256(protocol.encode()).hexdigest(),
            "image": state["image"],
            "instance_id": state["instance_id"],
            "machine_id": row.get("machine_id", state["offer"]["machine_id"]),
            "host": {
                key: row.get(key)
                for key in (
                    "gpu_name",
                    "num_gpus",
                    "driver_version",
                    "cuda_max_good",
                    "cpu_ram",
                    "gpu_ram",
                )
            },
            "readiness": state.get("readiness"),
        }
        if run_id in state["runs"]:
            raise ComputeError(
                "Run ID has already been submitted; use status, sync, or a new run ID"
            )
        state["runs"].append(run_id)
        self.save(state)  # Even an ambiguous SSH launch must prevent an unsafe destroy.
        print(f"Submitting {run_id}; use status if SSH disconnects.", file=sys.stderr, flush=True)
        return self.worker(
            state,
            "launch",
            spec=spec,
            experiment_config=config,
            protocol=protocol,
            source=base64.b64encode(source).decode(),
            token=require_secret(self.secrets, "HF_TOKEN"),
            timeout=90,
        )

    def sync(self, run_id=None):
        return self.worker(
            self.state(),
            "sync",
            run_id=run_id,
            token=require_secret(self.secrets, "HF_TOKEN"),
            timeout=3600,
        )

    def verify_receipt(self, run):
        receipt = run["upload"]
        if not run["ready_to_destroy"] or not receipt.get("manifest_sha256"):
            raise ComputeError("Run is not independently verified and idle")
        hub = JsonApi("https://huggingface.co", require_secret(self.secrets, "HF_TOKEN"))
        repo, revision = self.storage["repo_id"], receipt["revision"]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ComputeError("Invalid HF revision in upload receipt")
        info = hub.call(f"/api/datasets/{repo}/revision/{revision}")
        if info.get("private") is not True or info.get("sha") != revision:
            raise ComputeError("Pinned private HF revision cannot be verified")

        class SafeRedirect(HTTPRedirectHandler):
            def redirect_request(self, request, fp, code, msg, headers, newurl):
                if urlsplit(newurl).scheme != "https":
                    raise ComputeError("Refusing non-HTTPS artifact redirect")
                redirected = super().redirect_request(request, fp, code, msg, headers, newurl)
                if urlsplit(request.full_url).netloc != urlsplit(newurl).netloc:
                    redirected.remove_header("Authorization")
                return redirected

        prefix = self.storage["path_prefix"]
        url = (
            f"https://huggingface.co/datasets/{repo}/resolve/{revision}/"
            f"{prefix}/{run['run_id']}/manifest.json"
        )
        request = Request(url, headers={"Authorization": "Bearer " + hub.token})
        with build_opener(SafeRedirect()).open(request, timeout=60) as response:
            data = response.read(8 * 1024 * 1024 + 1)
        if hashlib.sha256(data).hexdigest() != receipt["manifest_sha256"]:
            raise ComputeError("Independent manifest download did not match the verified run")

    def destroy(self):
        state = self.state()
        row = self.instance(state)
        if row is None:
            if state.get("instance_id") is None:
                raise ComputeError("Creation remains unresolved; use recover before clearing state")
            state.update(phase="destroyed", destroyed_epoch=time.time())
            self.save(state)
            return {"instance_id": state["instance_id"], "removed": True}
        if state["runs"] or state.get("phase") == "ready":
            status = self.worker(state, "status")
            if any(run["supervisor_alive"] or run["job_alive"] for run in status["runs"]):
                raise ComputeError("A job is active. Use stop, then sync, before destroy.")
            if status["runs"]:
                self.sync()
                status = self.worker(state, "status")
            if not status["ready_to_destroy"]:
                raise ComputeError("Final backup verification is incomplete; rental retained")
            for run in status["runs"]:
                self.verify_receipt(run)
        response = self.api.call(f"/instances/{state['instance_id']}/", method="DELETE", payload={})
        if response.get("success") is not True:
            raise ComputeError("Provider did not confirm deletion; check status")
        for _ in range(15):
            if self.instance(state) is None:
                state.update(phase="destroyed", destroyed_epoch=time.time())
                self.save(state)
                return {"instance_id": state["instance_id"], "removed": True}
            time.sleep(2)
        raise ComputeError("Deletion requested but removal is not yet confirmed; check status")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="action", required=True)
    for name in ("search", "up"):
        command = commands.add_parser(name)
        command.add_argument("--hours", required=True, type=positive)
        command.add_argument("--max-hourly", required=True, type=positive)
        command.add_argument(
            "--download-gb",
            required=True,
            type=nonnegative,
            help="Expected input downloads, excluding image and final read-back",
        )
        command.add_argument(
            "--upload-gb",
            required=True,
            type=positive,
            help="Total output transfer allowance for the session",
        )
        command.add_argument("--budget-usd", required=name == "up", type=positive)
        command.add_argument("--offer-id", type=int)
        if name == "up":
            command.add_argument(
                "--storage-profile", choices=tuple(STORAGE_REPOSITORIES), default="baseline",
                help="Private output dataset, pinned to this session (default: baseline)",
            )
    for name in ("status", "recover", "destroy"):
        commands.add_parser(name)
    ssh = commands.add_parser("ssh", help="Interactive SSH or a quoted diagnostic command")
    ssh.add_argument("command", nargs=argparse.REMAINDER)
    for name in ("logs", "sync", "stop"):
        command = commands.add_parser(name)
        command.add_argument("--run-id")
        if name == "logs":
            command.add_argument("--lines", type=int, default=100)
    run = commands.add_parser("run")
    run.add_argument("--source", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--protocol", required=True)
    run.add_argument("--seed", required=True, type=int, action="append")
    run.add_argument("--max-seconds", required=True, type=positive)
    run.add_argument("--run-id")
    run.add_argument("command", nargs=argparse.REMAINDER)
    return result


def main():
    args = parser().parse_args()
    try:
        controller = Controller()
        if args.action == "search":
            emit(
                {
                    "offers": controller.search(args)[:3],
                    "cost_basis": "compute, allocated disk, estimated transfers and read-back",
                }
            )
        elif args.action == "status":
            emit(controller.status())
        elif args.action == "logs":
            output = controller.worker(
                controller.state(), "logs", run_id=args.run_id, lines=args.lines
            )
            print(redact(output["text"], controller.secrets.values()), end="")
        elif args.action == "ssh":
            state = controller.state()
            endpoint = controller.connected(state)
            command = args.command[1:] if args.command[:1] == ["--"] else args.command
            reject_secrets(json.dumps(command).encode(), controller.secrets.values())
            argv = controller.ssh_arguments(state, endpoint)
            if command:
                argv.append(shlex.join(command))
            subprocess.run(argv, check=True)
        else:
            with locked(controller.state_dir / "controller.lock"):
                if args.action == "up":
                    emit(controller.up(args))
                elif args.action == "recover":
                    emit(controller.recover())
                elif args.action == "run":
                    emit(controller.run(args))
                elif args.action == "sync":
                    emit(controller.sync(args.run_id))
                elif args.action == "stop":
                    emit(controller.worker(controller.state(), "stop", run_id=args.run_id))
                elif args.action == "destroy":
                    emit(controller.destroy())
    except KeyboardInterrupt:
        print(
            "Interrupted. No rental is destroyed automatically; use status or recover.",
            file=sys.stderr,
        )
        raise SystemExit(130) from None
    except Exception as error:
        print(error_summary(error), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
