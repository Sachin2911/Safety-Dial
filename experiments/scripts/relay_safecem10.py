"""Relay queued task-10 episodes from the primary rental to a helper rental's workers.

Added 25 October 2026 when the user asked for a second GPU to finish stage C2 sooner. The
primary's frozen orchestrator and workers are untouched: the relay claims tickets from the
back of the primary's pending queue (the local workers take the front) and marks them as
relayed, hands them to the helper's frozen workers (same freeze, same sources, same data
files; a re-run of a finished episode on the helper reproduced its saved states bit for
bit), copies each finished attempt directory back and moves it into place with an atomic
rename, and only then moves the done ticket into the queue the orchestrator waits on. A
helper failure is never forwarded: the ticket goes back to the primary and the relay stops
using the helper. A relayed ticket the helper does not hold on two consecutive polls goes
back to the primary. On start it adopts or returns any relayed tickets left by a previous
run, so it can be stopped at any moment. Runs on the workstation; no key reaches a rental.

Each poll tops the helper up first (so its workers never wait on a harvest), then harvests.
Near the end of a generation it claims at most half of the primary's remaining tickets, so
both machines run out of work at about the same time.
"""

import argparse
import datetime as dt
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import atomic_json, ssh_endpoint, utc
from helpers.safecem10_infra import instances, ssh_options

REMOTE = "/workspace/Safety-Dial/outputs/safecem10"
CONTROL = ROOT / "outputs/safecem10/control"
STAGING = ROOT / "outputs/safecem10/relay-staging"

CLAIM = r"""
import json, os, sys, datetime
q = sys.argv[1]; prefix = sys.argv[2]; count = int(sys.argv[3]); tag = sys.argv[4]
names = sorted(n for n in os.listdir(q + "/pending") if n.startswith(prefix) and n.endswith(".json"))
count = min(count, len(names) // 2)
out = []
for name in reversed(names):
    if len(out) >= count:
        break
    try:
        os.rename(q + "/pending/" + name, q + "/claimed/" + name)
    except FileNotFoundError:
        continue
    path = q + "/claimed/" + name
    data = json.load(open(path))
    data["claimed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    data["worker"] = tag
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)
    out.append(data)
print(json.dumps(out))
"""

PUT = r"""
import json, os, sys
q = sys.argv[1]
for data in json.load(sys.stdin):
    path = q + "/pending/" + data["job"]["id"] + ".json"
    with open(path + ".tmp", "w") as f:
        json.dump(data, f, indent=1)
    os.replace(path + ".tmp", path)
print("ok")
"""

# Lifecycle order: workers write the next location before removing the previous one, so a
# ticket moving forward between two checks is still found.
STATUS = r"""
import json, os, sys
q = sys.argv[1]; ids = json.load(sys.stdin); res = {}; tickets = {}
for i in ids:
    for s in ("pending", "claimed", "done", "failed"):
        if os.path.exists(f"{q}/{s}/{i}.json"):
            res[i] = s
            if s == "done":
                with open(f"{q}/done/{i}.json") as f:
                    tickets[i] = json.load(f)
            break
print(json.dumps({"status": res, "tickets": tickets}))
"""

FINALIZE = r"""
import json, os, shutil, sys
root = sys.argv[1]; stage = root + "/relay-staging"; blob = json.load(sys.stdin)
plan = blob["plan"]; tickets = blob["tickets"]
for job_id, attempt in plan:
    target = f"{root}/attempts/{attempt}"
    if os.path.exists(target):
        shutil.rmtree(f"{stage}/{attempt}", ignore_errors=True)
    else:
        os.rename(f"{stage}/{attempt}", target)
for job_id, attempt in plan:
    done = f"{root}/queue/done/{job_id}.json"
    with open(done + ".tmp", "w") as f:
        json.dump(tickets[job_id], f, indent=1)
    os.replace(done + ".tmp", done)
    try:
        os.remove(f"{root}/queue/claimed/{job_id}.json")
    except FileNotFoundError:
        pass
print("ok")
"""


class Host:
    def __init__(self, rental_path, name):
        self.rental = json.loads(Path(rental_path).read_text())
        self.name = name
        self.refresh()

    def refresh(self):
        provider = next(r for r in instances() if r["id"] == self.rental["instance_id"])
        self.host, self.port = ssh_endpoint(provider)
        # The scratchpad path is too long for a UNIX socket; ~/.ssh is the usual home.
        socket = Path.home() / ".ssh" / f"cm-relay10-{self.name}-%C"
        self.opts = [
            *ssh_options(self.port),
            "-o",
            "ControlMaster=auto",
            "-o",
            f"ControlPath={socket}",
            "-o",
            "ControlPersist=600",
        ]

    def run(self, command, data=None, timeout=90):
        result = subprocess.run(
            ["ssh", *self.opts, f"root@{self.host}", command],
            input=data,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode:
            raise RuntimeError(
                f"ssh {self.name} failed ({result.returncode}): {result.stderr[-300:]}"
            )
        return result.stdout

    def python(self, script, args, data=None, timeout=90):
        quoted = " ".join(shlex.quote(a) for a in args)
        command = f"/venv/main/bin/python -c {shlex.quote(script)} {quoted}"
        return self.run(command, data=data, timeout=timeout)

    def rsync(self, args, timeout=300):
        transport = shlex.join(["ssh", *self.opts])
        result = subprocess.run(
            ["rsync", "-a", "--timeout=120", "-e", transport, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode:
            raise RuntimeError(
                f"rsync {self.name} failed ({result.returncode}): {result.stderr[-300:]}"
            )


def log(event, **fields):
    CONTROL.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"utc": utc(), "event": event, **fields})
    with (CONTROL / "relay.log").open("a") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def give_back(primary, job_id):
    """Return a relayed ticket to the primary's pending queue (no-op if it already left)."""
    primary.run(
        f"cd {REMOTE}/queue && "
        f"{{ [ ! -e claimed/{job_id}.json ] || mv claimed/{job_id}.json pending/{job_id}.json; }}"
    )


def helper_status(helper, ids):
    """Queue state of each relayed id on the helper, plus the done tickets themselves."""
    if not ids:
        return {}, {}
    blob = json.loads(helper.python(STATUS, [f"{REMOTE}/queue"], data=json.dumps(ids)))
    return blob["status"], blob["tickets"]


def reconcile(primary, helper, tag):
    """Adopt relayed tickets the helper knows about; return orphans to the primary queue."""
    out = primary.run(
        f'cd {REMOTE}/queue/claimed && grep -l \'"worker": "{tag}"\' *.json 2>/dev/null || true'
    )
    relayed = [n[:-5] for n in out.split()]
    status, _ = helper_status(helper, relayed)
    adopted, returned = [], []
    for job_id in relayed:
        if job_id in status:
            adopted.append(job_id)
        else:
            give_back(primary, job_id)
            returned.append(job_id)
    log("reconciled", adopted=adopted, returned=returned)
    return adopted


def harvest(primary, helper, inflight, misses, status, tickets):
    """Install finished helper episodes on the primary; return (finished, failed, missing)."""
    missing = []
    for job_id in inflight:
        if job_id in status:
            misses.pop(job_id, None)
            continue
        misses[job_id] = misses.get(job_id, 0) + 1
        if misses[job_id] >= 2:
            give_back(primary, job_id)
            misses.pop(job_id)
            missing.append(job_id)
            log("missing_returned", job=job_id)
    finished = [i for i in inflight if status.get(i) == "done"]
    failed = [i for i in inflight if status.get(i) == "failed"]
    for job_id in failed:
        give_back(primary, job_id)
        helper.run(f"cd {REMOTE}/queue && mv failed/{job_id}.json failed/helper-{job_id}.json")
        log("helper_failed_returned", job=job_id)
    if not finished:
        return [], failed, missing
    plan = []
    for job_id in finished:
        ticket = tickets[job_id]
        if ticket.get("status") != "complete":
            raise RuntimeError(f"Helper done ticket for {job_id} is not complete")
        plan.append((job_id, Path(ticket["attempt_dir"]).name))
    STAGING.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=CONTROL) as tmp:
        dirs_list = Path(tmp) / "dirs.txt"
        dirs_list.write_text("\n".join(a for _, a in plan) + "\n")
        helper.rsync(
            [
                "-r",
                f"--files-from={dirs_list}",
                f"root@{helper.host}:{REMOTE}/attempts/",
                f"{STAGING}/",
            ]
        )
        for job_id, attempt in plan:
            if not (STAGING / attempt / "result.json").exists():
                raise RuntimeError(f"Helper finished {job_id} without a result")
        primary.rsync(
            [
                "-r",
                f"--files-from={dirs_list}",
                f"{STAGING}/",
                f"root@{primary.host}:{REMOTE}/relay-staging/",
            ]
        )
        blob = {"plan": plan, "tickets": {i: tickets[i] for i in finished}}
        primary.python(FINALIZE, [REMOTE], data=json.dumps(blob))
    for job_id, attempt in plan:
        shutil.rmtree(STAGING / attempt, ignore_errors=True)
        log("harvested", job=job_id)
    return finished, failed, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary", type=Path, default=ROOT / "progress/rental-safecem10-20261025.json"
    )
    parser.add_argument(
        "--helper", type=Path, default=ROOT / "progress/rental-safecem10b-20261025.json"
    )
    parser.add_argument("--prefix", default="C2-")
    parser.add_argument("--capacity", type=int, default=9, help="tickets on the helper at once")
    parser.add_argument("--poll", type=float, default=2)
    parser.add_argument("--until", required=True, help="UTC time to stop claiming, e.g. 20:40")
    args = parser.parse_args()
    now = dt.datetime.now(dt.timezone.utc)
    hh, mm = map(int, args.until.split(":"))
    stop_at = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    primary, helper = Host(args.primary, "primary"), Host(args.helper, "helper")
    tag = f"relay-to-{helper.rental['instance_id']}"
    helper.run(f"mkdir -p {REMOTE}/queue/pending {REMOTE}/queue/claimed")
    primary.run(f"mkdir -p {REMOTE}/relay-staging")
    state_path = CONTROL / "relay-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    state.setdefault("done", 0)
    state.setdefault("helper_failures", 0)
    inflight = reconcile(primary, helper, tag)
    state["inflight"] = inflight
    atomic_json(state_path, state)
    log("start", capacity=args.capacity, until=stop_at.isoformat(), inflight=len(inflight))
    errors = 0
    misses = {}
    ended = False
    ended_checked = 0.0
    while True:
        try:
            status, tickets = helper_status(helper, inflight)
            if time.monotonic() - ended_checked > 30:
                marker = primary.run(f"test -f {REMOTE}/run-C2-ended.txt && echo ended || true")
                ended = marker.strip() == "ended"
                ended_checked = time.monotonic()
            stopping = (
                state["helper_failures"] > 0 or dt.datetime.now(dt.timezone.utc) >= stop_at or ended
            )
            load = sum(1 for i in inflight if status.get(i) in ("pending", "claimed"))
            if not stopping and load < args.capacity:
                claimed = json.loads(
                    primary.python(
                        CLAIM,
                        [f"{REMOTE}/queue", args.prefix, str(args.capacity - load), tag],
                    )
                )
                if claimed:
                    ids = [t["job"]["id"] for t in claimed]
                    inflight += ids
                    state["inflight"] = inflight
                    atomic_json(state_path, state)
                    helper.python(PUT, [f"{REMOTE}/queue"], data=json.dumps(claimed))
                    status.update({i: "pending" for i in ids})
                    log("relayed", jobs=ids)
            finished, failed, missing = harvest(primary, helper, inflight, misses, status, tickets)
            for job_id in finished + failed + missing:
                inflight.remove(job_id)
            state["done"] += len(finished)
            state["helper_failures"] += len(failed)
            if stopping:
                for job_id in [i for i in inflight if status.get(i) == "pending"]:
                    moved = helper.run(
                        f"cd {REMOTE}/queue && mv pending/{job_id}.json returned-{job_id}.json "
                        "2>/dev/null && echo moved || echo busy"
                    ).strip()
                    if moved != "moved":
                        continue
                    give_back(primary, job_id)
                    inflight.remove(job_id)
                    log("returned", job=job_id)
                if not inflight:
                    state["inflight"] = inflight
                    atomic_json(state_path, state)
                    log("stopped", done=state["done"], helper_failures=state["helper_failures"])
                    break
            state["inflight"] = inflight
            atomic_json(state_path, state)
            errors = 0
        except (
            RuntimeError,
            OSError,
            subprocess.TimeoutExpired,
            StopIteration,
            ValueError,
            KeyError,
        ) as error:
            errors += 1
            log("error", kind=type(error).__name__, detail=str(error)[:200], errors=errors)
            if errors >= 30:
                log("giving_up", inflight=inflight)
                raise SystemExit(1)
            try:
                primary.refresh()
                helper.refresh()
            except Exception:
                pass
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
