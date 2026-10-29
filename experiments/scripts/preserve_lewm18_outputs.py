"""Hash the completed remote cohorts, sync, and verify every preserved file locally."""

import importlib.util
import json
import shlex
import subprocess

from package_lewm18_evidence import ROOT, read, sha, write


def main():
    spec = importlib.util.spec_from_file_location(
        "infra18", ROOT / "experiments/scripts/manage_lewm18.py"
    )
    infra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infra)
    rental, row = infra.current()
    assert row and row["id"] == rental["instance_id"] == 57851717
    command = r"""
import hashlib,json,pathlib,shutil,datetime
roots={'57851717':pathlib.Path('/workspace/Safety-Dial/outputs/lewm18'),
       '57851717-pusht-r1':pathlib.Path('/workspace/Safety-Dial-lewm18r1/outputs/lewm18')}
assert (roots['57851717']/'campaign-complete-cube.json').exists()
assert (roots['57851717-pusht-r1']/'campaign-complete-pusht.json').exists()
result={}
for name,root in roots.items():
    if name=='57851717':
        shutil.copy2(root/'network-cost.json',root/'environment/final-network-cost.json')
    files={}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            assert path==root/'STOP';continue
        if not path.is_file() or path in (root/'network-cost.json',root/'preservation-files.json'):
            continue
        digest=hashlib.sha256()
        with path.open('rb') as source:
            for chunk in iter(lambda:source.read(8*1024*1024),b''):digest.update(chunk)
        files[str(path.relative_to(root))]={'bytes':path.stat().st_size,'sha256':digest.hexdigest()}
    manifest=root/'preservation-files.json'
    manifest.write_text(json.dumps(files,indent=2)+'\n')
    result[name]={'files':len(files),'bytes':sum(x['bytes'] for x in files.values()),
                  'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cohorts':result}))
"""
    process = infra.remote("/venv/main/bin/python -c " + shlex.quote(command), timeout=90)
    if process.returncode:
        raise RuntimeError("Remote final snapshot failed; rental retained")
    remote = json.loads(process.stdout)
    host, port = infra.endpoint(row)
    locations = {
        "57851717": ("/workspace/Safety-Dial/outputs/lewm18/", ROOT / "outputs/lewm18/remote"),
        "57851717-pusht-r1": (
            "/workspace/Safety-Dial-lewm18r1/outputs/lewm18/",
            ROOT / "outputs/lewm18-pusht-r1/remote",
        ),
    }
    for cohort, (source, destination) in locations.items():
        subprocess.run(
            [
                "rsync",
                "-az",
                "--partial",
                "-e",
                shlex.join(["ssh", *infra.options(port)]),
                f"root@{host}:{source}",
                str(destination) + "/",
            ],
            check=True,
        )
        manifest = destination / "preservation-files.json"
        assert sha(manifest) == remote["cohorts"][cohort]["manifest_sha256"]
        files = read(manifest)
        for relative, expected in files.items():
            path = destination / relative
            assert path.stat().st_size == expected["bytes"] and sha(path) == expected["sha256"], (
                relative
            )
        remote["cohorts"][cohort]["local_directory"] = str(destination.relative_to(ROOT))
        remote["cohorts"][cohort]["all_files_verified"] = True
    remote["note"] = "All static completed-cohort files verified against remote SHA-256 hashes. "
    remote["note"] += "The changing network counter is excluded; its final snapshot is included."
    write(ROOT / "progress/receipts/18-final-local-preservation.json", remote)
    print(json.dumps(remote))


if __name__ == "__main__":
    main()
