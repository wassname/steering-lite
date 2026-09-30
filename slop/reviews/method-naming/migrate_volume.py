"""Migrate the benchmark Volume through its file API, without starting a GPU. PI/OpenAI."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

from migrate_artifacts import rename

ROOT = Path(__file__).resolve().parents[3]
LOCAL = ROOT / "outputs/bsbench"
BACKUP = "method-name-backup/bsbench/"
volume = modal.Volume.from_name("steering-lite-bsbench-v3")


def remote_digest(path):
    sha = hashlib.sha256()
    for chunk in volume.read_file(path):
        sha.update(chunk)
    return sha.hexdigest()


def main():
    records = json.loads((ROOT / "slop/reviews/method-naming/local-migration.json").read_text())
    by_old = {"bsbench/" + r["old"]: r for r in records}
    entries = {e.path: e for e in volume.iterdir("bsbench") if e.type == modal.volume.FileEntryType.FILE}
    paths = [p for p in entries if rename(p) != p]
    assert paths, "No old method paths on Volume"
    for path in paths:
        assert path in by_old, f"Volume artifact missing locally: {path}"
        assert rename(path) not in entries, f"destination exists: {rename(path)}"
        assert entries[path].size == by_old[path]["bytes_before"], f"size differs: {path}"
    changed = [p for p in paths if by_old[p]["before"] != by_old[p]["after"]]
    print(f"Volume plan: {len(paths)} files, {len(changed)} metadata rewrites", flush=True)

    def check_original(path):
        assert remote_digest(path) == by_old[path]["before"], f"Volume differs from local source: {path}"
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(check_original, paths))
    print("All remote originals match the backed-up local sources", flush=True)

    # One Volume layer for the batch; per-file copy_files exhausted the V1 layer limit. PI/OpenAI
    with volume.batch_upload(force=True) as batch:
        for path in changed:
            batch.put_file(ROOT / ".local/method-naming/original-artifacts" / by_old[path]["old"], BACKUP + by_old[path]["old"])
        for path in paths:
            batch.put_file(LOCAL / by_old[path]["new"], rename(path))

    def verify_new(path):
        assert remote_digest(rename(path)) == by_old[path]["after"], f"new artifact differs: {path}"
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(verify_new, paths))
    print("All renamed remote files verified; removing old active paths", flush=True)
    answer_dirs = {str(Path(p).parent) for p in paths if "/answers/" in p}
    for directory in sorted(answer_dirs):
        assert all(p in paths for p in entries if p.startswith(directory + "/")), directory
        volume.remove_file(directory, recursive=True)
    for path in paths:
        if "/answers/" not in path:
            volume.remove_file(path)
    report = {"files": len(paths), "metadata_rewrites": len(changed), "all_before_and_after_hashes_verified": True, "original_metadata_backup": BACKUP}
    (ROOT / "slop/reviews/method-naming/volume-migration.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
