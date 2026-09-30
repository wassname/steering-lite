"""One-off method-ID migration; tensor payloads and answer bytes stay unchanged. PI/OpenAI."""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

NAMES = {
    "svdkv_resid": "sink_split_resid", "svdkv": "sink_split",
    "vjp_delta": "vjp_resid", "vjp_cache": "vjp_value",
    "kv_cache_gram": "value_gram", "super_sspace": "sspace_pool",
    "sspace_damp_amp": "sspace_scale",
}
PATTERN = re.compile(r"(^|/)(" + "|".join(NAMES) + r")(?=$|[./-]|_s\d)")


def rename(text):
    return PATTERN.sub(lambda m: m[1] + NAMES[m[2]], text)


def metadata(value):
    if isinstance(value, str):
        return rename(value)
    if isinstance(value, list):
        return [metadata(v) for v in value]
    if isinstance(value, dict):
        return {rename(k): metadata(v) for k, v in value.items()}
    return value


def digest(path, offset=0):
    with path.open("rb") as f:
        f.seek(offset)
        return hashlib.file_digest(f, "sha256").hexdigest()


def migrate(root, backup, report):
    files = sorted(p for model in root.glob("*-g*") for p in model.rglob("*") if p.is_file())
    plan = [(p, root / rename(str(p.relative_to(root)))) for p in files]
    plan = [(p, q) for p, q in plan if p != q]
    for p, q in plan:
        assert not q.exists(), f"destination exists: {q}"
        assert not (backup / p.relative_to(root)).exists(), f"backup exists: {p}"
    records = []
    for p, q in plan:
        old = str(p.relative_to(root))
        saved = backup / old
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, saved)
        q.parent.mkdir(parents=True, exist_ok=True)
        before = digest(p)
        if p.suffix == ".json":
            original = json.loads(p.read_text())
            updated = metadata(original)
            q.write_text(json.dumps(updated, indent=2) + "\n")
            assert json.loads(q.read_text()) == updated
        elif p.suffix == ".safetensors":
            with p.open("rb") as src:
                size = int.from_bytes(src.read(8), "little")
                header = json.loads(src.read(size))
                cfg = json.loads(header["__metadata__"]["cfg"])
                header["__metadata__"]["cfg"] = json.dumps(metadata(cfg))
                encoded = json.dumps(header, separators=(",", ":")).encode()
                encoded += b" " * (-len(encoded) % 8)
                with q.open("wb") as dst:
                    dst.write(len(encoded).to_bytes(8, "little"))
                    dst.write(encoded)
                    shutil.copyfileobj(src, dst)
            assert digest(p, 8 + size) == digest(q, 8 + len(encoded)), f"tensor payload changed: {p}"
        else:
            shutil.copy2(p, q)
            assert digest(q) == before, f"answer/other bytes changed: {p}"
        records.append({"old": old, "new": str(q.relative_to(root)), "before": before, "after": digest(q), "bytes_before": p.stat().st_size})
        p.unlink()
    for directory in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        if not any(directory.iterdir()):
            directory.rmdir()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(records, indent=2) + "\n")
    print(f"Migrated {len(records)} files; verified tensor payloads and answer bytes; originals: {backup}; manifest: {report}", flush=True)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("backup", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    migrate(args.root, args.backup, args.report)
