"""Part C.1 of docs/plan_followup_2026-11-12.md: fingerprint of the frozen journal results.

git tracks the code and the small result tables, but NOT the stored model outputs (results/certificate/*.parquet)
or the logs (both git-ignored, too large). This script writes a SHA-256 checksum for every such file, so the
frozen state (git tag `journal-results-frozen` + this manifest) can be checked later, byte for byte.

  python scripts/freeze_manifest.py write    -> results/FROZEN_MANIFEST.sha256 (sorted; "sha256  bytes  path")
  python scripts/freeze_manifest.py check    -> compares the files on disk now with the manifest; exit 1 if any
                                                file is missing or different (new files are listed, not errors)
  python scripts/freeze_manifest.py check --root /some/clone   -> same, for another copy of the project
"""
import argparse, hashlib, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "results/FROZEN_MANIFEST.sha256"
PATTERNS = ("results/raw/*.parquet", "results/certificate/*.parquet", "results/certificate/*.csv", "results/certificate/*.txt",
            "logs/*.log", "configs/*.json", "splits/*.json")


def sha(p, buf=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def files(root):
    out = set()
    for pat in PATTERNS:
        out.update(p for p in root.glob(pat) if p.is_file())
    return sorted(out, key=lambda p: p.relative_to(root).as_posix())


def read_manifest(path):
    rows = {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        h, n, rel = line.split("  ", 2)
        rows[rel] = (h, int(n))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("write", "check"))
    ap.add_argument("--root", default=str(ROOT), help="project copy to fingerprint / check (default: this one)")
    ap.add_argument("--manifest", default=None, help="manifest to check against (default: <root>/" + MANIFEST + ")")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    if a.mode == "write":
        fs = files(root)
        lines = [f"# frozen journal results: {len(fs)} files; patterns: {', '.join(PATTERNS)}"]
        total = 0
        for i, p in enumerate(fs, 1):
            n = p.stat().st_size; total += n
            lines.append(f"{sha(p)}  {n}  {p.relative_to(root).as_posix()}")
            if i % 25 == 0:
                print(f"  hashed {i}/{len(fs)}", flush=True)
        (root / MANIFEST).write_text("\n".join(lines) + "\n")
        print(f"wrote {MANIFEST}: {len(fs)} files, {total / 1e9:.2f} GB")
        return
    ref = read_manifest(Path(a.manifest) if a.manifest else root / MANIFEST)
    bad, missing = [], []
    for rel, (h, n) in ref.items():
        p = root / rel
        if not p.is_file():
            missing.append(rel)
        elif p.stat().st_size != n or sha(p) != h:
            bad.append(rel)
    now = {p.relative_to(root).as_posix() for p in files(root)}
    new = sorted(now - set(ref))
    print(f"checked {len(ref)} frozen files under {root}: {len(ref) - len(bad) - len(missing)} identical, "
          f"{len(bad)} different, {len(missing)} missing; {len(new)} new files not in the manifest")
    for tag, lst in (("DIFFERENT", bad), ("MISSING", missing), ("new (not frozen)", new)):
        for rel in lst:
            print(f"  {tag}: {rel}")
    sys.exit(1 if bad or missing else 0)


if __name__ == "__main__":
    main()
