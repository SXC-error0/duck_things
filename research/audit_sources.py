from pathlib import Path
from collections import defaultdict
import hashlib, json, re
base = Path(__file__).resolve().parent
repo = base / "microduck-replica"
links = defaultdict(set)
for p in repo.rglob("*.md"):
    if ".git" in p.parts:
        continue
    for u in re.findall(r"https?://[^\s<>\"']+", p.read_text(encoding="utf-8", errors="replace").replace(chr(96), " ")):
        u = u.rstrip(").,;，。；")
        links[u].add(p.relative_to(repo).as_posix())
out = [{"url": u, "referenced_by": sorted(fs), "status": "extracted; not individually verified"} for u, fs in sorted(links.items())]
(base / "repository-link-index.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
old = base.parent / "imu_to_dxl(1)" / "firmware"
new = repo / "hardware" / "imu_to_dxl" / "firmware"
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
old_core = {p.relative_to(old / "Core").as_posix(): digest(p) for p in (old / "Core").rglob("*") if p.is_file()}
new_core = {p.relative_to(new / "Core").as_posix(): digest(p) for p in (new / "Core").rglob("*") if p.is_file()}
local_hex = old / "Build" / "imu_to_dxl.hex"
repo_hex = new / "Build" / "imu_to_dxl.hex"
result = {
    "date": "2026-10-16",
    "main_repository_commit": "2b734d5b472688607f5eb80314d3c603eef27094",
    "external_http_links_extracted": len(out),
    "local_core_file_count": len(old_core),
    "repository_core_file_count": len(new_core),
    "core_identical": old_core == new_core,
    "local_hex_sha256": digest(local_hex),
    "repository_hex_sha256": digest(repo_hex),
    "hex_identical": local_hex.read_bytes() == repo_hex.read_bytes(),
    "scope_note": "File comparison only; not a hardware flashing or electrical test."
}
(base / "local-comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))

