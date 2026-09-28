"""Probe: every getattr(<config-like>, "name", default) / config.name access in
src/ + run_variant.py must name a real PipelineConfig field; a misspelled name
silently returns the default (a production-ON flag could be ignored)."""
import pathlib, re
from src.config import PipelineConfig

fields = set(PipelineConfig.__dataclass_fields__)
pat_get = re.compile(r'getattr\(\s*(config|cfg|self\.config|self\.cfg|_cfg|sub_cfg|base_cfg)\s*,\s*["\']([A-Za-z_0-9]+)["\']')
pat_attr = re.compile(r'\b(config|cfg|self\.config|sub_cfg)\.([a-z_][a-z_0-9]*)\b')
methods = {n for n in dir(PipelineConfig) if not n.startswith("__")}
files = list(pathlib.Path("src").rglob("*.py")) + [pathlib.Path("run_variant.py")]
bad = []
for p in files:
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        for m in pat_get.finditer(line):
            if m.group(2) not in fields:
                bad.append((str(p), i, m.group(2), line.strip()[:110]))
        for m in pat_attr.finditer(line):
            name = m.group(2)
            if name not in fields and name not in methods:
                bad.append((str(p), i, name, line.strip()[:110]))
for b in bad:
    print(b)
print("n_unknown:", len(bad))
