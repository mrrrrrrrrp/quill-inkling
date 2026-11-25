"""Summarize one ab.ps1 run: residency, decode medians (warm runs), verify width, CPU-tier misses, peak VRAM."""
import re
import statistics
import sys

path, label = sys.argv[1], sys.argv[2]
text = open(path, encoding="utf-8", errors="replace").read()
res = re.search(r"(\d+) resident", text)
peak = re.search(r"peak device memory (\d+) MiB.*?exit (-?\d+)(.*)", text)
out = [f"{label:14s}", f"resident {res.group(1) if res else '?':>5s}"]
cpu = [float(x) for x in re.findall(r"\[cpu tier\][^,]*, ([\d.]+) experts/token", text)]
for name in ("chat", "code"):
    m = re.search(rf"^{name}: decode ([\d.]+) tok/s \(runs ([^)]*)\)(.*)$", text, re.M)
    if not m:
        out.append(f"{name} ?")
        continue
    runs = [float(x) for x in m.group(2).split(",")]
    warm = runs[1:] if len(runs) > 1 else runs
    tpv = re.search(r"([\d.]+) tokens/verify", m.group(3))
    out.append(f"{name} {statistics.median(warm):6.1f} (warm {', '.join(f'{x:.1f}' for x in warm)})"
               + (f" {float(tpv.group(1)):.2f} t/v" if tpv else ""))
if cpu:
    k = len(cpu) // 2  # per prompt: first run cold
    warm_cpu = cpu[1:k] + cpu[k + 1:] if k > 1 else cpu
    out.append(f"cpu-miss {statistics.mean(warm_cpu):5.1f} exp/tok")
if peak:
    out.append(f"peak {peak.group(1)} MiB" + (" KILLED" if "KILLED" in peak.group(3) or peak.group(2) != "0" else ""))
line = " | ".join(out)
print(line)
with open("ab_results.txt", "a", encoding="utf-8") as f:
    f.write(line + "\n")
