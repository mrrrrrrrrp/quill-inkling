"""Top kernels of an nsys cuda_gpu_kern_sum CSV. usage: kern_summary.py CSV [N] [tokens]"""
import csv
import re
import sys

rows = list(csv.DictReader(open(sys.argv[1])))
n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
tokens = float(sys.argv[3]) if len(sys.argv) > 3 else 0
tot = sum(int(r["Total Time (ns)"]) for r in rows)
print(f"total GPU kernel time {tot / 1e9:.2f} s" + (f" = {tot / 1e6 / tokens:.2f} ms/token" if tokens else ""))
for r in rows[:n]:
    name = re.sub(r"ingine::kernels::|<unnamed>::", "", r["Name"])
    short = re.sub(r"\(.*", "", name)[:58]
    targs = re.search(r"<([^()]*)>", name)
    t = int(r["Total Time (ns)"])
    per = f" {t / 1e6 / tokens:6.3f} ms/tok" if tokens else ""
    print(f"{float(r['Time (%)']):5.1f}% {t / 1e6:8.1f} ms{per} n={int(r['Instances']):6d} avg {float(r['Avg (ns)']) / 1e3:7.1f} us"
          f"  {short}{'' if not targs else ' <' + targs.group(1)[:40] + '>'}")
