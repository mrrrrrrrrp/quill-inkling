"""GPU idle gaps between consecutive kernels in an nsys SQLite export (CUPTI_ACTIVITY_KIND_KERNEL).
usage: gaps.py profile.sqlite [tokens]
Reports total kernel time, total gap time, gap histogram, and the kernel pairs with the largest summed gaps.
"""
import collections
import sqlite3
import sys

db = sqlite3.connect(sys.argv[1])
tokens = float(sys.argv[2]) if len(sys.argv) > 2 else 0
names = dict(db.execute("SELECT id, value FROM StringIds"))
rows = db.execute("SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
busy = sum(e - s for s, e, _ in rows)
gaps = []
pairs = collections.Counter()
pair_n = collections.Counter()
last_end = None
last_name = None
for s, e, n in rows:
    if last_end is not None:
        g = s - last_end
        if 0 < g < 200_000:  # ignore idle periods > 200 us (host syncs, between items)
            gaps.append(g)
            key = (names.get(last_name, "?")[:40], names.get(n, "?")[:40])
            pairs[key] += g
            pair_n[key] += 1
    last_end = max(last_end or 0, e)
    last_name = n
tg = sum(gaps)
per = f" | per token: kernels {busy / 1e6 / tokens:.2f} ms, gaps {tg / 1e6 / tokens:.2f} ms" if tokens else ""
print(f"kernels {len(rows)}, busy {busy / 1e9:.2f} s, short gaps {tg / 1e9:.3f} s ({len(gaps)} gaps, mean {tg / max(len(gaps), 1) / 1e3:.2f} us){per}")
for lo, hi in ((0, 1000), (1000, 2000), (2000, 4000), (4000, 10000), (10000, 200000)):
    sel = [g for g in gaps if lo <= g < hi]
    print(f"  gaps {lo / 1e3:5.1f}-{hi / 1e3:6.1f} us: {len(sel):7d}  sum {sum(sel) / 1e6:8.1f} ms")
print("largest summed gaps (prev kernel -> next kernel):")
for (a, b), v in pairs.most_common(12):
    print(f"  {v / 1e6:8.1f} ms  n={pair_n[(a, b)]:6d}  avg {v / pair_n[(a, b)] / 1e3:5.2f} us  {a} -> {b}")
