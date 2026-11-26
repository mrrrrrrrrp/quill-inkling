# quill trace analysis: replay a QUILL_TIER_TRACE run (teacher-forced) offline.
#
#   python trace_sim.py <trace.txt> summary      0a: round/visit/call structure, miss bytes, tier timeline
#   python trace_sim.py <trace.txt> residency   0b: observed miss bytes vs token-LFU (engine) / union-LFU
#                                                   / LRU / per-item static top-B / Belady MIN, equal budget
#   python trace_sim.py <trace.txt> truncation  item 1: oracle and static miss-aware row truncation
#   python trace_sim.py <trace.txt> l3          item 3 gate: loop-pass miss reuse + intervening tier bytes
#   python trace_sim.py <trace.txt> calib        item 16 gate: per-position conf -> acceptance
#
# Trace lines (QUILL_TIER_TRACE): "V visit layer n M <(expert<<1)|res of the M*8 (row,k) slots>" (48 per
# round, dumped after the round completes), "R m acc wall_us_since_last_R conf[0..6]" (round marker),
# "C visit t0_us t1_us layer M <unique experts>" (CPU tier call, worker 0, during the round).
# V's n = dispatch ENTRIES (one per non-resident (row, k) slot, duplicates included); the unique misses of
# a visit are the distinct experts among those slots (what the tier actually streams).
# Expert bytes NVFP4 = 4,478,976 (w13 2654208 + w13s 331776 + w2 1327104 + w2s 165888).
import array
import pickle
import sys
from collections import defaultdict

EXPERT_B = 4478976
EXP = 128
NVISITS = 48


def visit_layer_pass(v):
    if v < 6:
        return v, 0
    if v < 18:
        return v, 1
    if v < 30:
        return v - 12, 2
    if v < 42:
        return v - 24, 3
    return v - 24, 0


def parse(path):
    rounds = []
    cur = [None] * NVISITS
    ctimes = {}
    for line in open(path, encoding="utf-8"):
        c0 = line[0]
        if c0 == "V":
            f = line.split()
            v, layer, m = int(f[1]), int(f[2]), int(f[4])
            pairs = array.array("B", [int(x) & 0xFF for x in f[5:5 + 8 * m]])
            miss = frozenset({x >> 1 for x in pairs if not (x & 1)})
            res = frozenset({x >> 1 for x in pairs if x & 1})
            cur[v] = (layer, int(f[3]), m, pairs, miss, res)
        elif c0 == "C":
            f = line.split()
            ctimes[int(f[1])] = (int(f[2]), int(f[3]), len(f) - 6)
        elif c0 == "R":
            f = line.split()
            assert all(x is not None for x in cur), "round with missing V records"
            nvis = [(x[0], x[1], x[2], x[3], x[4], x[5]) for x in cur]
            rounds.append({
                "visits": nvis, "m": int(f[1]), "acc": int(f[2]), "wall_us": int(f[3]),
                "conf": [float(x) for x in f[4:11]], "ctimes": ctimes,
                "uniques": sum(len(x[4]) for x in nvis),
            })
            cur = [None] * NVISITS
            ctimes = {}
    return rounds


def item_bounds(rounds, thresh_us=300000):
    """item starts: an R wall > thresh (it contains the next prompt's prefill)."""
    b = [i for i, r in enumerate(rounds) if r["wall_us"] > thresh_us]
    return b + [len(rounds)]


def summary(rounds):
    n_r = len(rounds)
    tot_tok = sum(r["acc"] + 1 for r in rounds)
    wide = defaultdict(int)
    for r in rounds:
        wide[r["m"]] += 1
    print(f"rounds {n_r}  tokens {tot_tok}  tok/verify {tot_tok / n_r:.3f}  "
          f"widths " + " ".join(f"{k}:{100 * v / n_r:.1f}%" for k, v in sorted(wide.items())))
    bnd = item_bounds(rounds)
    print(f"item boundaries (wall > 300ms): {bnd[:-1]} -> {len(bnd) - 1} items, rounds per item "
          f"{[bnd[i + 1] - bnd[i] for i in range(len(bnd) - 1)]}")
    tot_uni = sum(r["uniques"] for r in rounds)
    tot_u_c = sum(c[2] for r in rounds for c in r["ctimes"].values())
    busy = sum(c[1] - c[0] for r in rounds for c in r["ctimes"].values())
    print(f"unique misses/round (V): {tot_uni / n_r:.2f}   (C lines say {tot_u_c / n_r:.2f})  "
          f"= {tot_uni * EXPERT_B / n_r / 1e6:.1f} MB/round")
    ntier = sum(r["ctimes"].values().__len__() for r in rounds)
    print(f"tier calls/round {ntier / n_r:.1f}  busy {busy / n_r / 1e3:.2f} ms/round  "
          f"{tot_uni * EXPERT_B / max(busy * 1e-6, 1e-9) / 1e9:.1f} GB/s while busy")
    wall = sorted(r["wall_us"] for r in rounds)
    print(f"round wall us: median {wall[n_r // 2]}  p10 {wall[n_r // 10]}  p90 {wall[9 * n_r // 10]}")
    per_w = {}
    for w in wide:
        ww = sorted(r["wall_us"] for r in rounds if r["m"] == w)
        per_w[w] = ww[len(ww) // 2]
    print("median wall by width:", dict(per_w))
    by_v = defaultdict(lambda: [0, 0, 0.0, 0.0, 0])
    for r in rounds:
        prev_end = None
        for v, vis in enumerate(r["visits"]):
            if v in r["ctimes"]:
                t0, t1, _ = r["ctimes"][v]
                a = by_v[v]
                a[0] += 1
                a[2] += t1 - t0
                if prev_end is not None:
                    a[3] += t0 - prev_end
                prev_end = t1
            if r["visits"][v][4]:
                a = by_v[v]
                a[4] += 1
                a[1] += len(r["visits"][v][4])
    print("visit  calls%  meanU  busy_us  gap_us (over calls)")
    for v in range(NVISITS):
        a = by_v[v]
        l, p = visit_layer_pass(v)
        print(f"{v:3d} L{l:2d}.{p}  {100 * a[0] / n_r:5.1f}  {a[1] / max(a[0], 1):5.1f}  "
              f"{a[2] / max(a[0], 1):7.1f}  {a[3] / max(a[0], 1):7.1f}")


# ------------------------------------------------------------------ 0b residency


def init_set(rounds, count, bnd):
    """the resident set all simulated policies start from: ids observed resident during item 0, padded with
    the most-selected other ids of item 0 to the engine's resident count."""
    obs = set()
    cnt = defaultdict(int)
    end = bnd[1] if len(bnd) > 1 else len(rounds)
    for r in rounds[:end]:
        for layer, n, m, pairs, miss, res in r["visits"]:
            for e in res:
                obs.add(layer * EXP + e)
            for x in pairs:
                cnt[layer * EXP + (x >> 1)] += 1
    padded = list(obs)
    for e in sorted(cnt, key=lambda e: cnt[e], reverse=True):
        if len(padded) >= count:
            break
        if e not in obs:
            padded.append(e)
    return set(padded[:count])


def sim_lfu(rounds, init, union=False, lru=False, margin=1.25, floor=2.0, swaps=32, step=8, decay=0.96,
            eval_from=0):
    resident = set(init)
    score = defaultdict(float)
    last = {}
    miss_b = 0
    for rn, r in enumerate(rounds):
        cnt = defaultdict(float)
        for layer, n, m, pairs, miss, res in r["visits"]:
            if union:
                s = set(layer * EXP + (x >> 1) for x in pairs)
                for e in s:
                    cnt[e] += 1.0
            else:
                for x in set(pairs):
                    cnt[layer * EXP + (x >> 1)] += 1.0
            for x in pairs:
                last[layer * EXP + (x >> 1)] = rn
        for e, c in cnt.items():
            score[e] += c
        if rn % step == step - 1:
            cands = sorted((e for e in score if e not in resident), key=lambda e: score[e], reverse=True)
            if lru:
                cands = sorted((e for e in last if e not in resident), key=lambda e: last[e], reverse=True)
                victims = sorted(resident, key=lambda e: last.get(e, -1))
            else:
                victims = sorted(resident, key=lambda e: score.get(e, 0.0))
            vi = 0
            done = 0
            for e in cands:
                if done >= swaps or vi >= len(victims):
                    break
                v = victims[vi]
                ok = lru or score[e] > score.get(v, 0.0) * margin + floor
                if ok:
                    resident.discard(v)
                    resident.add(e)
                    vi += 1
                    done += 1
                else:
                    break
            for e in list(score):
                score[e] *= decay
                if score[e] < 1e-3:
                    del score[e]
        if rn >= eval_from:
            for layer, n, m, pairs, miss, res in r["visits"]:
                for e in miss:
                    if layer * EXP + e not in resident:
                        miss_b += EXPERT_B
    return miss_b


def sim_static(rounds, budget, bnd):
    miss_b = 0
    for ai in range(len(bnd) - 1):
        seg = rounds[bnd[ai]:bnd[ai + 1]]
        if not seg:
            continue
        cnt = defaultdict(int)
        for r in seg:
            for layer, n, m, pairs, miss, res in r["visits"]:
                for x in set(pairs):
                    cnt[layer * EXP + (x >> 1)] += 1
        top = set(sorted(cnt, key=lambda e: cnt[e], reverse=True)[:len(budget)])
        for r in seg:
            for layer, n, m, pairs, miss, res in r["visits"]:
                for e in miss:
                    if layer * EXP + e not in top:
                        miss_b += EXPERT_B
    return miss_b


def sim_belady(rounds, budget, eval_from=0):
    import heapq
    events = []
    for rn, r in enumerate(rounds):
        for layer, n, m, pairs, miss, res in r["visits"]:
            events.append(set(layer * EXP + (x >> 1) for x in pairs))
    uses = defaultdict(list)
    for i, s in enumerate(events):
        for e in s:
            uses[e].append(i)
    cur = defaultdict(int)
    resident = set(budget)
    INF = 1 << 60
    miss_b = 0
    for i, s in enumerate(events):
        for e in s:
            while cur[e] < len(uses[e]) and uses[e][cur[e]] < i:
                cur[e] += 1
        rn = i // NVISITS
        for e in s:
            if e in resident:
                continue
            if rn >= eval_from:
                miss_b += EXPERT_B
            if len(resident) >= len(budget):
                best = (0, -1)
                for v in resident:
                    lst = uses[v]
                    nu = lst[cur[v]] if cur[v] < len(lst) else INF
                    if nu > best[0]:
                        best = (nu, v)
                resident.discard(best[1])
            resident.add(e)
    return miss_b


def residency(rounds, budget=1788):
    bnd = item_bounds(rounds)
    eval_from = bnd[1] if len(bnd) > 2 else 0  # skip item 0: let every policy reach steady state
    init = init_set(rounds, budget, bnd)
    obs_b = sum(EXPERT_B * e for r in rounds[eval_from:] for e in
                (x for layer, n, m, pairs, miss, res in r["visits"] for x in miss))
    n_eval = len(rounds) - eval_from
    print(f"budget {len(init)} experts {len(init) * EXPERT_B / 1e6:.0f} MB, "
          f"evaluating rounds [{eval_from}, {len(rounds)})")
    print("policy               miss experts/round   bytes/round   vs observed")
    rows = []
    for name, b in [("observed (engine)", obs_b), ("token-LFU", sim_lfu(rounds, init, eval_from=eval_from)),
                    ("union-LFU", sim_lfu(rounds, init, union=True, eval_from=eval_from)),
                    ("LRU", sim_lfu(rounds, init, lru=True, eval_from=eval_from)),
                    ("static top-B / item", sim_static(rounds, init, bnd) * n_eval / len(rounds)),
                    ("Belady MIN", sim_belady(rounds, init, eval_from=eval_from))]:
        rows.append((name, b / n_eval / EXPERT_B, b / n_eval / 1e6, 100 * b / obs_b))
    for name, u, mb, pct in rows:
        print(f"{name:20} {u:12.2f} {mb:12.1f} {pct:11.1f}%")


# ------------------------------------------------------------------ item 1 truncation


def truncation(rounds, exposed=0.64, miss_us=124.4, verbose_top=16):
    n_r = len(rounds)
    tot_tok = sum(r["acc"] + 1 for r in rounds)
    wall2 = sorted(r["wall_us"] for r in rounds if r["m"] == 2)
    w2tok = sum(r["acc"] + 1 for r in rounds if r["m"] == 2)
    w2n = max(len(wall2), 1)
    regen_us = wall2[len(wall2) // 2] / max(w2tok / w2n, 1)  # per regenerated token (a fresh width-2 round)
    wall_of = lambda r: min(r["wall_us"], 60000)
    tot_wall = sum(wall_of(r) for r in rounds)
    print(f"cost model: miss {miss_us:.0f} us, exposed {exposed:.0%}, regen token {regen_us:.0f} us "
          f"into wall {tot_wall / n_r / 1e3:.1f} ms/round")

    def fprof(vis):  # f[j]: misses only rows >= j need (j >= 1); f of the visit's misses
        layer, n, m, pairs, miss, res = vis
        if m <= 2:
            return None
        rows = [{pairs[8 * rr + k] >> 1 for k in range(8)} for rr in range(m)]
        f = [0] * m
        head = set()
        for j in range(1, m):
            head |= rows[j - 1]
            tail = set()
            for rr in range(j, m):
                tail |= rows[rr]
            f[j] = len((tail - head) & miss)
        return f

    # oracle: per round, DP over (visit, dead threshold), actual acc; + regen cost of the final threshold
    oracle_us = 0.0
    oracle_raw = 0.0
    for r in rounds:
        if r["m"] <= 2:
            continue
        acc = r["acc"]
        profs = [fprof(v) for v in r["visits"]]
        cur = {1: 0}
        for v in range(NVISITS):
            f = profs[v]
            if not f:
                continue
            best = {}
            for j, s in cur.items():
                add = f[j] * miss_us
                best[j] = max(best.get(j, -1), s + add)
                for j2 in range(j + 1, len(f)):
                    if f[j2] > f[j]:
                        best[j2] = max(best.get(j2, -1), s + f[j2] * miss_us)
            cur = best
        bnet = 0.0
        braw = 0.0
        for j, s in cur.items():
            raw = s * exposed
            lost = max(0, acc - (j - 1))
            cost = lost * regen_us if lost else 0.0
            if raw - cost > bnet:
                bnet, braw = raw - cost, raw
        oracle_us += max(0.0, bnet)
        oracle_raw += braw

    # static sweep: per visit, kill rows >= j once (f[j]-f[dead] >= k and cumconf < theta)
    results = []
    for k in (1, 2, 3, 4, 6, 8):
        for th in (0.30, 0.45, 0.60, 0.75, 0.90):
            sav = 0.0
            lost_tot = 0
            n_act = 0
            for r in rounds:
                if r["m"] <= 2:
                    continue
                conf = r["conf"]
                dead = 1
                r_sav = 0.0
                for v in range(NVISITS):
                    f = fprof(r["visits"][v])
                    if not f:
                        continue
                    for j in range(dead + 1, len(f)):
                        if f[j] - f[dead] >= k:
                            cumc = 1.0
                            for i in range(1, j):
                                cumc *= conf[i - 1]
                            if cumc < th:
                                dead = j
                    r_sav += f[dead] * miss_us
                if dead > 1:
                    lost = max(0, r["acc"] - (dead - 1))
                    sav += r_sav * exposed - (lost * regen_us if lost else 0.0)
                    lost_tot += lost
                    n_act += 1
            results.append((sav, k, th, n_act, lost_tot))
    results.sort(reverse=True)
    print(f"ORACLE truncation: {oracle_us / n_r:.0f} us net = {100 * oracle_us / tot_wall:.1f}% of wall "
          f"(gross {oracle_raw / n_r:.0f} us = {100 * oracle_raw / tot_wall:.1f}%)")
    print("static (k, theta):   net us/round     %wall  rounds  lost tok")
    for sav, k, th, n_act, lost in results[:verbose_top]:
        print(f"  k={k} th={th:.2f}: {sav / n_r:8.1f} ({100 * sav / tot_wall:5.2f}%)  {n_act:6d}  "
              f"{lost:5d} = {100 * lost / tot_tok:.2f}%")


# ------------------------------------------------------------------ item 3 L3 reuse


def l3(rounds, l3_bytes=96 * 1024 * 1024):
    again = 0
    tot = 0
    dist = []
    for r in rounds:
        misses = [v[4] for v in r["visits"]]
        for v in range(6, 42):
            _, p = visit_layer_pass(v)
            if p < 2:
                continue
            prev = v - 12
            for e in r["visits"][v][4] & set().union(*[misses[prev]]) if False else r["visits"][v][4]:
                tot += 1
                if e in misses[prev]:
                    again += 1
                    b = sum(len(x) for x in misses[prev:v + 1]) * EXPERT_B
                    dist.append(b)
    print(f"loop pass-2/3 misses {tot}; also a miss 12 visits earlier: {again} "
          f"({100 * again / max(tot, 1):.1f}%)")
    if dist:
        dist.sort()
        n = len(dist)
        print(f"tier bytes streamed in between: median {dist[n // 2] / 1e6:.0f}MB  "
              f"p10 {dist[n // 10] / 1e6:.0f}MB  p90 {dist[9 * n // 10] / 1e6:.0f}MB  "
              f"within 80MB: {100 * sum(1 for b in dist if b < 80e6) / n:.1f}%")


# ------------------------------------------------------------------ item 16 calibration


def calib(rounds):
    pos = defaultdict(lambda: [0, 0])
    for r in rounds:
        acc = r["acc"]
        for j in range(min(r["m"] - 1, 7)):
            if acc < j:
                continue
            b = min(int(r["conf"][j] * 10), 9)
            pos[(j + 1, b)][0] += 1
            if acc > j:
                pos[(j + 1, b)][1] += 1
    print("pos  conf     n   accept(equality)")
    for (p, b), (n, a) in sorted(pos.items()):
        print(f"{p:3d}  {b / 10:.1f}-{b / 10 + 0.1:.1f}  {n:6d}   {100 * a / max(n, 1):5.1f}")


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("usage: trace_sim.py <trace.txt> <summary|residency|truncation|l3|calib> [budget]", file=sys.stderr)
        return 1
    path, mode = sys.argv[1], sys.argv[2]
    budget = int(sys.argv[3]) if len(sys.argv) == 4 else 1788
    cache = path + ".pkl"
    try:
        rounds = pickle.load(open(cache, "rb"))
    except Exception:
        rounds = parse(path)
        pickle.dump(rounds, open(cache, "wb"))
    {"summary": summary, "residency": lambda rs: residency(rs, budget), "truncation": truncation,
     "l3": l3, "calib": calib}[mode](rounds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
