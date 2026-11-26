# quill trace analysis: replay a QUILL_TIER_TRACE run (teacher-forced) offline.
#
# This is trace_sim_res.py, item-0b rework. Differences vs trace_sim.py:
#   - sim_belady is rewritten for speed (per-id event-index lists + cur[] cursors + a lazy max-heap
#     of (-next_use, id) with stale-entry discarding). The old one is kept verbatim as
#     sim_belady_orig for A/B. sim_belady_naive is an independent literal Belady MIN used to verify
#     that sim_belady's semantics (evict the resident whose next use strictly after the current
#     event is furthest; ties -> smaller id; refill immediately; evict only when full; count from
#     eval_from) are exact.
#   - extra diagnostics in residency mode: per-(row,k)-selection LFU (engine's scoring model,
#     per SPEEDUP-IDEAS item 2), the same Belady pass counted with sim_lfu's miss-pair accounting,
#     per-round observed-vs-LFU calibration, and a "verify" mode that A/Bs the three Belady
#     implementations on a slice.
#   - main() loads <trace>.pkl read-only and never re-dumps it.
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
import bisect
import heapq
import os
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
            eval_from=0, track=None, touch=None):
    """track: if a list is passed, the per-round miss expert count (RNIM semantics) is appended for
    rounds >= eval_from; the miss bytes semantics is unchanged.
    touch: if a list is passed, the per-round touch-accounted miss bytes are appended: every unique
    expert of a visit that the policy itself does not hold resident (what it would actually stream),
    which is comparable to observed no matter how the policy's residency differs from the engine's."""
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
            n_miss = 0
            t_miss = 0
            for layer, n, m, pairs, miss, res in r["visits"]:
                base = layer * EXP
                if touch is not None:
                    for e in {x >> 1 for x in pairs}:
                        if base + e not in resident:
                            t_miss += EXPERT_B
                for e in miss:
                    if base + e not in resident:
                        miss_b += EXPERT_B
                        n_miss += 1
            if track is not None:
                track.append(n_miss)
            if touch is not None:
                touch.append(t_miss)
    return miss_b


def sim_lfu_mult(rounds, init, eval_from=0, track=None, touch=None, margin=1.25, floor=2.0, swaps=32,
                 step=8, decay=0.96):
    """Engine-model "token-LFU": counts per (row,k) selection (pairs entries WITH multiplicity),
    which is what the live policy is described to do (per-(row,k) route counts); everything else -
    decay/margin/floor/swap cadence/eval accounting - identical to sim_lfu's token path.
    NB sim_lfu's token path counts a visit's expert once (set(pairs)); on this trace that equals
    union counting (an expert never appears with both res-bit values in one visit), while the
    engine's per-(row,k) counts weight bursty experts up to 8x more."""
    resident = set(init)
    score = defaultdict(float)
    last = {}
    miss_b = 0
    for rn, r in enumerate(rounds):
        cnt = defaultdict(float)
        for layer, n, m, pairs, miss, res in r["visits"]:
            for x in pairs:
                cnt[layer * EXP + (x >> 1)] += 1.0
        for e, c in cnt.items():
            score[e] += c
        if rn % step == step - 1:
            cands = sorted((e for e in score if e not in resident), key=lambda e: score[e], reverse=True)
            victims = sorted(resident, key=lambda e: score.get(e, 0.0))
            vi = 0
            done = 0
            for e in cands:
                if done >= swaps or vi >= len(victims):
                    break
                v = victims[vi]
                if score[e] > score.get(v, 0.0) * margin + floor:
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
            n_miss = 0
            t_miss = 0
            for layer, n, m, pairs, miss, res in r["visits"]:
                base = layer * EXP
                if touch is not None:
                    for e in {x >> 1 for x in pairs}:
                        if base + e not in resident:
                            t_miss += EXPERT_B
                for e in miss:
                    if base + e not in resident:
                        miss_b += EXPERT_B
                        n_miss += 1
            if track is not None:
                track.append(n_miss)
            if touch is not None:
                touch.append(t_miss)
    return miss_b


def sim_static(rounds, budget, bnd, both=False, mult_count=False):
    """both: also return touch-accounted misses (unique experts per visit not in the item's top-B),
    i.e. the bytes this static policy would actually stream; the default return is unchanged.
    mult_count: rank the top-B by raw per-(row,k) selection counts instead of per-visit distinct
    expert counts (control for the repack decomposition; miss accounting is unchanged)."""
    miss_b = 0
    touch_b = 0
    for ai in range(len(bnd) - 1):
        seg = rounds[bnd[ai]:bnd[ai + 1]]
        if not seg:
            continue
        cnt = defaultdict(int)
        for r in seg:
            for layer, n, m, pairs, miss, res in r["visits"]:
                base = layer * EXP
                if mult_count:
                    for x in pairs:
                        cnt[base + (x >> 1)] += 1
                else:
                    for x in set(pairs):
                        cnt[base + (x >> 1)] += 1
        top = set(sorted(cnt, key=lambda e: cnt[e], reverse=True)[:len(budget)])
        for r in seg:
            for layer, n, m, pairs, miss, res in r["visits"]:
                base = layer * EXP
                if both:
                    for e in {x >> 1 for x in pairs}:
                        if base + e not in top:
                            touch_b += EXPERT_B
                for e in miss:
                    if base + e not in top:
                        miss_b += EXPERT_B
    return (miss_b, touch_b) if both else miss_b


def _belady_uses(rounds):
    """pass 1: per-expert-id sorted list of event indices ("uses"); an event is one visit's unique
    expert set (duplicates across rows collapse, resident-bit ignored). Event i = round nv + visit."""
    uses = defaultdict(list)
    i = 0
    for r in rounds:
        for layer, n, m, pairs, miss, res in r["visits"]:
            base = layer * EXP
            for e in {x >> 1 for x in pairs}:
                uses[base + e].append(i)
            i += 1
    return uses


def sim_repack(rounds, init, bnd, eval_from=0, last2=False, no_repack=False, future=False,
               track=None, touch_track=None, margin=1.25, floor=2.0, swaps=32, step=8, decay=0.96):
    """Implementable 'full repack when idle' (SPEEDUP-IDEAS item 2B). Between item starts: the
    engine-model flow of sim_lfu_mult verbatim (per-(row,k) counts, decay folded every 8th round,
    margin/floor-gated swaps, up to 32 per cadence). At every item start (item_bounds, except
    round 0 where no past exists): resident := exact top-B by current score - no margin/floor
    gate and no swap cap for that one rebuild (if the score table were smaller than B the old
    residents are kept in ascending id order; does not happen on this trace).
    last2 ('repack+0.5 floor'): the repack is ranked by scores rebuilt fresh over the last two
    items' rounds (per-(row,k) counts re-accumulated from an empty table, decay folded on the
    same global 8-round clock); the throttled swaps between item starts keep using the
    full-history engine table. future (ORACLE diagnostic): rank the repack by the upcoming
    item's raw per-(row,k) counts, i.e. the semi-oracle ranking with the repack mechanism - to
    separate "past scores do not predict the next item" from "the throttle is already converged".
    no_repack: disables the boundary rebuilds (A/B check: with no_repack the miss bytes must
    equal sim_lfu_mult's exactly). track / touch_track: per-round engine-accounted miss expert
    counts / touch-accounted miss expert counts for rounds >= eval_from.
    Returns (miss_b, touch_b): miss_b = sim_lfu accounting (pairs the engine itself flagged
    non-resident); touch_b = visit-unique experts this policy does not hold = what it would stream."""
    resident = set(init)
    B = len(resident)
    score = defaultdict(float)
    starts = {} if no_repack else {b: i for i, b in enumerate(bnd[:-1])}
    miss_b = 0
    touch_b = 0
    for rn, r in enumerate(rounds):
        si = starts.get(rn)
        if rn and si is not None:
            if future:
                w1 = bnd[si + 1]
                rank = defaultdict(float)
                for rr in range(rn, w1):
                    for layer, n, m, pairs, miss, res in rounds[rr]["visits"]:
                        base = layer * EXP
                        for x in pairs:
                            rank[base + (x >> 1)] += 1.0
            elif last2:
                w0 = bnd[si - 2] if si >= 2 else 0
                rank = defaultdict(float)
                for rr in range(w0, rn):
                    cc = defaultdict(float)
                    for layer, n, m, pairs, miss, res in rounds[rr]["visits"]:
                        base = layer * EXP
                        for x in pairs:
                            cc[base + (x >> 1)] += 1.0
                    for e, c in cc.items():
                        rank[e] += c
                    if rr % step == step - 1:
                        for e in list(rank):
                            rank[e] *= decay
                            if rank[e] < 1e-3:
                                del rank[e]
            else:
                rank = score
            old = resident
            resident = set(sorted(rank, key=lambda e: rank[e], reverse=True)[:B])
            for e in sorted(old - resident):
                if len(resident) >= B:
                    break
                resident.add(e)
        # ---- engine-model round flow, identical to sim_lfu_mult
        cnt = defaultdict(float)
        for layer, n, m, pairs, miss, res in r["visits"]:
            for x in pairs:
                cnt[layer * EXP + (x >> 1)] += 1.0
        for e, c in cnt.items():
            score[e] += c
        if rn % step == step - 1:
            cands = sorted((e for e in score if e not in resident), key=lambda e: score[e], reverse=True)
            victims = sorted(resident, key=lambda e: score.get(e, 0.0))
            vi = 0
            done = 0
            for e in cands:
                if done >= swaps or vi >= len(victims):
                    break
                v = victims[vi]
                if score[e] > score.get(v, 0.0) * margin + floor:
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
            n_miss = 0
            t_exp = 0
            for layer, n, m, pairs, miss, res in r["visits"]:
                base = layer * EXP
                for e in {x >> 1 for x in pairs}:
                    if base + e not in resident:
                        touch_b += EXPERT_B
                        t_exp += 1
                for e in miss:
                    if base + e not in resident:
                        miss_b += EXPERT_B
                        n_miss += 1
            if track is not None:
                track.append(n_miss)
            if touch_track is not None:
                touch_track.append(t_exp)
    return miss_b, touch_b


def sim_belady(rounds, budget, eval_from=0, diag=None):
    """Belady MIN (efficient): evict the resident expert whose next use - the first use strictly
    after the current event - is furthest away; ties -> smaller id; a missed expert becomes
    resident immediately once refilled; evict only when full (capacity len(budget)); miss bytes
    counted from round eval_from; touch set = the visit's unique experts.

    cur[id] = index into uses[id] of the first use strictly after the current event (so
    next_use = uses[id][cur[id]], or INF when exhausted); advanced for the ids of each event.
    Lazy max-heap of (-next_use, id): an entry (-nu, id) is valid iff id is resident and
    nxt[id] == nu; entries are pushed at init for every resident and after each event for every id
    resident right after it (the touched ones that stayed, and the misses, which are resident once
    refilled); invalid entries are discarded when they surface at eviction pops. The heap is
    periodically compacted to its valid entries, which changes no eviction decision.

    diag: if a dict is passed, it is filled with extra accounts (miss_b_pairs: same misses counted
    with sim_lfu's accounting, i.e. only pairs the engine itself reported as non-resident).
    """
    INF = 1 << 60
    uses = _belady_uses(rounds)
    resident = set(budget)
    B = len(budget)
    cur = defaultdict(int)
    nxt = {}
    heap = []
    push = heapq.heappush
    pop = heapq.heappop
    for v in resident:
        u = uses.get(v)
        nu = u[0] if u else INF
        nxt[v] = nu
        push(heap, (-nu, v))
    miss_b = 0
    miss_b_pairs = 0
    miss_b_atomic = 0
    i = 0
    rn = 0
    for r in rounds:
        in_eval = rn >= eval_from
        for layer, n, m, pairs, miss, res in r["visits"]:
            base = layer * EXP
            ids = [base + e for e in {x >> 1 for x in pairs}]
            # advance cursors; push fresh next-use entries for touched experts still resident
            for e in ids:
                u = uses[e]
                lu = len(u)
                c = cur[e]
                while c < lu and u[c] <= i:
                    c += 1
                cur[e] = c
                nu = u[c] if c < lu else INF
                nxt[e] = nu
                if e in resident:
                    push(heap, (-nu, e))
            # misses: count, evict the valid top, refill immediately
            if diag is not None and in_eval:
                # atomic-event view: misses if residency were atomic per visit (no same-event
                # eviction churn): an expert used while resident costs nothing even if a later
                # miss of the same visit evicts and re-streams it in the sequential model.
                miss_b_atomic += EXPERT_B * sum(1 for e in ids if e not in resident)
            for e in ids:
                if e in resident:
                    continue
                if in_eval:
                    miss_b += EXPERT_B
                    if (e - base) in miss:
                        miss_b_pairs += EXPERT_B
                if len(resident) > B - 1:
                    while heap:
                        nnu, v = pop(heap)
                        if v in resident and nxt[v] == -nnu:
                            resident.discard(v)
                            break
                    else:
                        raise RuntimeError("belady heap exhausted with residents left")
                resident.add(e)
                push(heap, (-nxt[e], e))
            i += 1
            if i % 30000 == 0:  # compact the heap to its valid entries (same valid set)
                heap = [(-nxt[v], v) for v in resident]
                heapq.heapify(heap)
        rn += 1
    if diag is not None:
        diag["miss_b_pairs"] = miss_b_pairs
        diag["miss_b_atomic"] = miss_b_atomic
    return miss_b


def sim_belady_naive(rounds, budget, eval_from=0):
    """Independent slow-but-literal Belady MIN, for verifying sim_belady on a slice: same touch
    sets and refill/counting semantics (i.e. the same per-event id order: ids built from the same
    per-visit set comprehension); on every miss, rescan all residents and pick the max (next use
    strictly after the current event), ties -> smaller id. next use via bisect."""
    INF = 1 << 60
    br = bisect.bisect_right
    uses = _belady_uses(rounds)
    resident = set(budget)
    B = len(budget)
    miss_b = 0
    i = 0
    rn = 0
    for r in rounds:
        in_eval = rn >= eval_from
        for layer, n, m, pairs, miss, res in r["visits"]:
            base = layer * EXP
            ids = [base + e for e in {x >> 1 for x in pairs}]
            for e in ids:
                if e in resident:
                    continue
                if in_eval:
                    miss_b += EXPERT_B
                if len(resident) >= B:
                    best_nu = -1
                    best_id = -1
                    for v in resident:
                        u = uses[v]
                        p = br(u, i)
                        nu = u[p] if p < len(u) else INF
                        if nu > best_nu or (nu == best_nu and v < best_id):
                            best_nu, best_id = nu, v
                    resident.discard(best_id)
                resident.add(e)
            i += 1
        rn += 1
    return miss_b


def sim_belady_orig(rounds, budget, eval_from=0):
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
    # NB trace_sim.py's obs_b weighs each miss by its local expert id (sum(EXPERT_B * e for e in
    # miss), e in 0..127), inflating the "observed" reference by the mean expert id (~63.6x) and so
    # deflating every "vs observed" percentage by that factor. Fixed here: one EXPERT_B per miss,
    # matching the header docs (unique misses/round * EXPERT_B). The original value is printed below.
    obs_b = sum(EXPERT_B for r in rounds[eval_from:]
                for x in (b for layer, n, m, pairs, miss, res in r["visits"] for b in miss))
    obs_b_orig_weighted = sum(EXPERT_B * e for r in rounds[eval_from:] for e in
                              (x for layer, n, m, pairs, miss, res in r["visits"] for x in miss))
    n_eval = len(rounds) - eval_from
    print(f"budget {len(init)} experts {len(init) * EXPERT_B / 1e6:.0f} MB, "
          f"evaluating rounds [{eval_from}, {len(rounds)})")
    print("policy               miss experts/round   bytes/round   vs observed")
    tok_track, mult_track = [], []
    tok_touch, lru_touch, mult_touch = [], [], []
    rp_track = []
    bdiag = {}
    rows = []

    def add(name, b):
        rows.append((name, b / n_eval / EXPERT_B, b / n_eval / 1e6, 100 * b / obs_b))

    add("observed (engine)", obs_b)
    add("token-LFU", sim_lfu(rounds, init, eval_from=eval_from, track=tok_track, touch=tok_touch))
    add("union-LFU", sim_lfu(rounds, init, union=True, eval_from=eval_from))
    add("LRU", sim_lfu(rounds, init, lru=True, eval_from=eval_from, touch=lru_touch))
    st_miss_b, st_touch_b = sim_static(rounds, init, bnd, both=True)
    add("static top-B / item", st_miss_b * n_eval / len(rounds))
    add("Belady MIN", sim_belady(rounds, init, eval_from=eval_from, diag=bdiag))
    add("per-(row,k)-LFU eng", sim_lfu_mult(rounds, init, eval_from=eval_from, track=mult_track,
                                            touch=mult_touch))
    add("Belady MIN eng-acc", bdiag["miss_b_pairs"])
    # implementable "full repack when idle" (item 2B): at each item start, resident := top-B by
    # the engine-model score table;'+0.5 floor' variant ranks that repack by scores rebuilt fresh
    # over the last 2 items. touch = what the policy would actually stream.
    rp_miss, rp_touch = sim_repack(rounds, init, bnd, eval_from=eval_from, track=rp_track)
    add("repack@item eng-acc", rp_miss)
    add("repack@item touch", rp_touch)
    rp2_miss, rp2_touch = sim_repack(rounds, init, bnd, eval_from=eval_from, last2=True)
    add("repack+0.5f eng-acc", rp2_miss)
    add("repack+0.5f touch", rp2_touch)
    rpo_miss, rpo_touch = sim_repack(rounds, init, bnd, eval_from=eval_from, future=True)
    for name, u, mb, pct in rows:
        print(f"{name:20} {u:12.2f} {mb:12.1f} {pct:11.1f}%")
    # ---- extra diagnostics of trace_sim_res (not part of the trace_sim.py table) ----
    obs_v = [sum(len(v[4]) for v in r["visits"]) for r in rounds[eval_from:]]

    def corr(x, y):
        n = len(x)
        mx, my = sum(x) / n, sum(y) / n
        sx = sum((a - mx) ** 2 for a in x) ** 0.5
        sy = sum((b - my) ** 2 for b in y) ** 0.5
        return 0.0 if not sx or not sy else sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)

    print(f"[diag] observed: {obs_b / n_eval / EXPERT_B:.2f} miss experts/round over the eval window"
          f" (whole-trace mean including item 0: "
          f"{sum(r['uniques'] for r in rounds) / len(rounds):.2f}; C-lines agree).")
    print(f"[diag] trace_sim.py's expert-id-weighted obs_b was {obs_b_orig_weighted / n_eval / EXPERT_B:.2f}"
          f" 'experts'/round - the bug this rework fixes.")
    s = sum(tok_touch) / n_eval / EXPERT_B
    print(f"[diag] touch-accounted (policy-consistent stream) miss experts/round: "
          f"token/union-LFU {s:.2f}  LRU {sum(lru_touch) / n_eval / EXPERT_B:.2f}  "
          f"per-(row,k)-LFU {sum(mult_touch) / n_eval / EXPERT_B:.2f}  "
          f"static {st_touch_b * n_eval / len(rounds) / n_eval / EXPERT_B:.2f}  "
          f"Belady {rows[5][1]:.2f}   (observed {obs_b / n_eval / EXPERT_B:.2f})")
    print(f"[diag] Belady sequential-model vs atomic-visit accounting: {rows[5][1]:.2f} vs "
          f"{bdiag['miss_b_atomic'] / n_eval / EXPERT_B:.2f} miss experts/round "
          f"(the gap is the same-event evict-and-re-stream churn; see sim_belady docstring).")
    print(f"[diag] repack variants (item-boundary top-B; '+0.5f' repack ranked by last-2-items "
          f"fresh scores): eng-acc all {rp_miss / n_eval / EXPERT_B:.2f} vs last2 "
          f"{rp2_miss / n_eval / EXPERT_B:.2f};  touch all {rp_touch / n_eval / EXPERT_B:.2f} vs last2 "
          f"{rp2_touch / n_eval / EXPERT_B:.2f}/round  (last2 vs all touch delta "
          f"{100 * (rp2_touch - rp_touch) / rp_touch:+.1f}%)  vs static semi-oracle touch "
          f"{st_touch_b * n_eval / len(rounds) / n_eval / EXPERT_B:.2f}")
    print(f"[diag] ORACLE repack (same mechanism, ranked by the upcoming item's raw per-(row,k) "
          f"counts): eng-acc {rpo_miss / n_eval / EXPERT_B:.2f}  touch {rpo_touch / n_eval / EXPERT_B:.2f}"
          f"/round  -> past vs future ranking gap = "
          f"{(rp_touch - rpo_touch) / n_eval / EXPERT_B:.2f} touch experts/round")
    st2_miss_b, st2_touch_b = sim_static(rounds, init, bnd, both=True, mult_count=True)
    print(f"[diag] control: static per-(row,k)-count ranking (oracle set, no mid-item swaps): "
          f"eng-acc {st2_miss_b * n_eval / len(rounds) / n_eval / EXPERT_B:.2f}  "
          f"touch {st2_touch_b * n_eval / len(rounds) / n_eval / EXPERT_B:.2f}/round"
          f"  => oracle-repack minus this = mid-item throttle churn on a perfect set")
    rpf_miss, rpf_touch = sim_repack(rounds, init, bnd, eval_from=eval_from, swaps=0)
    print(f"[diag] control: repack + freeze (past ranks, no mid-item swaps): eng-acc "
          f"{rpf_miss / n_eval / EXPERT_B:.2f}  touch {rpf_touch / n_eval / EXPERT_B:.2f}/round"
          f" -> mid-item swaps are essential adaptation from a past-rank core")
    bs = [b - eval_from for b in bnd[1:-1] if eval_from <= b < len(rounds)]
    hot = set(k for b in bs for k in range(b, min(b + 16, n_eval)))
    mh = [rp_track[k] for k in range(n_eval) if k in hot]
    mc = [rp_track[k] for k in range(n_eval) if k not in hot]
    uh = [mult_track[k] for k in range(n_eval) if k in hot]
    um = [mult_track[k] for k in range(n_eval) if k not in hot]
    oh = [obs_v[k] for k in range(n_eval) if k in hot]
    om = [obs_v[k] for k in range(n_eval) if k not in hot]
    print(f"[diag] eng-miss experts/round in the 16 rounds after each of the {len(bs)} item bounds "
          f"vs mid-item: observed {sum(oh) / len(oh):.1f}/{sum(om) / len(om):.1f}  "
          f"per-(row,k)-LFU (no repack) {sum(uh) / len(uh):.1f}/{sum(um) / len(um):.1f}  "
          f"repack@item {sum(mh) / len(mh):.1f}/{sum(mc) / len(mc):.1f}")
    print(f"[diag] per-round miss experts over eval window ({n_eval} rounds):"
          f" observed mean {sum(obs_v) / n_eval:.2f}")
    for name, v in [("token-LFU(sim-dedup)", tok_track), ("per-(row,k)-LFU", mult_track)]:
        d = [abs(a - b) for a, b in zip(v, obs_v)]
        print(f"[diag] calib {name:20} mean {sum(v) / len(v):6.2f}/rnd  mean|d| {sum(d) / len(d):5.2f}  "
              f"max|d| {max(d):4d}  corr {corr(v, obs_v):+.3f}  len-ok {len(v) == n_eval}")


def verify(rounds, budget=1788, slice_n=200):
    """A/B of the sim_belady rework and the LFU variants against the reference implementations.
    CPU-only; truncation/l3/calib are not touched."""
    import time
    bnd = item_bounds(rounds)
    init = init_set(rounds, budget, bnd)
    sl = rounds[:slice_n]
    print(f"verify: slice rounds [0, {slice_n}) with the full-trace init ({len(init)} experts)")
    t0 = time.perf_counter()
    a = sim_belady(sl, init, 0)
    t1 = time.perf_counter()
    b = sim_belady_naive(sl, init, 0)
    t2 = time.perf_counter()
    c = sim_belady_orig(sl, init, 0)
    t3 = time.perf_counter()
    print(f"sim_belady (heap)   {a:14d} miss bytes   {t1 - t0:6.2f}s")
    print(f"sim_belady_naive    {b:14d} miss bytes   {t2 - t1:6.2f}s")
    print(f"sim_belady_orig     {c:14d} miss bytes   {t3 - t2:6.2f}s")
    print(f"heap == naive (semantics exact): {a == b};  orig == heap: {a == c}"
          f" (orig delta {c - a:+d} bytes = {(c - a) // EXPERT_B:+d} misses)")
    tot_sp = sum(len(set(v[3])) for r in sl for v in r["visits"])
    tot_id = sum(len({x >> 1 for x in v[3]}) for r in sl for v in r["visits"])
    print(f"slice token==union counts: set(pairs) total {tot_sp} vs distinct-id total {tot_id} "
          f"-> identical: {tot_sp == tot_id}")
    print(f"slice sim_lfu token {sim_lfu(sl, init, eval_from=0)} vs union "
          f"{sim_lfu(sl, init, union=True, eval_from=0)} miss bytes")
    print(f"slice sim_lfu_mult (per-(row,k), engine model) {sim_lfu_mult(sl, init, eval_from=0)} miss bytes")
    try:
        import trace_sim
        r1 = trace_sim.sim_lfu(sl, init)
        r2 = sim_lfu(sl, init)
        r3 = trace_sim.sim_lfu(sl, init, union=True)
        print(f"sim_lfu vs pristine trace_sim.sim_lfu on slice: token {r1 == r2} ({r1}) "
              f"union {r3 == sim_lfu(sl, init, union=True)}")
    except Exception as ex:
        print(f"trace_sim.py cross-check skipped: {ex}")


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
        print("usage: trace_sim_res.py <trace.txt> <summary|residency|verify|truncation|l3|calib> [budget]",
              file=sys.stderr)
        return 1
    path, mode = sys.argv[1], sys.argv[2]
    budget = int(sys.argv[3]) if len(sys.argv) == 4 else 1788
    cache = path + ".pkl"
    if not os.path.exists(cache):  # run from another cwd: look next to this script
        cand = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.path.basename(path) + ".pkl")
        if os.path.exists(cand):
            cache = cand
    try:
        rounds = pickle.load(open(cache, "rb"))
    except Exception:
        rounds = parse(path)  # read-only rework: never rewrite the shared .pkl
    {"summary": summary, "residency": lambda rs: residency(rs, budget),
     "verify": lambda rs: verify(rs, budget=budget), "truncation": truncation,
     "l3": l3, "calib": calib}[mode](rounds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
