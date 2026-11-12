#!/usr/bin/env python3
"""Check the sampled-draft wiring on a FNGINE_DRAFT_TRACE (format 2) of a sampling run.

  dist_check.py trace.bin [temperature=1.0] [top_p=0.95] [top_k=20] [presence=0] [frequency=0] [seed=N]

Per step: the stored draft distributions must equal the ones recomputed from the traced draft
logit rows (bridge row of the accepted column, then the AR rows) and hold the drafts. Across
steps: verified MTP drafts are the previous step's drafts; every acceptance has p(d) > 0; every
rejection's correction (the next anchor) has p - q > 0; the acceptance frequency matches
sum min(p, q). With seed=N (the request seed) every acceptance decision and every correction or
bonus token is recomputed exactly.

The traces came from a scratch build with draft-trace hooks (top-32 verify and draft-head logits
per step, plus the stored distributions); the hooks are not part of the repository.
"""
import math
import struct
import sys

import numpy as np

TOP, SUPPORT = 32, 20
path = sys.argv[1]
temp = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
top_p = float(sys.argv[3]) if len(sys.argv) > 3 else 0.95
top_k = int(sys.argv[4]) if len(sys.argv) > 4 else 20
presence = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
frequency = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
data = open(path, "rb").read()
off = 0
recs = []


def rows(n):
    global off
    out = []
    for _ in range(n):
        ids = np.frombuffer(data, dtype="<i4", count=TOP, offset=off)
        off += 4 * TOP
        vals = np.frombuffer(data, dtype="<f4", count=TOP, offset=off)
        off += 4 * TOP
        out.append((ids, vals))
    return out


while off < len(data):
    magic, T, k, acc, pos, first, bridge_rows, calls = struct.unpack_from("<8i", data, off)
    assert magic == 0x44524655, hex(magic)
    off += 32
    toks = np.frombuffer(data, dtype="<i4", count=16, offset=off)[:T].copy()
    off += 64
    verify = rows(T)
    bridge = rows(bridge_rows)
    ar = rows(max(calls - 1, 0))
    kind, d0, d1, d2 = struct.unpack_from("<4i", data, off)
    off += 16
    dists = []
    for _ in range(3):
        n = struct.unpack_from("<i", data, off)[0]
        ids = np.frombuffer(data, dtype="<i4", count=SUPPORT, offset=off + 4)[:n].tolist()
        probs = np.frombuffer(data, dtype="<f4", count=SUPPORT, offset=off + 4 + 4 * SUPPORT)[:n].tolist()
        dists.append(dict(zip(ids, probs)) if n else None)
        off += 176
    recs.append(dict(T=T, k=k, acc=acc, pos=pos, first=first, toks=toks, verify=verify, bridge=bridge, ar=ar,
                     kind=kind, drafts=[d0, d1, d2], dists=dists, calls=calls))
print(f"{len(recs)} records; sampling T={temp} top_p={top_p} top_k={top_k} presence={presence} frequency={frequency}")


def dist(ids, logits, counts=None, overlay=()):
    """Truncated distribution; `counts` + `overlay` are the penalty occurrence counts."""
    logits = logits.astype(np.float64).copy()
    if (presence or frequency) and counts is not None:
        for j, t in enumerate(ids.tolist()):
            c = counts.get(t, 0) + sum(1 for o in overlay if o == t)
            if c > 0:
                logits[j] -= presence + frequency * c
        order = sorted(range(len(ids)), key=lambda j: (-logits[j], int(ids[j])))
        ids, logits = ids[order], logits[order]
    ids, logits = ids[:top_k], logits[:top_k]
    w = np.exp((logits - logits[0]) / temp)
    total = w.sum()
    n = len(w)
    if top_p < 1.0:
        cum = 0.0
        for j in range(len(w)):
            cum += w[j]
            if cum >= top_p * total:
                n = j + 1
                break
    w = w[:n] / w[:n].sum()
    return dict(zip(ids[:n].tolist(), w.tolist()))


M64 = (1 << 64) - 1


def splitmix(x):
    x = (x + 0x9E3779B97F4A7C15) & M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M64
    return x ^ (x >> 31)


def uniform(seed, position, purpose, sub=0):
    key = splitmix(seed ^ ((position & 0xFFFFFFFF) * 0xD1B54A32D192ED03 & M64))
    key = splitmix(key ^ ((purpose & 0xFFFFFFFF) << 21) ^ (sub * 0x2545F4914F6CDD1D & M64))
    return (key >> 40) / 16777216.0


SEED = None
for arg in sys.argv:
    if arg.startswith("seed="):
        SEED = int(arg[5:])


def same(a, b, tol=2e-4):
    return a is not None and set(a) == set(b) and all(abs(a[t] - b[t]) < tol for t in a)


# counts_after[i]: occurrences of the generated tokens once record i's step has committed
# (its accepted drafts and its terminal, the next record's anchor).
counts_after = []
running = {}
for i, r in enumerate(recs):
    emitted = [] if r["first"] else r["toks"][1:1 + r["acc"]].tolist()
    if i + 1 < len(recs) and not recs[i + 1]["first"]:
        emitted.append(int(recs[i + 1]["toks"][0]))
    if r["first"]:
        running = {}
    for t in emitted:
        running[t] = running.get(t, 0) + 1
    counts_after.append(dict(running))

bad_dist = bad_member = checked = 0
for i, r in enumerate(recs):
    if r["calls"] == 0:
        continue  # no drafting (context tail)
    if i + 1 >= len(recs) or recs[i + 1]["first"]:
        continue  # the terminal of the last step is unknown
    col = min(r["acc"], len(r["bridge"]) - 1)
    sources = [r["bridge"][col]] + r["ar"]
    for j, src in enumerate(sources[:3]):
        want = dist(*src, counts_after[i], r["drafts"][:j])
        checked += 1
        if not same(r["dists"][j], want):
            bad_dist += 1
            if bad_dist <= 3:
                print("  dist mismatch at pos", r["pos"], "draft", j, r["dists"][j], want)
        if r["dists"][j] is None or r["drafts"][j] not in r["dists"][j]:
            bad_member += 1
print(f"stored distributions: {checked} checked, {bad_dist} differ from the traced logits, {bad_member} without their draft")

coupled_rounds = onehot_rounds = bad_tokens = 0
acc_events = acc_bad = rej_events = rej_bad = 0
expect = actual = var = 0.0
expect_d = var_d = 0.0
sure_rejected = 0
bypos, byratio = {}, {}
rng_checked = rng_bad = 0
term_checked = term_bad = 0
unsafe_rows = safe_n = 0
safe_expect = safe_var = safe_actual = 0.0
for i, (prev, cur, nxt) in enumerate(zip(recs, recs[1:], recs[2:] + [None])):
    if cur["first"] or cur["k"] == 0:
        continue
    if cur["kind"] != 1:
        onehot_rounds += 1
        continue
    coupled_rounds += 1
    k, a = cur["k"], cur["acc"]
    if cur["toks"][1:k + 1].tolist() != prev["drafts"][:k]:
        bad_tokens += 1
        continue
    # Terminal token (the next anchor) recomputed with the request seed: a rejection's correction
    # from the residual, or the bonus draw from the last column when every draft was accepted.
    if SEED is not None and nxt is not None and not nxt["first"]:
        c = int(nxt["toks"][0])
        pt = dist(*cur["verify"][a], counts_after[i], prev["drafts"][:a])
        if a < k:
            qt = prev["dists"][a]
            weights = [(t, max(pv - qt.get(t, 0.0), 0.0)) for t, pv in pt.items()]
            u = uniform(SEED, cur["pos"] + a + 1, 3)
        else:
            weights = list(pt.items())
            u = uniform(SEED, cur["pos"] + k + 1, 4)
        goal, acc_w, pick, margin = u * sum(w for _, w in weights), 0.0, weights[0][0], 1.0
        for t, w in weights:
            if w <= 0.0:
                continue
            acc_w += w
            pick = t
            margin = min(margin, abs(goal - acc_w))
            if goal < acc_w:
                break
        term_checked += 1
        if pick != c and margin > 1e-3:
            term_bad += 1
            if term_bad <= 8:
                print(f"  terminal mismatch pos {cur['pos']} a={a}/{k}: expected {pick}, engine {c}")
    for j in range(min(a + 1, k)):
        p = dist(*cur["verify"][j], counts_after[i], prev["drafts"][:j])
        q = prev["dists"][j]
        d = prev["drafts"][j]
        overlap = sum(min(pv, q.get(t, 0.0)) for t, pv in p.items())
        expect += overlap
        var += overlap * (1 - overlap)
        # Is the traced top-32 enough? An untraced token has a raw logit <= the 32nd traced one.
        vids, vlog = cur["verify"][j]
        def occ(t):
            return counts_after[i].get(t, 0) + prev["drafts"][:j].count(t)
        adj = sorted((float(vlog[x]) - (presence + frequency * occ(int(vids[x])) if occ(int(vids[x])) > 0 else 0.0)
                      for x in range(TOP)), reverse=True)
        safe = len(p) < top_k or adj[min(top_k, TOP) - 1] >= float(vlog[TOP - 1])
        unsafe_rows += not safe
        if safe:
            r2 = min(1.0, p.get(d, 0.0) / q[d])
            safe_expect += r2
            safe_var += r2 * (1 - r2)
            safe_actual += 1 if j < a else 0
            safe_n += 1
        ratio = min(1.0, p.get(d, 0.0) / q[d])
        bypos.setdefault(j, [0, 0.0, 0])
        bypos[j][0] += 1; bypos[j][1] += ratio; bypos[j][2] += 1 if j < a else 0
        b = min(int(ratio * 5), 4) if ratio < 1.0 else 5
        byratio.setdefault(b, [0, 0.0, 0])
        byratio[b][0] += 1; byratio[b][1] += ratio; byratio[b][2] += 1 if j < a else 0
        if SEED is not None:
            u = uniform(SEED, cur["pos"] + j + 1, 2)
            want = u * q[d] < p.get(d, 0.0)
            margin = abs(u * q[d] - p.get(d, 0.0))
            rng_checked += 1
            if want != (j < a) and margin > 1e-3:
                rng_bad += 1
                if rng_bad <= 12:
                    print(f"  decision mismatch pos {cur['pos']} draft {j}: u={u:.4f} q(d)={q[d]:.4f} p(d)={p.get(d, 0.0):.4f} -> expected {'ACC' if want else 'REJ'}, engine {'ACC' if j < a else 'REJ'}; occ(d)={occ(d)} |p|={len(p)}")
        expect_d += ratio
        var_d += ratio * (1 - ratio)
        if 0.5 <= ratio < 1.0 and "-v" in sys.argv:
            c = int(nxt["toks"][0]) if nxt is not None and j >= a else -1
            print(f"  pos {cur['pos']} draft {j} {'ACC' if j < a else 'REJ'} d={d} p(d)={p.get(d,0):.4f} q(d)={q[d]:.4f} ratio={ratio:.3f} occ(d)={occ(d)} |p|={len(p)} |q|={len(q)} corr={c} p(c)={p.get(c,0):.4f} q(c)={q.get(c,0):.4f} kind={cur['kind']} k={cur['k']}")
        if j >= a and ratio >= 1.0:
            sure_rejected += 1
            if sure_rejected <= 5:
                pen = sum(1 for t in cur["verify"][j][0].tolist() if counts_after[i].get(t, 0) + prev["drafts"][:j].count(t) > 0)
                print(f"  rejected with p(d) >= q(d): pos {cur['pos']} draft {j} d={d} p={p.get(d, 0.0):.4f} q={q[d]:.4f} |p|={len(p)} penalized among traced top-32: {pen}")
        if j < a:
            actual += 1
            acc_events += 1
            acc_bad += p.get(d, 0.0) <= 0.0
        else:
            rej_events += 1
            if nxt is not None and not nxt["first"]:
                c = int(nxt["toks"][0])
                rej_bad += not (p.get(c, 0.0) - q.get(c, 0.0) > 0.0)
z = (actual - expect) / math.sqrt(max(var, 1e-9))
print(f"rounds: {coupled_rounds} coupled, {onehot_rounds} one-hot; verified drafts differ from the previous MTP drafts in {bad_tokens}")
print(f"acceptances {acc_events} ({acc_bad} with p(d) = 0), rejections {rej_events} ({rej_bad} with a correction outside the residual)")
print(f"accepted {actual:.0f} of {acc_events + rej_events} examined drafts; expected sum min(p,q) = {expect:.1f} (z = {z:+.2f})")
zd = (actual - expect_d) / math.sqrt(max(var_d, 1e-9))
print(f"given the drafts: expected sum min(1, p(d)/q(d)) = {expect_d:.1f} (z = {zd:+.2f}); rejections with p(d) >= q(d): {sure_rejected}")
for j in sorted(bypos):
    print(f"  draft {j}: n {bypos[j][0]}, expected accepted {bypos[j][1]:.1f}, actual {bypos[j][2]}")
for b in sorted(byratio):
    name = "ratio = 1" if b == 5 else f"ratio [{b/5:.1f},{(b+1)/5:.1f})"
    print(f"  {name}: n {byratio[b][0]}, expected accepted {byratio[b][1]:.1f}, actual {byratio[b][2]}")
if SEED is not None:
    print(f"acceptance decisions recomputed with the request seed: {rng_checked} checked, {rng_bad} differ")
    print(f"correction/bonus tokens recomputed with the request seed: {term_checked} checked, {term_bad} differ")
zs = (safe_actual - safe_expect) / math.sqrt(max(safe_var, 1e-9))
print(f"rows whose traced top-{TOP} may miss candidates: {unsafe_rows}; on the other {safe_n}: accepted {safe_actual:.0f}, expected {safe_expect:.1f} (z = {zs:+.2f})")
ok = rng_bad == 0 and term_bad == 0 and bad_dist == 0 and bad_member == 0 and bad_tokens == 0 and acc_bad == 0 and rej_bad == 0 and abs(z) < 4
print("WIRING OK" if ok else "WIRING FAILED")
sys.exit(0 if ok else 1)
