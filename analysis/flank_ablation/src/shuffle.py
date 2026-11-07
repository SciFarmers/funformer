"""Dinucleotide-preserving shuffle (Altschul-Erikson), numpy implementation.

A mononucleotide shuffle would destroy dinucleotide composition, and CpG
depletion alone is enough to shift a model's predictions, so the ablation has
to preserve the 16 dinucleotide counts exactly or "the flank stopped helping"
is confounded with "the flank stopped looking like DNA".

The algorithm: an Eulerian walk on the 4-node graph whose edges are the
dinucleotides. Shuffle each node's outgoing edge list, then fix the last-edge
set so the walk stays connected to the final vertex (Altschul & Erikson 1985),
and read the walk off. Connectivity is what makes the result a valid sequence
rather than a truncated one.
"""
import numpy as np


def _last_edge_tree(verts, last, succ, rng):
    """Pick one outgoing 'last edge' per vertex so they form a tree into `last`.

    This is the step that makes the walk Eulerian. Picking any edge that merely
    points at the final vertex is not enough: the chosen edges must let every
    vertex reach `last` by following them, or the walk dead-ends early and the
    dinucleotide counts are not preserved. With 4 vertices, rejection sampling
    converges immediately.
    """
    others = [v for v in verts if v != last and len(succ[v])]
    for _ in range(100):
        choice = {v: int(succ[v][rng.integers(len(succ[v]))]) for v in others}
        ok = True
        for v in others:                      # follow the chosen edges to `last`
            seen, cur = set(), v
            while cur != last:
                if cur in seen or cur not in choice:
                    ok = False
                    break
                seen.add(cur)
                cur = choice[cur]
            if not ok:
                break
        if ok:
            return choice
    return None


def _shuffle_run(seq, rng):
    """Eulerian-walk shuffle of one N-free run."""
    if len(seq) < 3:
        return seq.copy()
    verts = [int(v) for v in np.unique(seq)]
    succ = {v: seq[1:][seq[:-1] == v].copy() for v in verts}
    last = int(seq[-1])

    choice = _last_edge_tree(verts, last, succ, rng)
    if choice is None:
        return seq.copy()

    for v in verts:
        arr = succ[v]
        if len(arr) == 0:
            continue
        if v == last:
            rng.shuffle(arr)
            continue
        pick = int(np.flatnonzero(arr == choice[v])[0])
        rest = np.delete(arr, pick)
        rng.shuffle(rest)
        succ[v] = np.append(rest, choice[v])

    pos = {v: 0 for v in verts}
    out = np.empty(len(seq), dtype=seq.dtype)
    out[0] = cur = int(seq[0])
    for i in range(1, len(seq)):
        arr, p = succ[cur], pos[cur]
        if p >= len(arr):
            return seq.copy()
        pos[cur] = p + 1
        out[i] = cur = int(arr[p])
    return out


def dinuc_shuffle(tokens: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle preserving length and exact dinucleotide counts.

    `tokens` are Tokenizer ids (N=0, A=1, T=2, G=3, C=4). Each maximal N-free
    run is shuffled independently and N positions are left untouched. Pooling
    the runs first would be wrong: it creates dinucleotides spanning the gaps,
    so symbols migrate between runs and the within-run counts -- the only ones
    that exist in the original -- are not preserved.
    """
    if len(tokens) < 3:
        return tokens.copy()
    res = tokens.copy()
    nz = tokens != 0
    if not nz.any():
        return res
    # boundaries of maximal non-N runs
    pad = np.concatenate(([False], nz, [False]))
    d = np.diff(pad.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    for a, b in zip(starts.tolist(), ends.tolist()):
        if b - a >= 3:
            res[a:b] = _shuffle_run(tokens[a:b], rng)
    return res


def dinuc_counts(tokens: np.ndarray) -> np.ndarray:
    """16-vector of ACGT dinucleotide counts, N-containing pairs ignored."""
    a, b = tokens[:-1], tokens[1:]
    ok = (a != 0) & (b != 0)
    return np.bincount((a[ok] - 1) * 4 + (b[ok] - 1), minlength=16)


def gc_fraction(tokens: np.ndarray) -> float:
    ok = tokens != 0
    if not ok.any():
        return float("nan")
    return float(np.isin(tokens[ok], (3, 4)).mean())      # G=3, C=4
