"""Stratified AUROC from the cached predictions.

AUROC is computed as the rank statistic (Mann-Whitney U with average ranks for
ties) rather than via sklearn, because the login node has no sklearn and the
aggregation is pure CPU. The unstratified regression check in validate.py is
what proves the two agree: it recomputes every direction through this path and
compares against the sklearn numbers already committed in
eval_results/cross_species/.

Stratification is on POSITIVES only. Negatives are shared across strata --
every interval that is a negative for a track stays in both the proximal and
the distal calculation. That keeps the two AUROCs on a common scale: they
differ in which positives must be ranked above the same background, not in the
background itself.

A target track enters the analysis only if it clears the positive floor in
BOTH strata, so proximal and distal rest on an identical channel set and a
proximal-vs-distal difference cannot be an artefact of which channels were
measurable.
"""
import json
import os
import sys

import numpy as np
import pyarrow.ipc as ipc

# Self-contained so it does not depend on HERE, which is defined below
# and may itself point at a code snapshot.
ROOT = os.environ.get("REPO_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Outputs belong in the real analysis directory. Deriving this from
# __file__ puts them inside the code snapshot when one is in use.
HERE = os.path.join(ROOT, "analysis", "tss_strata") if os.environ.get("REPO_ROOT") \
    else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# REPO_ROOT lets a frozen snapshot of this code run against the real
# repository, where the checkpoints and datasets live.
sys.path.insert(0, os.path.join(HERE, "src"))
from manifest import TAG                                       # noqa: E402
from tss_distance import DATA as HF_DATA, HF                   # noqa: E402

MIN_POS = 50
STRATA = {"proximal": (0, 2_000), "distal": (10_001, np.inf)}
STRATA_SENS = {"proximal": (0, 1_000), "distal": (20_001, np.inf)}
STRATA_FULL = {"proximal": (0, 2_000), "middle": (2_001, 10_000),
               "distal": (10_001, np.inf)}

_cache = {}


def _pair_key(a, b):
    order = ["human", "pig", "cattle", "chicken", "salmon", "zebrafish"]
    return f"{a}_{b}" if order.index(a) <= order.index(b) else f"{b}_{a}"


def labels(species):
    if ("lab", species) not in _cache:
        p = os.path.join(HF_DATA, HF[species], "test", "data-00000-of-00001.arrow")
        with open(p, "rb") as fh:
            t = ipc.open_stream(fh).read_all()
        _cache[("lab", species)] = np.array(t["labels"].to_pylist(), dtype=np.int8)
    return _cache[("lab", species)]


def interval_keep(species):
    """Boolean mask of test intervals usable for a flank ablation.

    An interval whose flank is mostly N cannot be GC-matched to a donor, and
    the dinucleotide shuffle preserves N, so it comes out identical to intact
    in every condition and dilutes the measured effect. Excluding it from the
    ablated conditions AND from the intact reference keeps all comparisons on
    one interval set. Built by analysis/common/n_content.py.
    """
    if ("keep", species) not in _cache:
        p = os.path.join(ROOT, "analysis", "common", "data", "n_content",
                         f"{species}.tsv")
        _cache[("keep", species)] = np.loadtxt(p, skiprows=1, usecols=3,
                                               dtype=bool)
    return _cache[("keep", species)]


def tss_dist(species):
    if ("tss", species) not in _cache:
        p = os.path.join(HERE, "data", "tss_distance", f"{species}.tsv")
        _cache[("tss", species)] = np.loadtxt(p, skiprows=1, usecols=3, dtype=np.int64)
    return _cache[("tss", species)]


def matches(source, target):
    """Matched groups for a species pair. Self-pairs have no entry by design."""
    mt = json.load(open(os.path.join(ROOT, "meta_info", "matched_tracks.json")))
    return mt[_pair_key(source, target)]


def load_probs(source, target, model, cache_root=None):
    root = cache_root or os.path.join(HERE, "data", "cache")
    p = os.path.join(root, f"{source}_to_{target}", f"{model}.npz")
    z = np.load(p)
    col_of = {int(c): i for i, c in enumerate(z["cols"])}
    return z["probs"], col_of


def auroc(score, pos_mask, neg_mask):
    """Mann-Whitney AUROC over the selected positives and negatives."""
    s_pos, s_neg = score[pos_mask], score[neg_mask]
    n_p, n_n = len(s_pos), len(s_neg)
    if n_p == 0 or n_n == 0:
        return np.nan
    allv = np.concatenate([s_pos, s_neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks within ties
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[:n_p].sum() - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


def eligible_tracks(target, strata, keep=None):
    """Target tracks clearing MIN_POS positives in EVERY stratum.

    `keep` is the N-flank interval mask. When the exclusion is in force the
    floor must be counted on the intervals that survive it, not on the full
    set: a track whose positives all sit in excluded regions would otherwise
    clear the floor and then be scored on almost nothing.
    """
    L, d = labels(target), tss_dist(target)
    masks = {k: (d >= lo) & (d <= hi) for k, (lo, hi) in strata.items()}
    if keep is not None:
        masks = {k: m & keep for k, m in masks.items()}
    ok = {}
    for ti in range(L.shape[1]):
        y = L[:, ti] == 1
        counts = {k: int((y & m).sum()) for k, m in masks.items()}
        ok[ti] = (min(counts.values()) >= MIN_POS, counts)
    return ok, masks


def eligible_tracks_pooled(target, keep=None):
    """Target tracks clearing MIN_POS positives on the kept intervals.

    The stratified test demands the floor in every stratum, which is right
    when each stratum is scored separately. An unstratified call produces one
    pooled AUROC, so the floor belongs on the pooled count. Applying it in
    both places is what keeps one convention from yielding two different
    channel sets depending on the caller.
    """
    L = labels(target)
    ok = {}
    for ti in range(L.shape[1]):
        y = L[:, ti] == 1
        if keep is not None:
            y = y & keep
        n = int(y.sum())
        ok[ti] = (n >= MIN_POS, {"all": n})
    return ok


def direction(source, target, model, strata=STRATA, unstratified=False,
              cache_root=None, groups_for=None, drop_n_flanks=False):
    """Per-(assay, tissue) AUROC for one direction, per stratum.

    `groups_for` names the species pair whose matched groups define the
    channels, defaulting to (source, target). It differs only for the
    self-reference used by the retention figure: there the model and the labels
    are both the target's, so the pair is (source, target) while the indices on
    both sides are the target's own. matched_tracks.json holds no self-pair, so
    without this the self-reference would look up a key that cannot exist.

    Returns rows of dicts; `stratum` is "all" when unstratified=True.
    """
    probs, col_of = load_probs(source, target, model, cache_root)
    L = labels(target)
    usable = interval_keep(target) if drop_n_flanks else None
    if unstratified:
        ok, masks = eligible_tracks_pooled(target, keep=usable), {}
    else:
        ok, masks = eligible_tracks(target, strata, keep=usable)
    ga, gb = groups_for or (source, target)
    src_key = source if source != target else target
    rows = []
    for m in matches(ga, gb):
        src = [s for s in m[src_key] if s in col_of]
        tgt = [t for t in m[target] if ok[t][0]]
        if not src or not tgt:
            continue
        acc = {k: [] for k in (["all"] if unstratified else strata)}
        for si in src:
            sc = probs[:, col_of[si]].astype(np.float32)
            for ti in tgt:
                y = L[:, ti] == 1
                neg = ~y
                if usable is not None:
                    y = y & usable
                    neg = neg & usable
                if unstratified:
                    acc["all"].append(auroc(sc, y, neg))
                else:
                    for k, mk in masks.items():
                        acc[k].append(auroc(sc, y & mk, neg))
        for k, v in acc.items():
            v = [x for x in v if not np.isnan(x)]
            if v:
                rows.append(dict(source=source, target=target, model=model,
                                 assay=m["assay"], tissue=m["tissue"],
                                 stratum=k, n_pairs=len(v), auroc=float(np.mean(v))))
    return rows
