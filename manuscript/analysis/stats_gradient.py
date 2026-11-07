"""Significance tests for the phylogenetic transfer gradient.

Operates on the primary set (n=23 directions) from cross_species_margins.csv and
tests, non-parametrically, that the FunFormer-vs-baseline margin decreases with
divergence time. Two assay conventions are reported: all-assay (primary,
includes chromatin accessibility) and excl-ATAC (robustness check).

Each test is run for three margin definitions (see aggregate_cross_species.py):
  margin_all         FunFormer-20k vs best 1 kb CNN baseline (originally published;
                     confounds architecture with context length)
  margin_arch_all    FunFormer-20k vs FunFormer-1k -- architecture and parameter
                     count held fixed, so this isolates context length
  margin_best1kb_all FunFormer-20k vs the best of all four 1 kb models (strictest)
A margin column that is entirely absent (NaN) is skipped with a note, so this
runs before the FunFormer-1k jobs have landed.

Tests:

  1. Pearson r(margin, divergence_mya) with a permutation p-value.
  2. Jonckheere-Terpstra ordered-trend test across the three phylogenetic bins
     (mammal-mammal > mammal-chicken > tetrapod-fish), permutation p-value.
     This is the most appropriate test for the ordered hypothesis.
  3. mammal-mammal vs tetrapod-fish group difference (permutation).
  4. Sensitivity: drop the salmon->chicken outlier and re-run (1)-(3).

n=23 is small, and the 23 directions are drawn from only six genomes and four
distinct divergence times, so all p-values are permutation-based (exact-ish)
rather than asymptotic, and the ordered-trend (JT) test across the three bins is
the primary one. Chicken-alone comparisons (the mammal-chicken bin is n=3) are
underpowered and are NOT tested here on purpose -- rely on the pooled tests.
"""
import os
import itertools

import numpy as np
import pandas as pd

from divergence_times import BIN_ORDER, SPECIES, divergence, pair_bin

HERE = os.path.dirname(os.path.abspath(__file__))
# Under a frozen code snapshot __file__ points into the snapshot, which
# would send every figure and table there instead of into the repository.
if os.environ.get("REPO_ROOT"):
    HERE = os.path.join(os.environ["REPO_ROOT"], "manuscript", "analysis")
MARGINS = os.path.join(HERE, "out", "cross_species_margins.csv")
N_PERM = 200_000
SEED = 0


def pearson_perm(x, y, rng, n_perm=N_PERM):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    r = np.corrcoef(x, y)[0, 1]
    yp = y.copy()
    count = 0
    for _ in range(n_perm):
        rng.shuffle(yp)
        if abs(np.corrcoef(x, yp)[0, 1]) >= abs(r) - 1e-12:
            count += 1
    p = (count + 1) / (n_perm + 1)  # add-one, two-sided
    return r, p


def jt_statistic(values, groups, order):
    """Jonckheere-Terpstra J for groups ordered by `order` (dict group->rank).
    Counts, over every ordered pair of groups (i<j by rank), how often a value
    in the higher-rank group exceeds one in the lower-rank group (ties = 0.5).
    Under the *decreasing* hypothesis (margin falls as rank rises), J is small."""
    ranks = np.array([order[g] for g in groups])
    vals = np.asarray(values, float)
    uniq = sorted(set(ranks))
    J = 0.0
    for a_i in range(len(uniq)):
        for b_i in range(a_i + 1, len(uniq)):
            va = vals[ranks == uniq[a_i]]
            vb = vals[ranks == uniq[b_i]]
            diff = vb[:, None] - va[None, :]
            J += np.sum(diff > 0) + 0.5 * np.sum(diff == 0)
    return J


def jt_perm(values, groups, order, rng, n_perm=N_PERM):
    """Permutation p for the decreasing trend: how often a random group
    labelling gives a J at least as small as observed."""
    obs = jt_statistic(values, groups, order)
    groups = np.asarray(groups)
    vals = np.asarray(values, float)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(vals)
        if jt_statistic(perm, groups, order) <= obs + 1e-9:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return obs, p


def group_diff_perm(a, b, rng, n_perm=N_PERM):
    """One-sided permutation test that group a > group b (mean difference)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    na = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if (pool[:na].mean() - pool[na:].mean()) >= obs - 1e-12:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return obs, p


def relabelled_axes(df, mapping):
    """(divergence, bin) for every direction under one relabelling of the taxa.

    The single definition of what a relabelling does. Figure 3b, the
    sensitivity table and the statistics report all read the same null
    through it, so the tail convention cannot drift between them.
    """
    pairs = list(zip(df["source"], df["target"]))
    return (np.array([divergence(mapping[s], mapping[t]) for s, t in pairs]),
            np.array([pair_bin(mapping[s], mapping[t]) for s, t in pairs]))


def species_null_r(df, col):
    """(null r values over all 6! relabellings, observed r).

    Relabellings that leave divergence constant are dropped: r is undefined
    there. The observed value is computed from the column, not from the
    identity relabelling, so it matches what the rest of the pipeline reports.
    """
    y = df[col].to_numpy(float)
    rs = []
    for perm in itertools.permutations(SPECIES):
        x, _ = relabelled_axes(df, dict(zip(SPECIES, perm)))
        if x.std() > 0:
            rs.append(np.corrcoef(x, y)[0, 1])
    return np.asarray(rs), float(np.corrcoef(df["divergence_mya"].to_numpy(float), y)[0, 1])


def species_exact_p_r(df, col):
    """Observed r and its two-sided exact P -- the convention reported."""
    rs, obs = species_null_r(df, col)
    return obs, float((np.abs(rs) >= abs(obs) - 1e-12).mean())


def species_label_perm(df, col):
    """Exact permutation test with the SPECIES, not the direction, as the unit.

    The direction-level tests above treat the 23 primary directions as
    exchangeable. They are not: they are drawn from only six genomes related by
    a tree, so directions sharing a source or a target species are correlated,
    and the species themselves are not independent draws. Treating them as
    independent inflates the evidence.

    This test instead permutes the assignment of the six species onto the tree,
    recomputing each direction's divergence time and phylogenetic bin from the
    relabelled assignment while holding the observed margins fixed. It is the
    directional analogue of a Mantel test, and it respects both sources of
    non-independence at once. With six species there are exactly 6! = 720
    relabellings, so the null distribution is enumerated rather than sampled and
    the p-value is exact.

    Note the resolution floor: with 720 relabellings no p-value below 1/720 =
    0.0014 is attainable, however strong the effect.
    """
    y = df[col].to_numpy(float)

    ident = {sp: sp for sp in SPECIES}
    x0, b0 = relabelled_axes(df, ident)
    r_obs = np.corrcoef(x0, y)[0, 1]
    j_obs = jt_statistic(y, b0, BIN_ORDER)
    mm0, tf0 = y[b0 == "mammal-mammal"], y[b0 == "tetrapod-fish"]
    d_obs = mm0.mean() - tf0.mean()

    n_r = c_r = n_j = c_j = n_d = c_d = 0
    for perm in itertools.permutations(SPECIES):
        m = dict(zip(SPECIES, perm))
        x, b = relabelled_axes(df, m)
        if x.std() > 0:                      # two-sided, on |r|
            n_r += 1
            if abs(np.corrcoef(x, y)[0, 1]) >= abs(r_obs) - 1e-12:
                c_r += 1
        # A relabelling can map a pair onto the fish-fish bin, which has no
        # rank in BIN_ORDER (it is excluded from the primary set by design);
        # those rows are dropped from the JT statistic for that relabelling.
        keep = np.array([bb in BIN_ORDER for bb in b])
        if keep.any() and len(set(b[keep])) >= 2:
            n_j += 1
            if jt_statistic(y[keep], b[keep], BIN_ORDER) <= j_obs + 1e-9:
                c_j += 1
        mm, tf = y[b == "mammal-mammal"], y[b == "tetrapod-fish"]
        if len(mm) and len(tf):
            n_d += 1
            if (mm.mean() - tf.mean()) >= d_obs - 1e-12:
                c_d += 1
    return dict(r=r_obs, p_r=c_r / n_r, n_r=n_r,
                J=j_obs, p_j=c_j / n_j, n_j=n_j,
                d=d_obs, p_d=c_d / n_d, n_d=n_d)


def run(df, col, label, rng):
    print(f"\n{'='*60}\n{label}  (margin column: {col}, n={len(df)})\n{'='*60}")
    r, pr = pearson_perm(df["divergence_mya"], df[col], rng)
    print(f"Pearson r(margin, divergence) = {r:+.3f}   perm p = {pr:.5f}")

    obs, pj = jt_perm(df[col].values, df["bin"].values, BIN_ORDER, rng)
    print(f"Jonckheere-Terpstra (decreasing trend): J={obs:.1f}  perm p = {pj:.5f}")

    mm = df[df["bin"] == "mammal-mammal"][col].values
    sal = df[df["bin"] == "tetrapod-fish"][col].values
    d, pg = group_diff_perm(mm, sal, rng)
    print(f"mammal-mammal > tetrapod-fish: diff={d*100:+.2f}pp  perm p = {pg:.5f}")

    sp = species_label_perm(df, col)
    print(f"  [species-level exact test, all {sp['n_r']} relabellings -- "
          f"the valid unit of replication]")
    print(f"    Pearson r                     = {sp['r']:+.3f}   exact p = {sp['p_r']:.4f}")
    print(f"    Jonckheere-Terpstra  J={sp['J']:.1f}          exact p = {sp['p_j']:.4f}")
    print(f"    mammal-mammal > tetrapod-fish        exact p = {sp['p_d']:.4f}")


def main():
    df = pd.read_csv(MARGINS)
    prim = df[df["in_primary_set"]].copy()

    def _run_if_present(frame, col, label):
        """Run the test battery, unless that margin has no data yet."""
        if col not in frame.columns or frame[col].isna().all():
            print(f"\n[skipped] {label}: column '{col}' has no data "
                  f"(FunFormer-1k cross-species jobs not landed yet)")
            return
        rng = np.random.default_rng(SEED)
        run(frame.dropna(subset=[col]), col, label, rng)

    for col, label in [
        ("margin_arch_all",      "PRIMARY, architecture-matched (all-assay): FF-20k vs FF-1k"),
        ("margin_arch_noatac",   "PRIMARY, architecture-matched (excl-ATAC): FF-20k vs FF-1k"),
        ("margin_best1kb_all",   "STRICTEST (all-assay): FF-20k vs best of four 1 kb models"),
        ("margin_all",           "AS PUBLISHED (all-assay): FF-20k vs best CNN baseline"),
        ("margin_noatac",        "AS PUBLISHED (excl-ATAC): FF-20k vs best CNN baseline"),
    ]:
        _run_if_present(prim, col, label)

    # Sensitivity: drop the salmon->chicken outlier.
    loo = prim[~((prim["source"] == "salmon") & (prim["target"] == "chicken"))]
    for col, label in [
        ("margin_arch_all",    "SENSITIVITY excl salmon->chicken (arch-matched, all-assay)"),
        ("margin_arch_noatac", "SENSITIVITY excl salmon->chicken (arch-matched, excl-ATAC)"),
        ("margin_all",         "SENSITIVITY excl salmon->chicken (as published, all-assay)"),
        ("margin_noatac",      "SENSITIVITY excl salmon->chicken (as published, excl-ATAC)"),
    ]:
        _run_if_present(loo, col, label)


if __name__ == "__main__":
    main()
