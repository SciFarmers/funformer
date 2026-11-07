"""Pairwise divergence times (millions of years ago) between the six species.

Single source of truth for the phylogeny used in the cross-species analysis.
Values are verified against TimeTree (timetree.org) and the primary literature
rather than memorised guesses. Where sources give a range, we store the midpoint
and keep the range in ``DIVERGENCE_RANGE_MYA`` for the Methods text.

Key phylogenetic fact for the paper: the Sarcopterygii/Actinopterygii split
(tetrapods vs. ray-finned fish, ~435 Mya) PREDATES the Synapsida/Sauropsida
(mammal vs. bird, ~320 Mya) split. So salmon is roughly equidistant from
chicken and from mammals -- it is NOT phylogenetically closer to chicken,
contrary to naive intuition.

Zebrafish is a second ray-finned fish, added to control for the salmonid
specific whole-genome duplication (Ss4R, ~80-100 Mya) that Atlantic salmon
carries and zebrafish does not. Both are teleosts and both therefore sit at
the same ~435 Mya split from every tetrapod here, so a tetrapod->zebrafish
transfer and a tetrapod->salmon transfer are matched on divergence time and
differ (among other things) in Ss4R. Zebrafish and salmon diverged from each
other ~206 Mya (Otophysi vs Protacanthopterygii within Teleostei), which is a
fish-fish comparison and is binned separately -- it must NOT be pooled with
the 435 Mya tetrapod-fish pairs.
"""

SPECIES = ["human", "pig", "cattle", "salmon", "chicken", "zebrafish"]

# Clade of each species, for grouping directions by the coarser phylogenetic bin.
CLADE = {
    "human": "mammal",
    "pig": "mammal",
    "cattle": "mammal",
    "chicken": "bird",
    "salmon": "fish",
    "zebrafish": "fish",
}

# Midpoint divergence time (Mya), symmetric. Keyed by frozenset of the two species.
DIVERGENCE_MYA = {
    frozenset({"pig", "cattle"}): 58.0,       # ~51-65 Mya (Laurasiatheria: Cetartiodactyla)
    frozenset({"human", "pig"}): 94.0,        # ~90-96 Mya (Boreoeutheria split)
    frozenset({"human", "cattle"}): 94.0,     # ~90-96 Mya (Boreoeutheria split)
    frozenset({"human", "chicken"}): 320.0,   # Synapsida/Sauropsida amniote split
    frozenset({"pig", "chicken"}): 320.0,
    frozenset({"cattle", "chicken"}): 320.0,
    frozenset({"human", "salmon"}): 435.0,    # Sarcopterygii/Actinopterygii split (~430-450)
    frozenset({"pig", "salmon"}): 435.0,
    frozenset({"cattle", "salmon"}): 435.0,
    frozenset({"chicken", "salmon"}): 435.0,  # same tetrapod/fish split -- predates the amniote split
    # Zebrafish: same Sarcopterygii/Actinopterygii split as salmon, so matched
    # on divergence with every tetrapod; ~206 Mya from salmon within Teleostei.
    frozenset({"human", "zebrafish"}): 435.0,
    frozenset({"pig", "zebrafish"}): 435.0,
    frozenset({"cattle", "zebrafish"}): 435.0,
    frozenset({"chicken", "zebrafish"}): 435.0,
    frozenset({"salmon", "zebrafish"}): 206.0,  # Otophysi vs Protacanthopterygii
}

DIVERGENCE_RANGE_MYA = {
    frozenset({"pig", "cattle"}): (51.0, 65.0),
    frozenset({"human", "pig"}): (90.0, 96.0),
    frozenset({"human", "cattle"}): (90.0, 96.0),
    frozenset({"human", "chicken"}): (312.0, 330.0),
    frozenset({"pig", "chicken"}): (312.0, 330.0),
    frozenset({"cattle", "chicken"}): (312.0, 330.0),
    frozenset({"human", "salmon"}): (430.0, 450.0),
    frozenset({"pig", "salmon"}): (430.0, 450.0),
    frozenset({"cattle", "salmon"}): (430.0, 450.0),
    frozenset({"chicken", "salmon"}): (430.0, 450.0),
    frozenset({"human", "zebrafish"}): (430.0, 450.0),
    frozenset({"pig", "zebrafish"}): (430.0, 450.0),
    frozenset({"cattle", "zebrafish"}): (430.0, 450.0),
    frozenset({"chicken", "zebrafish"}): (430.0, 450.0),
    frozenset({"salmon", "zebrafish"}): (200.0, 230.0),
}


def divergence(a, b):
    """Divergence time in Mya between two species (order-independent)."""
    if a == b:
        return 0.0
    return DIVERGENCE_MYA[frozenset({a, b})]


def pair_bin(a, b):
    """Coarse phylogenetic bin for a directed/undirected pair, for the ordered
    (Jonckheere-Terpstra) trend test, ordered by increasing divergence:
    mammal-mammal (58-94) < mammal-chicken (320) < tetrapod-fish (435).

    "tetrapod-fish" covers every tetrapod-to-fish pair, salmon and zebrafish
    alike. The lone fish-fish pair
    (salmon-zebrafish, ~206 Mya) is labelled separately so that
    aggregate_cross_species.py can exclude it from the primary set: it is a
    single pair whose two directions disagree in sign, and pooling it with
    the 435 Mya pairs would also place a 206 Mya comparison at the far end of
    the divergence axis.
    """
    clades = {CLADE[a], CLADE[b]}
    if clades == {"mammal"}:
        return "mammal-mammal"
    if clades == {"fish"}:
        return "fish-fish"
    if "fish" in clades:
        return "tetrapod-fish"
    if "bird" in clades:
        return "mammal-chicken"
    raise ValueError(f"unhandled pair: {a}, {b}")


# Ordinal rank of each bin for the JT ordered-trend test (increasing divergence).
# fish-fish is deliberately absent: it is excluded from the primary set.
BIN_ORDER = {"mammal-mammal": 0, "mammal-chicken": 1, "tetrapod-fish": 2}
