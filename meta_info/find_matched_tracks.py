"""
Find matched (assay, tissue) track indices across all four species.

Reads the four *_track_names.txt files in meta_info/ and the ENTEX metadata
TSV, then identifies which track index in each species corresponds to the same
(assay, tissue) combination in every other species.

CTCF is excluded by experimental design (not shared by all species).
Brain-related tissues (Cerebellum, Cortex, Hypothalamus) are excluded because
human ENTEX does not contain them.
Salmon-specific tissues (Brain, Gill, Gonad, Sperm) are excluded.

Matched assays : ATAC, H3K27ac, H3K27me3, H3K4me1, H3K4me3
Matched tissues:
  human ↔ pig / cattle / pig-cattle : Adipose, Liver, Lung, Muscle, Spleen
  any ↔ salmon                       : Liver, Muscle only

Outputs (relative to workspace root):
    meta_info/matched_tracks.json  — machine-readable, keyed by species pair
    meta_info/matched_tracks.csv   — human-readable flat table

Run from workspace root (login node, no GPU/container needed):
    python meta_info/find_matched_tracks.py
"""

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

META_DIR = Path("meta_info")

TRACK_FILES = {
    "human":   META_DIR / "ENTEX_track_names.txt",
    "pig":     META_DIR / "pig_track_names.txt",
    "cattle":  META_DIR / "cattle_track_names.txt",
    "chicken": META_DIR / "chicken_track_names.txt",
    "salmon":  META_DIR / "salmon_track_names.txt",
    "zebrafish": META_DIR / "zebrafish_track_names.txt",
}

ENTEX_METADATA = META_DIR / "metadata_entex.tsv"

KEEP_ASSAYS = {"ATAC", "H3K27ac", "H3K27me3", "H3K4me1", "H3K4me3"}

# Human ENTEX uses free-text UBERON tissue names — map to canonical labels
HUMAN_TISSUE_MAP = {
    "subcutaneous adipose tissue": "Adipose",
    "right lobe of liver":         "Liver",
    "upper lobe of left lung":     "Lung",
    "gastrocnemius medialis":      "Muscle",
    "spleen":                      "Spleen",
}

# Canonical species ordering (determines JSON key names: earlier_later)
SPECIES_ORDER = ["human", "pig", "cattle", "chicken", "salmon", "zebrafish"]


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_human_tracks(track_file: Path, metadata_file: Path) -> List[Dict]:
    """Return [{idx, assay, tissue}] for human ENTEX 423 tracks."""
    lines = [l.strip() for l in track_file.read_text().splitlines() if l.strip()]
    acc_to_idx = {l.replace(".bed.gz", ""): i for i, l in enumerate(lines)}

    records = []
    with open(metadata_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            acc = row["File accession"].strip()
            if acc not in acc_to_idx:
                continue

            tissue = HUMAN_TISSUE_MAP.get(row["Biosample term name"].strip().lower())
            if tissue is None:
                continue

            assay_col  = row["Assay"].strip()
            target_col = row["Experiment target"].strip()

            if assay_col == "ATAC-seq":
                assay = "ATAC"
            elif assay_col in ("Histone ChIP-seq", "TF ChIP-seq") and target_col:
                assay = target_col.replace("-human", "").replace("-Human", "")
            else:
                continue

            if assay not in KEEP_ASSAYS:
                continue

            records.append({"idx": acc_to_idx[acc], "assay": assay, "tissue": tissue})
    return records


def parse_simple_tracks(track_file: Path) -> List[Dict]:
    """
    Parse pig or cattle track names.
    Pattern: <ASSAY>_<TISSUE>_<INDIVIDUAL>_Peaks.bed
    """
    records = []
    for i, line in enumerate(track_file.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        parts = line.replace("_Peaks.bed", "").split("_")
        if len(parts) < 2:
            continue
        assay, tissue = parts[0], parts[1]
        if assay not in KEEP_ASSAYS:
            continue
        records.append({"idx": i, "assay": assay, "tissue": tissue})
    return records


def parse_chicken_tracks(track_file: Path) -> List[Dict]:
    """
    Parse chicken track names.
    Pattern: <ASSAY>_<TISSUE>_<INDIVIDUAL>_Peaks.bed
    DNaseSeq is treated as equivalent to ATAC (both measure open chromatin)
    and is remapped to the canonical assay label 'ATAC'.
    Brain tissues (Cerebellum, Cortex, Hypothalamus) are excluded because
    human ENTEX does not contain them (kept for pig/cattle ↔ chicken pairs).
    CTCF excluded by design.
    """
    CHICKEN_BRAIN = {"Cerebellum", "Cortex", "Hypothalamus"}
    records = []
    for i, line in enumerate(track_file.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        parts = line.replace("_Peaks.bed", "").split("_")
        if len(parts) < 2:
            continue
        assay, tissue = parts[0], parts[1]
        # Treat DNaseSeq as open-chromatin equivalent to ATAC
        if assay == "DNaseSeq":
            assay = "ATAC"
        if assay not in KEEP_ASSAYS:
            continue
        records.append({"idx": i, "assay": assay, "tissue": tissue,
                        "_brain": tissue in CHICKEN_BRAIN})
    return records


def parse_salmon_tracks(track_file: Path) -> List[Dict]:
    """
    Parse salmon track names.
    Pattern: <ASSAY>_<TISSUE>_<STAGE>_<SEX>_<REP>.bed
    Assay may carry a 'ChIP-' prefix: ChIP-H3K27ac → H3K27ac.
    Only Liver and Muscle have matched tissues in other species.
    """
    SALMON_KEEP_TISSUES = {"Liver", "Muscle"}
    records = []
    for i, line in enumerate(track_file.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        parts = Path(line).stem.split("_")
        if len(parts) < 2:
            continue
        assay  = parts[0].replace("ChIP-", "")
        tissue = parts[1]
        if assay not in KEEP_ASSAYS or tissue not in SALMON_KEEP_TISSUES:
            continue
        records.append({"idx": i, "assay": assay, "tissue": tissue})
    return records


def parse_zebrafish_tracks(track_file: Path) -> List[Dict]:
    """
    Parse zebrafish track names.
    Pattern: zebrafish_<ASSAY>_<TISSUE>_merged_narrowpeak.bed

    Zebrafish profiles only three of the five shared assays (ATAC, H3K27ac,
    H3K4me3); H3K27me3 and H3K4me1 are absent, so zebrafish contributes to the
    pooled comparison but not to the per-assay ordering. Embryonic tissues have
    no counterpart in any other species and drop out of the intersection
    naturally, but are excluded explicitly so the intent is visible.
    """
    ZEBRAFISH_DROP_TISSUES = {"EmbryonicBrain", "EmbryonicTrunk"}
    records = []
    for i, line in enumerate(track_file.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        parts = Path(line).stem.replace("zebrafish_", "", 1).split("_")
        if len(parts) < 2:
            continue
        assay, tissue = parts[0], parts[1]
        if assay not in KEEP_ASSAYS or tissue in ZEBRAFISH_DROP_TISSUES:
            continue
        records.append({"idx": i, "assay": assay, "tissue": tissue})
    return records


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def find_matches(records_a: List[Dict], records_b: List[Dict],
                 sp_a: str, sp_b: str) -> List[Dict]:
    """Return list of matched (assay, tissue) groups with track indices per species."""
    def group(records):
        d = defaultdict(list)
        for r in records:
            d[(r["assay"], r["tissue"])].append(r["idx"])
        return d

    ga, gb = group(records_a), group(records_b)
    matches = []
    for key in sorted(ga.keys() & gb.keys()):
        assay, tissue = key
        matches.append({
            "assay":  assay,
            "tissue": tissue,
            sp_a:     sorted(ga[key]),
            sp_b:     sorted(gb[key]),
        })
    return matches


def pair_key(sp_a: str, sp_b: str) -> str:
    """Canonical JSON key for a species pair (earlier in SPECIES_ORDER comes first)."""
    if SPECIES_ORDER.index(sp_a) <= SPECIES_ORDER.index(sp_b):
        return f"{sp_a}_{sp_b}"
    return f"{sp_b}_{sp_a}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load all species tracks (chicken includes brain-flagged records)
    chicken_all = parse_chicken_tracks(TRACK_FILES["chicken"])

    tracks_all = {
        "human":   parse_human_tracks(TRACK_FILES["human"], ENTEX_METADATA),
        "pig":     parse_simple_tracks(TRACK_FILES["pig"]),
        "cattle":  parse_simple_tracks(TRACK_FILES["cattle"]),
        "chicken": chicken_all,
        "salmon":  parse_salmon_tracks(TRACK_FILES["salmon"]),
        "zebrafish": parse_zebrafish_tracks(TRACK_FILES["zebrafish"]),
    }

    # Chicken without brain — used when matching with human or salmon
    # (brain tissues absent from human ENTEX and salmon)
    tracks_chicken_no_brain = [r for r in chicken_all if not r.get("_brain")]

    print("Tracks passing assay+tissue filter:")
    for sp, recs in tracks_all.items():
        print(f"  {sp:8s}: {len(recs)}")
    print(f"  chicken (no brain): {len(tracks_chicken_no_brain)}")

    output = {}
    csv_rows = []

    for i, sp_a in enumerate(SPECIES_ORDER):
        for sp_b in SPECIES_ORDER[i + 1:]:
            key = pair_key(sp_a, sp_b)

            # Use brain-excluded chicken only when paired with human or salmon
            recs_a = tracks_all[sp_a]
            recs_b = tracks_all[sp_b]
            if sp_a == "chicken" and sp_b in ("human", "salmon"):
                recs_a = tracks_chicken_no_brain
            if sp_b == "chicken" and sp_a in ("human", "salmon"):
                recs_b = tracks_chicken_no_brain

            matches = find_matches(recs_a, recs_b, sp_a, sp_b)
            output[key] = matches

            print(f"\n{sp_a} ↔ {sp_b}  ({key}): {len(matches)} matched groups")
            for m in matches:
                print(f"  {m['assay']:12s} {m['tissue']:8s}  "
                      f"{sp_a}={m[sp_a]}  {sp_b}={m[sp_b]}")
                csv_rows.append({
                    "pair": key, "species_a": sp_a, "species_b": sp_b,
                    "assay": m["assay"], "tissue": m["tissue"],
                    "a_indices": ",".join(map(str, m[sp_a])),
                    "b_indices": ",".join(map(str, m[sp_b])),
                    "n_a": len(m[sp_a]), "n_b": len(m[sp_b]),
                })

    # Save JSON
    json_out = META_DIR / "matched_tracks.json"
    json_out.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {json_out}")

    # Save CSV
    csv_out = META_DIR / "matched_tracks.csv"
    fieldnames = ["pair", "species_a", "species_b", "assay", "tissue",
                  "a_indices", "b_indices", "n_a", "n_b"]
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved: {csv_out}")

    total = sum(len(v) for v in output.values())
    print(f"\nTotal matched groups across all pairs: {total}")


if __name__ == "__main__":
    main()
