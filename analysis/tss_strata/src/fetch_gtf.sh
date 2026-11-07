#!/usr/bin/env bash
# Fetch one Ensembl GTF per species, pinned to the ASSEMBLY each model was
# trained on -- not to the newest release. Getting this wrong decouples the
# TSS coordinates from the sequence the model saw, silently.
#
#   zebrafish GRCz10   -> release 91  (last release carrying GRCz10; 92+ is GRCz11)
#   chicken   GRCg6a   -> release 105 (106+ is bGalGal1.mat.broiler.GRCg7b)
#   cattle    ARS-UCD1.2 -> release 110 (112 is ARS-UCD1.3)
#   human     GRCh38   -> release 112
#   salmon    Ssal_v3.1 -> release 112
#   pig       Sscrofa11.1 -> already on disk beside the genome, release 106
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GTF="$HERE/data/gtf"; mkdir -p "$GTF"
BASE=https://ftp.ensembl.org/pub

fetch () {  # species url_path
    local sp=$1 url=$2 dest="$GTF/$1.gtf.gz"
    if [[ -s "$dest" ]]; then echo "have   $sp"; return; fi
    echo "fetch  $sp"
    curl -sSf --retry 3 -o "$dest.part" "$url" && mv "$dest.part" "$dest"
}

fetch human     "$BASE/release-112/gtf/homo_sapiens/Homo_sapiens.GRCh38.112.gtf.gz"
fetch cattle    "$BASE/release-110/gtf/bos_taurus/Bos_taurus.ARS-UCD1.2.110.gtf.gz"
fetch chicken   "$BASE/release-105/gtf/gallus_gallus/Gallus_gallus.GRCg6a.105.gtf.gz"
fetch salmon    "$BASE/release-112/gtf/salmo_salar/Salmo_salar.Ssal_v3.1.112.gtf.gz"
fetch zebrafish "$BASE/release-91/gtf/danio_rerio/Danio_rerio.GRCz10.91.gtf.gz"

# pig ships with the training data
PIG=/cluster/projects/nn9780k/dat/training_data/genomes/Sus_scrofa.Sscrofa11.1.106.gtf
if [[ ! -s "$GTF/pig.gtf.gz" ]]; then echo "gzip   pig (local)"; gzip -c "$PIG" > "$GTF/pig.gtf.gz"; fi

ls -la "$GTF"
