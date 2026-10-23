#!/usr/bin/env bash
# Rebuild the New England benchmark end to end from the public GivingTuesday 990 data lake.
# Every Monte Carlo stream is seeded, so results match the committed ones up to BLAS floating point.
#
#   export EMBAUX_DATA=~/embaux-data      # derived data (~3 GB for NE; ~11 GB with the national XML)
#   export MINILM_DIR=$EMBAUX_DATA/models/minilm/onnx
#   bash scripts/reproduce_ne.sh [step ...]     # steps: data pop emb stage1 s2a s2b s2f s34 notebooks (default: all)
#
# Wall time on an Apple-silicon laptop with W workers is roughly: data 20-40 min (network), pop 10 min,
# emb 15 min, stage1 20 min, s2a 8.5 CPU-h / W, s2b+s2f 1.5 CPU-h / W, s34 2 CPU-h / W.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; SRC="$ROOT/src"
D="${EMBAUX_DATA:?set EMBAUX_DATA}"; export MINILM_DIR="${MINILM_DIR:-$D/models/minilm/onnx}"
W="${WORKERS:-$(python3 -c 'import os; print(max(1, (os.cpu_count() or 4) - 2))')}"
export OMP_NUM_THREADS=1
S3=https://gt990datalake-rawdata.s3.amazonaws.com
STEPS="${*:-data pop emb stage1 s2a s2b s2f s34 notebooks}"
has() { [[ " $STEPS " == *" $1 "* ]]; }
mkdir -p "$D"/{raw990,xml,parsed,pop,emb,results,models}

if has data; then
  for y in 2021 2022 2023; do curl -sS -o "$D/raw990/idx_$y.csv" "$S3/Indices/990xmls/yearly/${y}_efiledata_xmls_created_on_2026-08-25.csv"; done
  curl -sS -o "$D/raw990/bmf_2023_8_16.csv" "$S3/EfileData/BMF/IRS_BMF_2023_8_16_raw.csv"
  python3 -I "$SRC/fetch990.py" --index-dir "$D/raw990" --out "$D/xml" --years 2021 2022 2023 --states MA CT ME NH VT RI --workers 32
  [ -f "$MINILM_DIR/model.onnx" ] || { curl -sSL -o "$D/models/onnx.tar.gz" https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz \
      && mkdir -p "$D/models/minilm" && tar -xzf "$D/models/onnx.tar.gz" -C "$D/models/minilm"; }
fi
if has pop; then
  for y in 2021 2022 2023; do
    python3 -I "$SRC/parse_all.py" --xml-dir "$D/xml" --manifest "$D/xml/manifest_$y.csv" --year $y --out "$D/parsed/ne_$y.parquet" --procs "$W"; done
  python3 -I "$SRC/build_population.py" --prev "$D/parsed/ne_2022.parquet" --curr "$D/parsed/ne_2023.parquet" --bmf "$D/raw990/bmf_2023_8_16.csv" \
      --out "$D/pop/ne_pop_v3.parquet" --prev2 "$D/parsed/ne_2021.parquet" --avail-date 2024-06-30 --index-t "$D/raw990/idx_2023.csv"
  python3 -I "$SRC/build_population.py" --prev "$D/parsed/ne_2021.parquet" --curr "$D/parsed/ne_2022.parquet" --bmf "$D/raw990/bmf_2023_8_16.csv" \
      --out "$D/pop/ne_wave2022_pop_v3.parquet" --avail-date 2023-06-30 --index-t "$D/raw990/idx_2022.csv"
fi
has emb && python3 -I "$ROOT/scripts/build_embeddings.py"
has stage1 && python3 -I "$ROOT/scripts/stage1_select.py"
if has s2a; then
  python3 -I "$SRC/stage2_ne.py" prep
  for k in $(seq 0 $((W - 1))); do python3 -I "$SRC/stage2_ne.py" run $k "$W" 2000 & done; wait
  python3 -I "$SRC/stage2_ne.py" merge "$W"
  for k in $(seq 0 $((W - 1))); do python3 -I "$SRC/stage2a_lincal.py" run $k "$W" 2000 & done; wait
  python3 -I "$SRC/stage2a_lincal.py" merge "$W"
fi
if has s2b; then
  for v in w wprof; do for k in $(seq 0 $((W - 1))); do python3 -I "$SRC/stage2b_ne.py" run $v $k "$W" 1000 & done; wait
    python3 -I "$SRC/stage2b_ne.py" merge $v "$W"; done
fi
has s2f && python3 -I "$SRC/stage2f_ne.py"
if has s34; then
  python3 -I "$SRC/stage34_ne.py" sae 500 &
  for s in late follow; do python3 -I "$SRC/stage34_ne.py" nr $s 1000 0 1 && python3 -I "$SRC/stage34_ne.py" merge_nr $s 1; done
  for k in $(seq 0 $((W - 1))); do python3 -I "$SRC/stage34_ne.py" dr 1000 $k "$W" & done; wait
  python3 -I "$SRC/stage34_ne.py" merge_dr "$W"
fi
if has notebooks; then
  cd "$ROOT/notebooks"
  for nb in 03_protocol_stage2_ne_verdict 04_protocol_nonresponse_and_sae 05_protocol_stage2b_weak_register 06_protocol_stage2f_stratification; do
    jupytext --to ipynb "$nb.py" -q && jupyter nbconvert --to notebook --execute --inplace "$nb.ipynb" --ExecutePreprocessor.timeout=7200; done
fi
