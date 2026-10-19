#!/bin/bash

TOTAL_SIZE=37633 # For 7 data members
NUM_BATCHES=4
BATCH_SIZE=$(( (TOTAL_SIZE + NUM_BATCHES - 1) / (NUM_BATCHES) )) # Ceiling division

if [ $# -ne 2 ]; then
    echo "Usage: $0 INPUT_LIST_FILE OUTPUT_PREFIX"
    echo "  INPUT_LIST_FILE  File listing the input datasets, see ./main --help"
    echo "  OUTPUT_PREFIX    Results of batch i are written to OUTPUT_PREFIX_<i>.json"
    exit 1
fi

INPUT_LIST_FILE=$1
OUTPUT_PREFIX=$2

cmake . -Dvec=on

for (( batch_num=0; batch_num < NUM_BATCHES; batch_num++ ))
do
    echo "Running batch $((batch_num + 1)) / $NUM_BATCHES (with batch size $BATCH_SIZE)"
    python3 layouts/generate_datastructures.py --data_spec layouts/specs/particle.spec --batch_size $BATCH_SIZE --batch_num $batch_num
    /usr/bin/time -v make main
    # -q: likwid-pin would otherwise print to stdout, into the JSON results
    likwid-pin -q -C 0 ./main --input "$INPUT_LIST_FILE" --benchmark_format=json > "${OUTPUT_PREFIX}_${batch_num}.json"
done
