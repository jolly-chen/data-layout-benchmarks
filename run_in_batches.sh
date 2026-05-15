#!/bin/bash

TOTAL_SIZE=37633 # For 7 data members
NUM_BATCHES=4
BATCH_SIZE=$(( (TOTAL_SIZE + NUM_BATCHES - 1) / (NUM_BATCHES) )) # Ceiling division

INPUT1_FILE=$1
INPUT2_FILE=$2
OUTPUT_FILE=$3

cmake . -Dvec=on

for (( batch_num=0; batch_num < NUM_BATCHES; batch_num++ ))
do
    echo "Running batch $((batch_num + 1)) / $NUM_BATCHES (with batch size $BATCH_SIZE)"
    python3 generate_datastructures.py --data_spec particle.spec --batch_size $BATCH_SIZE --batch_num $batch_num 
    /usr/bin/time -v make main
    likwid-pin -C 0 ./main --input1 $INPUT1_FILE --input2 $INPUT2_FILE --output $OUTPUT_FILE
done
