#!/bin/bash

TOTAL_SIZE=37633 # For 7 data members
NUM_BATCHES=400
BATCH_SIZE=$(( (TOTAL_SIZE + NUM_BATCHES - 1) / (NUM_BATCHES) )) # Ceiling division

INPUT1_FILE=/data/data-layout-benchmarks/data/3m
INPUT2_FILE=/data/data-layout-benchmarks/data/3m_v2
OUTPUT_FILE=/data/data-layout-benchmarks/260305/260305-local

# cmake . -Dvec=on
for (( batch_num=0; batch_num < NUM_BATCHES; batch_num++ ))
do
    echo "Running batch $((batch_num + 1)) / $NUM_BATCHES (with batch size $BATCH_SIZE)"
    make clean
    python3 generate_datastructures.py --batch_size $BATCH_SIZE --batch_num $batch_num  --data_spec particle.spec
    /usr/bin/time -v make main
    likwid-pin -C 0 ./main --input1 $INPUT1_FILE --input2 $INPUT2_FILE --output $OUTPUT_FILE-vec --repetitions 10 --warmup 1 --papi_events "PAPI_TOT_INS,PAPI_L1_DCM,FP_ARITH:256B_PACKED_DOUBLE:256B_PACKED_SINGLE,PAPI_TLB_DM"
done

# cmake . -Dvec=off
# for (( batch_num=0; batch_num < NUM_BATCHES; batch_num++ ))
# do
#     echo "Running batch $((batch_num + 1)) / $NUM_BATCHES (with batch size $BATCH_SIZE)"
#     make clean
#     python3 generate_datastructures.py --batch_size $BATCH_SIZE --batch_num $batch_num
#     /usr/bin/time -v make main
#     likwid-pin -C 0 ./main --input1 $INPUT1_FILE --input2 $INPUT2_FILE --output $OUTPUT_FILE-novec --repetitions 10 --warmup 1 --papi_events "PAPI_TOT_CYC,PAPI_TOT_INS,FP_ARITH:256B_PACKED_DOUBLE:256B_PACKED_SINGLE,L2_LINES_IN:ALL,L2_RQSTS:MISS,MEM_LOAD_RETIRED:L3_MISS"
# done

# python3 generate_datastructures.py
# cmake . -Dvec=on
# make clean
# /usr/bin/time -v make main
# likwid-pin -C 0  ./main --input $INPUT1_FILE --input2 $INPUT2_FILE --output $OUTPUT_FILE

# cmake . -Dvec=off
# make clean
# /usr/bin/time -v make main
# likwid-pin -C 0  ./main --input $INPUT1_FILE --input2 $INPUT2_FILE --output $OUTPUT_FILE


# python3 generate_datastructures.py --data_spec particle.spec
# rm -f CMakeCache.txt
# /usr/bin/time -v make main
# likwid-pin -C 0  ./main --input1 $INPUT1_FILE --input2 $INPUT2_FILE --output $OUTPUT_FILE-vec --papi_events "PAPI_TOT_CYC,PAPI_VEC_INS,PAPI_TOT_INS,ANY_DATA_CACHE_FILLS_FROM_SYSTEM:LCL_L2:LOCAL_CCX:NEAR_CACHE_NEAR_FAR:DRAM_IO_NEAR:FAR_CACHE_NEAR_FAR:DRAM_IO_FAR:ALT_MEM_NEAR_FAR,PAPI_L1_DCA"


# cmake . -Dvec=off -DCMAKE_CXX_COMPILER=/usr/bin/clang++-20 -DCMAKE_C_COMPILER=/usr/bin/clang-20
# make clean
# /usr/bin/time -v make main
# likwid-pin -C 0  ./main --input $INPUT1_FILE --input2 $INPUT2_FILE --output $OUTPUT_FILE-novec --papi_events "PAPI_TOT_CYC,PAPI_VEC_INS,PAPI_TOT_INS,ANY_DATA_CACHE_FILLS_FROM_SYSTEM:LCL_L2:LOCAL_CCX:NEAR_CACHE_NEAR_FAR:DRAM_IO_NEAR:FAR_CACHE_NEAR_FAR:DRAM_IO_FAR:ALT_MEM_NEAR_FAR,PAPI_L1_DCA"


# NGT FARM
# cmake -Dvec=on .
# python3 generate_datastructures.py --data_spec particle.spec
# /usr/bin/time -v make main
# likwid-pin -C 0 ./main --input1 $INPUT1_FILE --input2 $INPUT2_FILE --output $OUTPUT_FILE --papi_events "PAPI_TOT_CYC,PAPI_VEC_INS,PAPI_TOT_INS,ANY_DATA_CACHE_FILLS_FROM_SYSTEM:LCL_L2:LOCAL_CCX:NEAR_CACHE_NEAR_FAR:DRAM_IO_NEAR:FAR_CACHE_NEAR_FAR:DRAM_IO_FAR:ALT_MEM_NEAR_FAR,PAPI_L1_DCA"
# likwid-pin -C 0 ./main --input1 3m --input2 3m_v2 --papi_events "PAPI_TOT_CYC,PAPI_VEC_INS,PAPI_TOT_INS,ANY_DATA_CACHE_FILLS_FROM_SYSTEM:LCL_L2:LOCAL_CCX:NEAR_CACHE_NEAR_FAR:DRAM_IO_NEAR:FAR_CACHE_NEAR_FAR:DRAM_IO_FAR:ALT_MEM_NEAR_FAR,PAPI_L1_DCA"

