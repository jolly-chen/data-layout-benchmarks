#!/bin/bash

cd /root/src
mkdir results

#
# Generate the data structures for the benchmarks and compile with archjitecture-specific optimizations
#

if [[ "$*" == *"--quick"* ]]; then
    echo "Running in quick mode (only some data layouts)."
    python3 generate_datastructures.py --data_spec particle.spec --only_every 2000
else
    echo "Running in full mode (all data layouts)."
    python3 generate_datastructures.py --data_spec particle.spec
fi

cmake . -Dvec=on
make

#
# Execute the benchmrk suite
#

papi_flags=""
if papi_native_avail | grep -q "ANY_DATA_CACHE_FILLS_FROM_SYSTEM"; then
    papi_cache_fill_avail=true
    echo "Performance event ANY_DATA_CACHE_FILLS_FROM_SYSTEM available for Figure 5."
    papi_flags="ANY_DATA_CACHE_FILLS_FROM_SYSTEM:LCL_L2:LOCAL_CCX:NEAR_CACHE_NEAR_FAR:DRAM_IO_NEAR:FAR_CACHE_NEAR_FAR:DRAM_IO_FAR:ALT_MEM_NEAR_FAR"
else
    papi_cache_fill_avail=false
    echo "Warning: performance event ANY_DATA_CACHE_FILLS_FROM_SYSTEM not available for Figure 5. Will run without it."
fi

if papi_avail | grep "PAPI_TOT_INS" | grep -q "Yes"; then
    echo "Performance event PAPI_TOT_INS available for Figure 6."
    papi_flags="$papi_flags,PAPI_TOT_INS"
else
    echo "Warning: performance event PAPI_TOT_INS not available for Figure 6. Will run without it."
fi

./main --input1 datasets/3m --input2 datasets/3m_v2 --output results/results.csv --papi_events $papi_flags

python3 plot_results.py -i results/results.csv -o /root/src/results
