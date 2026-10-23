export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/gcc/install/lib64

partitioned_container_count=$(grep -oF 'PartitionedContainer01234' main.cpp | wc -l)
if [ "$partitioned_container_count" -ne 6 ]; then
    echo "Error: expected PartitionedContainer01234 to appear 4 times in main.cpp; found $partitioned_container_count" >&2
    exit 1
fi

DATE=$(date +%Y%m%d)

# RESULTS=test
RESULTS=/shared/dlb-experiments/${DATE}-multistruct

RUN_CMD="likwid-pin -q -m -C 179 ./main --input Particle=inputs/im_input --input Hit=inputs/hit_input \
        --benchmark_enable_random_interleaving --benchmark_repetitions=10 \
        --benchmark_min_warmup_time=1 --benchmark_min_time=1x --benchmark_format=json \
        --benchmark_perf_counters=cycles,\
RETIRED_FP_OPS_BY_TYPE:VECTOR_ALL,\
RETIRED_FP_OPS_BY_TYPE:SCALAR_ADD,\
ANY_DATA_CACHE_FILLS_FROM_SYSTEM:DRAM_IO_NEAR:DRAM_IO_FAR,\
ANY_DATA_CACHE_FILLS_FROM_SYSTEM:LCL_L2:LOCAL_CCX:NEAR_CACHE_NEAR_FAR:DRAM_IO_NEAR:FAR_CACHE_NEAR_FAR:DRAM_IO_FAR:ALT_MEM_NEAR_FAR,\
RETIRED_SSE_AVX_FLOPS:ANY"

$RUN_CMD &> $RESULTS

# Picks the layout whose mean real_time is lowest for the given value of the
# "optimized" counter. Uses google-benchmark's mean aggregate (one entry per
# layout over the 10 repetitions) rather than the raw iteration runs.
best_layout() {
    jq -r --argjson opt "$1" '
        [.benchmarks[]
         | select(.name | contains("InvariantMassRandom_"))
         | select(.run_type == "aggregate" and .aggregate_name == "mean")
         | select(.optimized == $opt)]
        | min_by(.real_time)
        | .run_name
        | sub("^.*PartitionedContainer"; "PartitionedContainer")
    ' "$RESULTS"
}
im_layout_opt1=$(best_layout 1)
im_layout_opt0=$(best_layout 0)
echo "im_optimized=1: $im_layout_opt1"
echo "im_optimized=0: $im_layout_opt0"

# Picks the layout whose mean real_time is lowest for the given value of the
# "optimized" counter. Uses google-benchmark's mean aggregate (one entry per
# layout over the 10 repetitions) rather than the raw iteration runs.
best_layout() {
    jq -r --argjson opt "$1" '
        [.benchmarks[]
         | select(.name | contains("InvariantMassRandomOut"))
         | select(.run_type == "aggregate" and .aggregate_name == "mean")
         | select(.optimized == $opt)]
        | min_by(.real_time)
        | .run_name
        | sub("^.*PartitionedContainer"; "PartitionedContainer")
    ' "$RESULTS"
}
imo_layout_opt1=$(best_layout 1)
imo_layout_opt0=$(best_layout 0)
echo "imo_optimized=1: $imo_layout_opt1"
echo "imo_optimized=0: $imo_layout_opt0"

# Picks the layout whose mean real_time is lowest for the given value of the
# "optimized" counter. Uses google-benchmark's mean aggregate (one entry per
# layout over the 10 repetitions) rather than the raw iteration runs.
best_layout() {
    jq -r --argjson opt "$1" '
        [.benchmarks[]
         | select(.name | contains("FitTracks"))
         | select(.run_type == "aggregate" and .aggregate_name == "mean")
         | select(.optimized == $opt)]
        | min_by(.real_time)
        | .run_name
        | sub("^.*PartitionedContainer"; "PartitionedContainer")
    ' "$RESULTS"
}
tf_layout_opt1=$(best_layout 1)
tf_layout_opt0=$(best_layout 0)
echo "optimised Track: $tf_layout_opt1"
echo "optimised Hit: $tf_layout_opt0"

sed -i -E "0,/BM_OptimizeFirstC<K, C, layouts::Particle::PartitionedContainer01234/s//BM_OptimizeFirstC<K, C, layouts::Particle::${im_layout_opt1}/" main.cpp
sed -i -E "0,/BM_OptimizeSecondC<K, C, layouts::Particle::PartitionedContainer01234/s//BM_OptimizeSecondC<K, C, layouts::Particle::${im_layout_opt0}/" main.cpp

sed -i -E "0,/BM_OptimizeFirstC<K, C, layouts::ParticleOut::PartitionedContainer02134/s//BM_OptimizeFirstC<K, C, layouts::ParticleOut::${imo_layout_opt1}/" main.cpp
sed -i -E "0,/BM_OptimizeSecondC<K, C, layouts::ParticleOut::PartitionedContainer01234/s//BM_OptimizeSecondC<K, C, layouts::ParticleOut::${imo_layout_opt0}/" main.cpp

sed -i -E "0,/layouts::Track::PartitionedContainer01234/s//layouts::Track::${tf_layout_opt1}/" main.cpp
sed -i -E "0,/layouts::Hit::PartitionedContainer01234/s//layouts::Hit::${tf_layout_opt0}/" main.cpp

make main

$RUN_CMD &> $RESULTS-opt
