export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/gcc/install/lib64

cmake . -DCMAKE_CXX_COMPILER=/root/gcc/install/bin/g++ -DCMAKE_C_COMPILER=/root/gcc/install/bin/gcc \
 -Dvec=off -Dbenchmark_DIR=/root/gbenchmark -DCMAKE_MODULE_PATH=/root/gbenchmark/cmake/Modules
 
datetime="$(date +%Y%m%d_%H%M%S)"
log_file="${datetime}.log"
out_file="${datetime}.out"
# log_file=/dev/stdout
#####################################
# Memory Access Footprint Experiment
#####################################
# 10 best and 10 worst for small problem size - 10 best and 10 worst for large problem size - aos - soa
python3 generate_datastructures.py --data_spec particle.spec --only \
    132540_6 034251_6 3164205 530124_6 452310_6 203415_6 3205641 5063241 512043_6 2105463  4350216 064_3521 31506_24 3510_264 0_125_36_4 30_15_24_6 063_521_4 0_1_4325_6 3054_1_2_6 4053_126 \
    240531_6 42301_56 04213_56 13024_65 053421_6 321450_6 02341_56 250431_6 324501_6 23140_65  3645201 02_53641 106_23_54 6430_15_2 3016_425 0_1_62_453 531402_6 10_6235_4 23405_61 503126_4 \
    0123456 0_1_2_3_4_5_6 

/usr/bin/time -v make main &> "${log_file}"
likwid-pin -C 0 ./main --input input_files --benchmark_enable_random_interleaving --benchmark_repetitions=10 \
    --benchmark_min_warmup_time=0.5 --benchmark_min_time=2s --benchmark_format=json &> "${out_file}"