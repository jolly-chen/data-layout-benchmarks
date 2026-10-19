"""Reference implementations of the benchmark kernels, used to generate validation data.

One module per kernel. Each module provides generate(inputs, input_size) and writes the
reference results next to the first input file, in the format main.cpp's --validation reads.
"""
