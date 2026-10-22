"""Generators of benchmark input datasets.

One module per struct. Each module provides COLUMNS and generate(ngen, seed), which yields
rows with values for COLUMNS. Columns must be named like data members of the struct, since
main.cpp maps the header row of a dataset to the members of the struct by name.
"""
