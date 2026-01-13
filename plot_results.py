import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def read_data(files):
    dataframes = {}
    for file in files:
        df = pd.read_csv(file)
        dataframes[file] = df
    return dataframes


def plot_runtime_histogram(df, output_dir):
    for file, df in df.items():
        plt.figure(figsize=(10, 6))
        plt.hist(df['runtime'], bins=30, color='skyblue', edgecolor='black')
        plt.title('Runtime Distribution')
        plt.xlabel('Runtime (seconds)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{file}_runtime_histogram.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, help="Set output directory", default="")
    parser.add_argument('-i', '--input', type=list, nargs='+', help='<Required> Set input file(s)', required=True)
    args = parser.parse_args()
    
    data = read_data(args.input)