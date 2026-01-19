import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import matplotlib

def read_data(files):
    dataframes = {}
    for file in files:
        df = pd.read_csv(file)
        dataframes[file] = df
    return dataframes


def plot_runtime_histogram(df, output_dir):
    for file, df in df.items():
        for benchmark in df['benchmark'].unique():
            plt.figure(figsize=(10, 6))
            plt.suptitle(f'Runtime Distribution of {benchmark}')

            for pi, problem_size in enumerate(df['problem_size'].unique()):
                ax = plt.subplot(1, len(df['problem_size'].unique()), pi + 1)
                plt.title(f'Problem Size: {problem_size}')

                subset = df[(df['benchmark'] == benchmark) & (df['problem_size'] == problem_size)]
                heights, edges, patches = plt.hist(subset['avg'], bins='auto', color='#164588', align='left')

                max_avg = subset['avg'].max()
                max_partition = subset[subset['avg'] == max_avg]['container'].iloc[0]
                plt.annotate(f'Max: {max_avg:.2f}\n({''.join(filter(lambda x: not x.isalpha(), max_partition))})',
                              xy=(max_avg, heights[-1]), xytext=(max_avg, heights[-1]*2),
                              arrowprops=dict(arrowstyle='fancy', facecolor='#EE8F00'),
                              bbox=dict(boxstyle="square", fc="w"),
                              fontsize=8, color="#EE8F00", horizontalalignment='center')

                min_avg = subset['avg'].min()
                min_partition = subset[subset['avg'] == min_avg]['container'].iloc[0]
                plt.annotate(f'Min: {min_avg:.2f}\n({''.join(filter(lambda x: not x.isalpha(), min_partition))})',
                              xy=(min_avg, heights[0]), xytext=(min_avg, heights[0]*2),
                              arrowprops=dict(arrowstyle='fancy', facecolor='#EE8F00'),
                              bbox=dict(boxstyle="square", fc="w"),
                              fontsize=8, color="#EE8F00", horizontalalignment='center')

                plt.xlabel(f'Runtime ({df["time_unit"].iloc[0]})')

                plt.ylabel('Frequency')
                plt.yscale('symlog')
                y_minor = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(1, 10)*0.1, numticks=10)
                ax.yaxis.set_minor_locator(y_minor)
                ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                plt.grid(axis='y', alpha=0.75)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{file}_{benchmark}_runtime_histogram.pdf'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, help="Set output directory", default=".")
    parser.add_argument('-i', '--input', type=str, nargs='+', help='<Required> Set input file(s)', required=True)
    args = parser.parse_args()

    data = read_data(args.input)
    plot_runtime_histogram(data, args.output)