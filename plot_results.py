# !/usr/bin/env python3
#
# Script for plotting the results gathered using main.cpp
#

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import argparse
import os
import matplotlib

from itertools import permutations


def read_data(files):
    """
    Read CSV files into pandas DataFrames.

    :param files: List of file paths
    :return: Dictionary of DataFrames keyed by file name
    """
    dataframes = {}
    for file in files:
        df = pd.read_csv(file)
        dataframes[file] = df
    return dataframes


def is_overlapping_2D(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    """
    Check if two 2D intervals overlap.

    :param xmin1: Minimum x of first interval
    :param ymin1: Minimum y of first interval
    :param xmax1: Maximum x of first interval
    :param ymax1: Maximum y of first interval
    :param xmin2: Minimum x of second interval
    :param ymin2: Minimum y of second interval
    :param xmax2: Maximum x of second interval
    :param ymax2: Maximum y of second interval
    :return: True if the intervals overlap, False otherwise
    """
    return xmax1 >= xmin2 and xmax2 >= xmin1 and ymax1 >= ymin2 and ymax2 >= ymin1


def adjust_annotations(ax, annotations):
    """
    Adjust any annotations that overlap by stacking them vertically and shifting them
    to the right so the arrows remain visible.

    :param ax: Matplotlib Axes object
    :param annotations: List of Matplotlib Annotation objects
    """

    def update_annotation_positions(annotations, coords, idx, new_x, new_y, zorder):
        """
        Update the position of an annotation and its corresponding coordinates.

        :param annotations: List of Matplotlib Annotation objects
        :param coords: Array of annotation coordinates
        :param idx: Index of the annotation to update
        :param new_x: New x-coordinate for the annotation
        :param new_y: New y-coordinate for the annotation
        :param zorder: New z-order for the annotation
        """
        annotations[idx].set(position=(new_x, new_y), zorder=zorder)

        # Update the coords array with the new boundaries
        (new_xmin, new_ymin), (new_xmax, new_ymax) = ax.transData.inverted().transform(
            annotations[idx].get_window_extent()
        )
        coords[idx] = np.array([[new_xmin, new_ymin, new_xmax, new_ymax]])

    # Annotations need to be drawn to get their initial position
    ax.figure.draw_without_rendering()
    ax_xmin = ax.get_xlim()[0]

    coords = np.zeros_like(annotations, dtype=(float, 4))
    for ia, ann in enumerate(annotations):
        # The boundaries of the annotation need to be transformed from display to data coordinates
        (xmin, ymin), (xmax, ymax) = ax.transData.inverted().transform(
            ann.get_window_extent()
        )

        # The boundaries include the annotation arrow below the textbox,
        # so we compute the textbox heights as twice the diff between
        # the ymax to the y-coordinate of the text.
        # textbox_height = (ymax - ann.xyann[1]) * 2
        coords[ia] = np.array([[xmin, ymin, xmax, ymax]])

    for ic, (coord, ann) in enumerate(zip(coords, annotations)):
        # Shift annotations that are out of bounds to the right inside the axes
        if coord[0] < ax_xmin:
            update_annotation_positions(
                annotations,
                coords,
                ic,
                ax_xmin + 0.5 * (coord[2] - coord[0]),
                ann.xyann[1],
                ann.zorder,
            )

    for ic, (coord, ann) in enumerate(zip(coords, annotations)):
        collisions_idx = np.array(
            [
                ioc
                for ioc, c in enumerate(coords)
                if not np.equal(coord, c).all() and is_overlapping_2D(*coord, *c)
            ]
        )

        if collisions_idx.size > 0:
            # Include the current annotation index to readjust
            idx_to_readjust = np.array([ic] + collisions_idx.tolist())

            # Sort collisions by ymin position
            sorted_idx = idx_to_readjust[
                np.argsort(coords[idx_to_readjust], axis=0)[:, 1]
            ]

            # Readjust the y position of the boxes that overlap so that they
            # stack vertically. Readjust the x position to the right and change
            # the z-order to keep the arrows visible without covering the boxes below.
            for ii, idx in enumerate(sorted_idx[1:], start=1):
                textbox_ymin = annotations[idx].xyann[1]
                textbox_xmin, _, textbox_xmax, textbox_ymax = coords[idx]
                prev_textbox_ymin = annotations[sorted_idx[ii - 1]].xyann[1]
                prev_textbox_xmin, _, prev_textbox_xmax, prev_textbox_ymax = coords[
                    sorted_idx[ii - 1]
                ]

                textbox_overlap = is_overlapping_2D(
                    prev_textbox_xmin,
                    prev_textbox_ymin,
                    prev_textbox_xmax,
                    prev_textbox_ymax,
                    textbox_xmin,
                    textbox_ymin,
                    textbox_xmax,
                    textbox_ymax,
                )

                # Adjust the position of the box that is more to the right between the current and previous box.
                if prev_textbox_xmax >= textbox_xmax:
                    # Move previous box to the ruight and above the current box if the textboxes overlap
                    change_idx = sorted_idx[ii - 1]
                    new_x = textbox_xmax + 0.45 * (
                        prev_textbox_xmax - prev_textbox_xmin
                    )
                    new_y = (
                        textbox_ymax * 1.1
                        if textbox_overlap
                        else annotations[change_idx].xyann[1]
                    )
                else:
                    # Move current box to the right and above the previous box if the textboxes overlap
                    change_idx = idx
                    new_x = prev_textbox_xmax + 0.45 * (textbox_xmax - textbox_xmin)
                    new_y = (
                        prev_textbox_ymax * 1.1
                        if textbox_overlap
                        else annotations[change_idx].xyann[1]
                    )

                update_annotation_positions(
                    annotations,
                    coords,
                    change_idx,
                    new_x,
                    new_y,
                    10 + len(sorted_idx) - ii,
                )


def annotate_partition(ax, x, y, text, color, annotations):
    """
    Annotate a point in the histogram.

    :param ax: Matplotlib Axes object
    :param x: x-coordinate of the point to annotate
    :param y: y-coordinate of the point to annotate
    :param text: Text of the annotation
    :param color: Color of the annotation
    :param annotations: List to store the created annotations
    """
    a = ax.annotate(
        text,
        xy=(x, y),
        xytext=(x, y * 2),
        arrowprops=dict(arrowstyle="fancy", facecolor=color, alpha=0.8),
        bbox=dict(boxstyle="square", fc="w", alpha=0.8),
        fontsize=8,
        color=color,
        horizontalalignment="center",
    )

    annotations.append(a)


def annotate_minmax(ax, df_bp, heights, annotations):
    """
    Annotate the minimum and maximum average runtimes in the histogram
    with their corresponding partition.

    :param ax: Matplotlib Axes object
    :param df_bp: DataFrame containing benchmark runtime data
    :param heights: Heights of the histogram bars
    :param annotations: List to store the created annotations
    """
    max_avg = df_bp["avg"].max()
    max_partition = df_bp[df_bp["avg"] == max_avg]["container"].iloc[0]
    annotate_partition(
        ax,
        max_avg,
        heights[-1],
        f"Max: {max_avg:.2f}\n({''.join(filter(lambda x: not x.isalpha(), max_partition))})",
        "k",
        annotations,
    )

    min_avg = df_bp["avg"].min()
    min_partition = df_bp[df_bp["avg"] == min_avg]["container"].iloc[0]
    annotate_partition(
        ax,
        min_avg,
        heights[0],
        f"Min: {min_avg:.2f}\n({''.join(filter(lambda x: not x.isalpha(), min_partition))})",
        "k",
        annotations,
    )


def annotate_common(ax, df, edges, color, annotations):
    """
    Annotate the average runtimes of common partitioning schemes (AoS and SoA) in the histogram.
    Parts of the histogram that include AoS layouts with reordered data members are highlighted.

    :param ax: Matplotlib Axes object
    :param df: DataFrame containing benchmark runtime data
    :param edges: Edges of the histogram bins
    :param color: Color of the annotation
    :param annotations: List to store the created annotations
    """
    n_members = np.max([int(c) for c in df["container"].iloc[0] if c.isdigit()]) + 1
    aos_string = "".join([str(i) for i in range(n_members)])
    aos_avg = df[df["container"].str.contains(aos_string)]["avg"].iloc[0]
    aos_reordered = ["".join(perm) for perm in permutations(aos_string)]
    common_avg = df[df["container"].str.contains("|".join(aos_reordered))][
        "avg"
    ].to_list()

    soa_string = "_".join([str(i) for i in range(n_members)])
    soa_avg = df[df["container"].str.contains(soa_string)]["avg"].iloc[0]

    if not df[df["container"].str.contains("Contiguous")].empty:
        soa_reordered = ["_".join(perm) for perm in permutations(aos_string)]
        common_avg.extend(
            df[df["container"].str.contains("|".join(soa_reordered))]["avg"].to_list()
        )
    else:
        common_avg.append(soa_avg)

    heights, edges, _ = ax.hist(
        common_avg, bins=edges, color=color, align="left", label="Common Partitions"
    )

    annotate_partition(
        ax,
        aos_avg,
        heights[int(np.digitize(aos_avg, edges)) - 1],
        f"AoS: {aos_avg:.2f}\n({aos_string})",
        "#EE8F00",
        annotations,
    )

    annotate_partition(
        ax,
        soa_avg,
        heights[int(np.digitize(soa_avg, edges)) - 1],
        f"SoA: {soa_avg:.2f}\n({soa_string})",
        "#EE8F00",
        annotations,
    )


def plot_runtime_histogram(df, output_dir):
    """
    Plot runtime histograms for each benchmark and problem size.

    :param df: Dictionary of DataFrames keyed by file name
    :param output_dir: Directory to save the output plots
    """
    for file, df in df.items():
        for benchmark in df["benchmark"].unique():
            plt.figure(figsize=(10, 6))
            plt.suptitle(f"Runtime Distribution of {benchmark}")

            for pi, problem_size in enumerate(df["problem_size"].unique()):
                ax = plt.subplot(1, len(df["problem_size"].unique()), pi + 1)
                plt.title(f"Problem Size: {problem_size}")
                annotations = []

                df_bp = df[
                    (df["benchmark"] == benchmark)
                    & (df["problem_size"] == problem_size)
                ]
                heights, edges, _ = plt.hist(
                    df_bp["avg"],
                    bins="auto",
                    color="#164588",
                    align="left",
                    label="All Partitions",
                )

                annotate_minmax(ax, df_bp, heights, annotations)
                annotate_common(ax, df_bp, edges, "#EE8F00", annotations)

                plt.ylabel("Frequency")
                plt.yscale("symlog")
                y_minor = matplotlib.ticker.LogLocator(
                    base=10.0, subs=np.arange(1, 10) * 0.1, numticks=10
                )
                ax.yaxis.set_minor_locator(y_minor)
                ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                plt.grid(axis="y", alpha=0.75)

                plt.xlabel(f'Runtime ({df["time_unit"].iloc[0]})')
                ax.legend()
                adjust_annotations(ax, annotations)

            plt.tight_layout()
            print(f"Saving {file}_{benchmark}_avg_runtime_histogram.pdf...")
            plt.savefig(
                os.path.join(
                    output_dir, f"{file}_{benchmark}_avg_runtime_histogram.pdf"
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output", type=str, help="Set output directory", default="."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        nargs="+",
        help="<Required> Set input file(s)",
        required=True,
    )
    args = parser.parse_args()

    data = read_data(args.input)
    plot_runtime_histogram(data, args.output)
