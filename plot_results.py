# !/usr/bin/env python3
#
# Script for plotting the results gathered using main.cpp
#

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import matplotlib
import matplotlib.ticker as ticker

from itertools import permutations

###########
# HELPERS #
###########


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


def get_agg_value(df, val, aggregate):
    """
    Get the aggregation of the given value for each partition in the DataFrame, using the specified aggregation method.

    :param df: DataFrame containing benchmark runtime data
    :param val: Value to aggregate
    :param aggregate: Metric to aggregate over (min, max, avg)
    :return: Aggregated value
    """
    if val == "time" and "time" not in df.columns:
        return df[aggregate]
    else:
        vals = df.groupby("container")[val]
        if aggregate == "min":
            return vals.min()
        elif aggregate == "max":
            return vals.max()
        elif aggregate == "avg":
            return vals.mean()
        elif aggregate == "stddev":
            return vals.std()
        else:
            raise ValueError(f"Unknown aggregate metric: {aggregate}")


def get_partition_from_val(df, val, aggregate):
    """
    Get the partition corresponding to a given aggregated value.

    :param df: DataFrame containing benchmark runtime data
    :param val: Aggregated value
    :param aggregate: Metric to aggregate over (min, max, avg)
    :return: Partition string
    """
    if "time" in df.columns:
        df_grouped = df.groupby("container")["time"].mean()
        partition = df_grouped[df_grouped == val].index[0]
    else:
        partition = df[df[aggregate] == val]["container"].iloc[0]

    return partition


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


#############
# HISTOGRAM #
#############


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


def annotate_minmax(ax, df_bp, heights, annotations, aggregate):
    """
    Annotate the minimum and maximum average runtimes in the histogram
    with their corresponding partition.

    :param ax: Matplotlib Axes object
    :param df_bp: DataFrame containing benchmark runtime data
    :param heights: Heights of the histogram bars
    :param annotations: List to store the created annotations
    :param aggregate: Metric to aggregate over (min, max, avg)
    """
    max_val = get_agg_value(df_bp, "time", aggregate).max()
    max_partition = get_partition_from_val(df_bp, max_val, aggregate)
    annotate_partition(
        ax,
        max_val,
        heights[-1],
        f"Max: {max_val:.2f}\n({''.join(filter(lambda x: not x.isalpha(), max_partition))})",
        "k",
        annotations,
    )

    min_val = get_agg_value(df_bp, "time", aggregate).min()
    min_partition = get_partition_from_val(df_bp, min_val, aggregate)
    annotate_partition(
        ax,
        min_val,
        heights[0],
        f"Min: {min_val:.2f}\n({''.join(filter(lambda x: not x.isalpha(), min_partition))})",
        "k",
        annotations,
    )


def annotate_common(ax, df, edges, annotations, aggregate):
    """
    Annotate the average runtimes of common partitioning schemes (AoS and SoA) in the histogram.
    Parts of the histogram that include AoS layouts with reordered data members are highlighted.

    :param ax: Matplotlib Axes object
    :param df: DataFrame containing benchmark runtime data
    :param edges: Edges of the histogram bins
    :param color: Color of the annotation
    :param annotations: List to store the created annotations
    :param aggregate: Metric to aggregate over (min, max, avg)
    """
    # AoS
    n_members = np.max([int(c) for c in df["container"].iloc[0] if c.isdigit()]) + 1
    aos_string = "".join([str(i) for i in range(n_members)])
    aos_reordered = ["".join(perm) for perm in permutations(aos_string)]
    aos_reordered_val = get_agg_value(
        df[df["container"].str.contains("|".join(aos_reordered))], "time", aggregate
    )

    heights, edges, _ = ax.hist(
        aos_reordered_val,
        bins=edges,
        color="#C00000",
        align="mid",
        label="AoS (Reordered)",
    )

    aos_val = get_agg_value(
        df[df["container"].str.contains(aos_string)], "time", aggregate
    )
    if not aos_val.empty:
        annotate_partition(
            ax,
            aos_val.iloc[0],
            heights[int(np.digitize(aos_val.iloc[0], edges)) - 1],
            f"AoS: {aos_val.iloc[0]:.2f}\n({aos_string})",
            "#C00000",
            annotations,
        )

    # SoA
    soa_string = "_".join([str(i) for i in range(n_members)])
    if not df[df["container"].str.contains("Contiguous")].empty:
        soa_reordered = ["_".join(perm) for perm in permutations(aos_string)]
        soa_reordered_val = get_agg_value(
            df[df["container"].str.contains("|".join(soa_reordered))], "time", aggregate
        )
    else:
        soa_reordered_val = get_agg_value(
            df[df["container"].str.contains(soa_string)], "time", aggregate
        )

    heights, edges, _ = ax.hist(
        soa_reordered_val,
        bins=edges,
        color="#EE8F00",
        align="mid",
        label="SoA (Reordered)",
    )

    soa_val = get_agg_value(
        df[df["container"].str.contains(soa_string)], "time", aggregate
    )
    if not soa_val.empty:
        annotate_partition(
            ax,
            soa_val.iloc[0],
            heights[int(np.digitize(soa_val.iloc[0], edges)) - 1],
            f"SoA: {soa_val.iloc[0]:.2f}\n({soa_string})",
            "#EE8F00",
            annotations,
        )


def plot_runtime_histogram(df, output_dir, aggregate):
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
                    get_agg_value(df_bp, "time", aggregate),
                    bins="auto",
                    color="#164588",
                    align="mid",
                    label="All Partitions",
                )

                annotate_minmax(ax, df_bp, heights, annotations, aggregate)
                annotate_common(ax, df_bp, edges, annotations, aggregate)

                ax.set_ylabel("Frequency")
                ax.set_yscale("symlog")
                y_minor = matplotlib.ticker.LogLocator(
                    base=10.0, subs=np.arange(1, 10) * 0.1, numticks=10
                )
                ax.yaxis.set_minor_locator(y_minor)
                ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                ax.grid(axis="y", alpha=0.75)

                ax.set_xlabel(f'Runtime ({df["time_unit"].iloc[0]})')
                ax.legend()
                adjust_annotations(ax, annotations)

            plt.tight_layout()
            print(f"Saving {file}_{benchmark}_{aggregate}_runtime_histogram.pdf...")
            plt.savefig(
                os.path.join(
                    output_dir, f"{file}_{benchmark}_{aggregate}_runtime_histogram.pdf"
                )
            )


############
# BARPLOTS #
############


def adjust_xticks(ax, xvals):
    """
    Adjust the xtick labels by lowering them if they are detected to be overlapping.

    :param ax: Matplotlib Axes object
    :param xvals: List of x values corresponding to the xticks
    """
    sorted_xlabels = sorted(ax.get_xticklabels(), key=lambda x: x.get_position()[0])
    for i, xtick in enumerate(sorted_xlabels[1:], start=1):
        if xtick.get_position()[0] - sorted_xlabels[i - 1].get_position()[
            0
        ] < 0.005 * len(xvals):
            xtick.set_y(-0.02)


def barplot(ax, df, val, aggregate, aos, soa, sorted_indices=None):
    """
    Plot a barplot of the given value for each partition in the DataFrame, sorted by the given value.
    AoS and SoA partitions are highlighted in red and orange respectively, and the standard deviation
    is plotted as error bars if there are less than 1000 partitions, or as a filled area otherwise.

    :param ax: Matplotlib Axes object
    :param df: DataFrame containing benchmark runtime data
    :param val: Value to plot (e.g., "time" or a hardware performance counter)
    :param aggregate: Aggregation function to use (e.g., "avg", "min", "max", "stddev")
    :param aos: List of AoS partition strings to highlight
    :param soa: List of SoA partition strings to highlight
    :param sorted_indices: Optional precomputed sorted indices to use for sorting the bars
    :return: Sorted indices and sorted container names
    """
    if sorted_indices is None:
        sorted_indices = np.argsort(get_agg_value(df, val, aggregate))
    sorted_vals = np.array(get_agg_value(df, val, aggregate).iloc[sorted_indices])
    sorted_stddev = np.array(get_agg_value(df, val, "stddev").iloc[sorted_indices])
    sorted_containers = (
        df.groupby("container")["time"].mean().index[sorted_indices]
        if "time" in df.columns
        else df["container"].iloc[sorted_indices]
    )
    sorted_containers = np.array(
        [
            (
                c.replace("PartitionedContiguousContainer", "")
                if "Contiguous" in c
                else c.replace("PartitionedContainer", "")
            )
            for c in sorted_containers
        ]
    )

    # Plot barplot of all partitions
    ax.bar(
        sorted_containers,
        sorted_vals,
        yerr=sorted_stddev if len(sorted_vals) < 1000 else None,
        color="#164588",
        ecolor="#EE8F00",
        error_kw={"alpha": 0.7, "zorder": 100},
        width=1,
        label="All Partitions",
    )

    # Overlap with bars for AoS partitions in red to highlight them
    aos_indices = [i for i, c in enumerate(sorted_containers) if c in aos]
    ax.bar(
        sorted_containers[aos_indices],
        sorted_vals[aos_indices],
        yerr=(sorted_stddev[aos_indices] if len(sorted_vals) < 1000 else None),
        color="#C00000",
        width=1,
        label="AoS (Reordered)",
    )

    # Overlap with bars for SoA partitions in orange to highlight them
    soa_indices = [i for i, c in enumerate(sorted_containers) if c in soa]
    bar = ax.bar(
        sorted_containers[soa_indices],
        sorted_vals[soa_indices],
        yerr=(sorted_stddev[soa_indices] if len(sorted_vals) < 1000 else None),
        color="#EE8F00",
        linewidth=0.01,
        width=1,
        label="SoA",
    )

    # If there is a lot of data, we plot the standard deviation as a filled area instead of error bars
    if len(sorted_vals) >= 1000:
        ax.fill_between(
            sorted_containers,
            sorted_vals - sorted_stddev,
            sorted_vals + sorted_stddev,
            color="#EE8F00",
            alpha=0.4,
        )
    ax.add_patch(
        plt.Rectangle(
            (0, 0),
            0,
            0,
            color="#EE8F00",
            alpha=0.4,
            label="Standard Deviation",
        )
    )

    return sorted_indices, sorted_containers


def plot_runtime_barplot(df, output_dir, aggregate):
    """
    Plot runtime barplots for each benchmark and problem size.

    :param df: Dictionary of DataFrames keyed by file name
    :param output_dir: Directory to save the output plots
    """

    for file, df in df.items():
        for benchmark in df["benchmark"].unique():
            fig = plt.figure(figsize=(20, 6 * len(df["problem_size"].unique())))
            plt.suptitle(f"Runtime Barplot of {benchmark}")

            n_members = (
                np.max([int(c) for c in df["container"].iloc[0] if c.isdigit()]) + 1
            )
            aos_containers = [
                "".join([str(p) for p in perm])
                for perm in permutations(range(n_members))
            ]
            soa_containers = [
                "_".join([str(p) for p in perm])
                for perm in permutations(range(n_members))
            ]

            xticks = []
            xticks.append("".join([str(i) for i in range(n_members)]))  # AoS
            xticks.append("_".join([str(i) for i in range(n_members)]))  # SoA

            subplots = []
            for pi, problem_size in enumerate(df["problem_size"].unique()):
                ax = plt.subplot(len(df["problem_size"].unique()), 1, pi + 1)
                subplots.append(ax)
                ax.set_title(f"Problem Size: {problem_size}")

                df_bp = df[
                    (df["benchmark"] == benchmark)
                    & (df["problem_size"] == problem_size)
                ]

                _, sorted_containers = barplot(
                    ax, df_bp, "time", aggregate, aos_containers, soa_containers
                )

                xticks.append(sorted_containers[0])  # Min
                xticks.append(sorted_containers[-1])  # Max

                ax.tick_params(axis="y", which="both")
                ax.set_xlabel("Partitions")

                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))
                if ax.get_ylim()[1] / ax.get_ylim()[0] < 10:
                    ax.yaxis.set_minor_formatter(ticker.StrMethodFormatter("{x:g}"))

                ax.set_ylabel(f'Runtime ({df["time_unit"].iloc[0]})')
                ax.legend()

            # Only show ticks for AoS, SoA, Min, and Max partitions for each problem size to avoid clutter in the x-axis.
            # Adjust x tick labels by lowering them if they are detected to be overlapping.
            for ax in subplots:
                ax.set_xticks(xticks)
                ax.set_xticklabels(
                    xticks, rotation=45, fontsize=5, ha="right", rotation_mode="anchor"
                )
                ax.get_xticklabels()[0].set_color("#C00000")  # AoS
                ax.get_xticklabels()[1].set_color("#EE8F00")  # SoA
                adjust_xticks(ax, sorted_containers)

            plt.tight_layout()
            print(f"Saving {file}_{benchmark}_{aggregate}_runtime_barplot.pdf...")
            plt.savefig(
                os.path.join(
                    output_dir, f"{file}_{benchmark}_{aggregate}_runtime_barplot.pdf"
                )
            )


def plot_runtime_counters_barplot(df, output_dir, aggregate):
    """
    Plot barplots with the runtime at the top and subplots below for the hardware performance counters
    for each benchmark and maximum problem size.

    :param df: Dictionary of DataFrames keyed by file name
    :param output_dir: Directory to save the output plots
    :param aggregate: Aggregation method for the runtime (e.g., "avg" or "median")
    """

    def plot_axes(xticks, sorted_containers, val):
        """
        Helper method to set the x-ticks, y-scale, and y-label for the runtime
        and hardware performance counter barplots.

        :param xticks: List of x-tick labels to set
        :param sorted_containers: List of container names sorted by runtime to adjust the
                                  x-ticks to avoid overlap
        :param val: Value being plotted (e.g., "time" or a hardware performance counter) to set the y-label
        """
        ax.set_xticks(xticks)
        ax.set_xticklabels(
            xticks, rotation=45, fontsize=5, ha="right", rotation_mode="anchor"
        )
        ax.get_xticklabels()[0].set_color("#C00000")  # AoS
        ax.get_xticklabels()[1].set_color("#EE8F00")  # SoA
        adjust_xticks(ax, sorted_containers)

        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))
        if ax.get_ylim()[1] / ax.get_ylim()[0] < 10:
            ax.yaxis.set_minor_formatter(ticker.StrMethodFormatter("{x:g}"))

        ax.set_ylabel(f'Runtime ({df["time_unit"].iloc[0]})' if val == "time" else val)

    #             ax.legend()
    for file, df in df.items():
        problem_size = df["problem_size"].max()

        # Exclude benchmark, container, problem_size, container_byte_size, time_unit, and time columns
        event_cols = df.columns[6:]

        for benchmark in df["benchmark"].unique():
            fig = plt.figure(figsize=(20, 6 * (len(event_cols) + 1)))
            plt.suptitle(f"Performance Barplots for {benchmark}")

            n_members = (
                np.max([int(c) for c in df["container"].iloc[0] if c.isdigit()]) + 1
            )
            aos_containers = [
                "".join([str(p) for p in perm])
                for perm in permutations(range(n_members))
            ]
            soa_containers = [
                "_".join([str(p) for p in perm])
                for perm in permutations(range(n_members))
            ]

            xticks = []
            xticks.append("".join([str(i) for i in range(n_members)]))  # AoS
            xticks.append("_".join([str(i) for i in range(n_members)]))  # SoA

            df_bp = df[
                (df["benchmark"] == benchmark) & (df["problem_size"] == problem_size)
            ]

            # Plot runtime barplot at the top
            ax = plt.subplot(len(event_cols) + 1, 1, 1)
            sorted_indices, sorted_containers = barplot(
                ax, df_bp, "time", aggregate, aos_containers, soa_containers
            )
            xticks.append(sorted_containers[0])  # Min
            xticks.append(sorted_containers[-1])  # Max
            plot_axes(xticks, sorted_containers, "time")

            # Plot barplots for each hardware performance counter, with the same order
            # of partitions as the runtime barplot
            for ei, event in enumerate(event_cols):
                ax = plt.subplot(len(event_cols) + 1, 1, ei + 2)
                barplot(
                    ax,
                    df_bp,
                    event,
                    aggregate,
                    aos_containers,
                    soa_containers,
                    sorted_indices,
                )
                plot_axes(xticks, sorted_containers, event)

            plt.legend(loc="upper left")
            plt.tight_layout()
            print(
                f"Saving {file}_{benchmark}_{aggregate}_runtime_counters_barplot.pdf..."
            )
            plt.savefig(
                os.path.join(
                    output_dir,
                    f"{file}_{benchmark}_{aggregate}_runtime_counters_barplot.pdf",
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
    parser.add_argument(
        "-a",
        "--aggregate",
        type=str,
        help="Which metric to aggregate over (min, max, avg)",
        choices=["min", "max", "avg"],
        default="avg",
    )
    args = parser.parse_args()

    data = read_data(args.input)
    plot_runtime_histogram(data, args.output, args.aggregate)
    plot_runtime_barplot(data, args.output, args.aggregate)
    plot_runtime_counters_barplot(data, args.output, args.aggregate)
