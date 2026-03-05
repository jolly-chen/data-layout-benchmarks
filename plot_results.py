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

from sklearn import metrics

###########
# HELPERS #
###########


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


def is_overlapping_1D(xmin1, xmax1, xmin2, xmax2):
    """
    Check if two 1D intervals overlap.

    :param xmin1: Minimum x of first interval
    :param xmax1: Maximum x of first interval
    :param xmin2: Minimum x of second interval
    :param xmax2: Maximum x of second interval
    :return: True if the intervals overlap, False otherwise
    """
    return xmax1 >= xmin2 and xmax2 >= xmin1


def adjust_annotations(ax, annotations):
    """
    Adjust any annotations that overlap by stacking them vertically and shifting them
    to the right so the arrows remain visible.

    :param ax: Matplotlib Axes object
    :param annotations: List of Matplotlib Annotation objects
    """

    def update_annotation_positions(idx, new_x, new_y, zorder):
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
                ic,
                ax_xmin + 0.5 * (coord[2] - coord[0]),
                ann.xyann[1],
                ann.zorder,
            )

    # Sort annotations by ymin and then by xmin
    sorted_idx = np.lexsort((coords[:, 0], coords[:, 1]))

    for ii, i in enumerate(sorted_idx[1:], start=1):
        for j in sorted_idx[:ii]:
            i_textbox_ymin = annotations[i].xyann[1]
            i_textbox_xmin, _, i_textbox_xmax, i_textbox_ymax = coords[i]
            j_textbox_xmin, _, j_textbox_xmax, j_textbox_ymax = coords[j]

            # If the current box overlaps horizontally with a box below and it is not completely above it,
            # we move the current box upward so they are stacked vertically.
            if (
                is_overlapping_1D(
                    i_textbox_xmin, i_textbox_xmax, j_textbox_xmin, j_textbox_xmax
                )
                and i_textbox_ymin < j_textbox_ymax
            ):
                # Move the current box upward:
                update_annotation_positions(
                    i,
                    annotations[i].xyann[0],
                    j_textbox_ymax + 2 * (i_textbox_ymax - i_textbox_ymin),
                    annotations[i].zorder,
                )

    # From top to bottom, increase the zorder so they are drawn latter
    sorted_idx = np.lexsort((coords[:, 0], -coords[:, 1]))
    for i in sorted_idx:
        update_annotation_positions(
            i,
            annotations[i].xyann[0],
            annotations[i].xyann[1],
            len(annotations) + i,
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
        fontsize=10,
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
    container_base_name = (
        "PartitionedContainerContiguous"
        if "PartitionedContainerContiguous" in df["container"].iloc[0]
        else "PartitionedContainer"
    )

    # AoS
    aos_vals = get_agg_value(
        df[df["container"].isin([f"{container_base_name}{c}" for c in aos_containers])],
        "time",
        aggregate,
    )

    heights, edges, _ = ax.hist(
        aos_vals,
        bins=edges,
        color=aos_color,
        align="mid",
        label="AoS (Reordered)",
    )

    if not aos_vals.empty:
        annotate_partition(
            ax,
            aos_vals.iloc[0],
            heights[int(np.digitize(aos_vals.iloc[0], edges)) - 1],
            f"AoS: {aos_vals.iloc[0]:.2f}\n({aos_containers[0]})",
            aos_color,
            annotations,
        )

    # SoA
    soa_vals = get_agg_value(
        df[df["container"].isin([f"{container_base_name}{c}" for c in soa_containers])],
        "time",
        aggregate,
    )

    heights, edges, _ = ax.hist(
        soa_vals,
        bins=edges,
        color=soa_color,
        align="mid",
        label="SoA",
    )

    if not soa_vals.empty:
        annotate_partition(
            ax,
            soa_vals.iloc[0],
            heights[int(np.digitize(soa_vals.iloc[0], edges)) - 1],
            f"SoA: {soa_vals.iloc[0]:.2f}\n({soa_containers[0]})",
            soa_color,
            annotations,
        )


def plot_runtime_histogram(file, df, output_dir, aggregate):
    """
    Plot runtime histograms for each benchmark and problem size.

    :param df: Dictionary of DataFrames keyed by file name
    :param output_dir: Directory to save the output plots
    """
    fig = plt.figure(figsize=(7 * len(df["problem_size"].unique()), 5))

    for benchmark in df["benchmark"].unique():
        fig.suptitle(f"Runtime Distribution of {benchmark}", fontsize=20)

        for pi, problem_size in enumerate(df["problem_size"].unique()):
            ax = fig.add_subplot(1, len(df["problem_size"].unique()), pi + 1)
            ax.set_title(f"Problem Size: {problem_size}", fontsize=18)
            annotations = []

            df_bp = df[
                (df["benchmark"] == benchmark) & (df["problem_size"] == problem_size)
            ]
            heights, edges, _ = plt.hist(
                get_agg_value(df_bp, "time", aggregate),
                bins="auto",
                color=other_color,
                align="mid",
                label="Other Layouts",
            )

            annotate_minmax(ax, df_bp, heights, annotations, aggregate)
            annotate_common(ax, df_bp, edges, annotations, aggregate)

            ax.set_ylabel("Frequency", fontsize=14)
            ax.set_yscale("symlog")
            y_minor = matplotlib.ticker.LogLocator(
                base=10.0, subs=np.arange(1, 10) * 0.1, numticks=10
            )
            ax.yaxis.set_minor_locator(y_minor)
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.grid(axis="y", alpha=0.75)

            ax.set_xlabel(f'Runtime ({df["time_unit"].iloc[0]})', fontsize=14)
            ax.legend(loc="best", bbox_to_anchor=(0, 0.7, 1, 0.3))
            adjust_annotations(ax, annotations)

        fig.tight_layout()
        print(f"Saving {file}_{benchmark}_{aggregate}_runtime_histogram.pdf...")
        fig.savefig(
            os.path.join(
                output_dir, f"{file}_{benchmark}_{aggregate}_runtime_histogram.pdf"
            )
        )

        fig.clf()  # Clear the figure to free memory for the next plot


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
        # If pixel difference is too small, we consider the labels to be overlapping and we lower the current label
        if (
            ax.transData.transform(xtick.get_position())[0]
            - ax.transData.transform(sorted_xlabels[i - 1].get_position())[0]
            < 20
        ):
            xtick.set_y(-0.045)


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

    error_bars_treshold = 10
    # Plot barplot of all partitions
    ax.bar(
        sorted_containers,
        sorted_vals,
        yerr=sorted_stddev if len(sorted_vals) < error_bars_treshold else None,
        color=other_color,
        ecolor=stddev_color,
        error_kw={"alpha": 0.3, "zorder": 100},
        width=1,
        label="Other Layouts",
    )

    # Overlap with bars for AoS partitions in red to highlight them
    aos_indices = [i for i, c in enumerate(sorted_containers) if c in aos]
    ax.bar(
        sorted_containers[aos_indices],
        sorted_vals[aos_indices],
        color=aos_color,
        width=1,
        label="AoS (Reordered)",
    )

    # Overlap with bars for SoA partitions in orange to highlight them
    soa_indices = [i for i, c in enumerate(sorted_containers) if c in soa]
    ax.bar(
        sorted_containers[soa_indices],
        sorted_vals[soa_indices],
        color=soa_color,
        linewidth=0.01,
        width=1,
        label="SoA",
    )

    # If there is a lot of data, we plot the standard deviation as a filled area instead of error bars
    if len(sorted_vals) >= error_bars_treshold:
        ax.fill_between(
            sorted_containers,
            sorted_vals - sorted_stddev,
            sorted_vals + sorted_stddev,
            color=stddev_color,
            alpha=0.3,
        )
    ax.add_patch(
        plt.Rectangle(
            (0, 0),
            0,
            0,
            color=stddev_color,
            alpha=0.3,
            label="Standard Deviation",
        )
    )

    return sorted_indices, sorted_containers


def map_metric_name(metric, time_unit):
    """
    Map a hardware performance counter name to a more readable format.

    :param metric: Hardware performance counter name
    :return: Mapped metric name
    """
    if (
        metric
        == "ANY_DATA_CACHE_FILLS_FROM_SYSTEM:LCL_L2:LOCAL_CCX:NEAR_CACHE_NEAR_FAR:DRAM_IO_NEAR:FAR_CACHE_NEAR_FAR:DRAM_IO_FAR:ALT_MEM_NEAR_FAR"
    ):
        return "Any L1 Data Cache Fills"
    elif metric == "PAPI_L1_DCA":
        return "L1 Data Cache Accesses"
    elif metric == "PAPI_TOT_CYC":
        return "Total Unhalted Cycles"
    elif metric == "PAPI_TOT_INS":
        return "Total Number of Retired Instructions"
    elif metric == "PAPI_VEC_INS":
        return "Total Number of Vector Instructions"
    elif metric == "time":
        return f"Runtime ({time_unit})"
    else:
        return metric


def plot_barplot(file, df, output_dir, aggregate, metrics):
    """
    Plot barplots for each benchmark and problem size, for the given metrics.
    The bars are sorted by the first metric.

    :param df: Dictionary of DataFrames keyed by file name
    :param output_dir: Directory to save the output plots
    :param aggregate: Aggregation method for the runtime (e.g., "avg" or "median")
    :param metrics: List of metrics to plot (e.g., ["time", "PAPI_TOT_INS"])
    """

    def plot_axes(ax, sorted_containers, metric):
        """
        Helper method to set the x-ticks, y-scale, and y-label for the runtime
        and hardware performance counter barplots.

        :param xticks: List of x-tick labels to set
        :param sorted_containers: List of container names sorted by runtime to adjust the
                                  x-ticks to avoid overlap
        :param metric: Metric being plotted (e.g., "time" or a hardware performance counter) to set the y-label
        """
        step = 1500
        xticks = np.concatenate(
            [
                aos_containers[:1],
                soa_containers[:1],
                sorted_containers[:1],
                sorted_containers[-1:],
                sorted_containers[
                    np.arange(step, len(sorted_containers), step).astype(int)
                ],
            ]
        )
        ax.set_xticks(xticks)
        ax.set_xticklabels(
            xticks, rotation=45, fontsize=8, ha="right", rotation_mode="anchor"
        )
        ax.get_xticklabels()[0].set_color(aos_color)  # AoS
        ax.get_xticklabels()[1].set_color(soa_color)  # SoA
        adjust_xticks(ax, sorted_containers)
        ax.set_xlabel("Layouts", fontsize=14)

        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))
        if ax.get_ylim()[1] / ax.get_ylim()[0] < 10:
            ax.yaxis.set_minor_formatter(ticker.StrMethodFormatter("{x:g}"))
        ax.set_ylabel(
            map_metric_name(metric, time_unit=df["time_unit"].iloc[0]), fontsize=14
        )
        ax.legend(loc="best", bbox_to_anchor=(0, 0.5, 1, 0.5))

    fig = plt.figure(figsize=(9, 5 * len(metrics)))
    problem_size = df["problem_size"].max()

    for ib, benchmark in enumerate(df["benchmark"].unique()):
        fig.suptitle(f"{benchmark}", fontsize=20)

        df_bp = df[
            (df["benchmark"] == benchmark) & (df["problem_size"] == problem_size)
        ]

        # Plot the first metric
        ax = fig.add_subplot(len(metrics), 1, 1)
        sorted_indices, sorted_containers = barplot(
            ax, df_bp, metrics[0], aggregate, aos_containers, soa_containers
        )
        plot_axes(ax, sorted_containers, metrics[0])

        # Plot barplots for all other metrics, with the same order of containers as the first metric
        for mi, metric in enumerate(metrics[1:], start=1):
            ax = fig.add_subplot(len(metrics), 1, mi + 1)
            barplot(
                ax,
                df_bp,
                metric,
                aggregate,
                aos_containers,
                soa_containers,
                sorted_indices,
            )
            plot_axes(ax, sorted_containers, metric)

        fig.tight_layout()
        print(
            f"Saving {file}_{benchmark}_{aggregate}_barplot_{'_'.join([m.split(':')[0] for m in metrics])}.pdf..."
        )
        fig.savefig(
            os.path.join(
                output_dir,
                f"{file}_{benchmark}_{aggregate}_barplot_{'_'.join([m.split(':')[0] for m in metrics])}.pdf",
            )
        )
        fig.clf()  # Clear the figure to free memory for the next plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output", type=str, help="Set output directory", default="."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="<Required> Set input file",
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

    df = pd.read_csv(args.input)
    n_members = np.max([int(c) for c in df["container"].iloc[0] if c.isdigit()]) + 1
    aos_containers = [
        "".join([str(p) for p in perm]) for perm in permutations(range(n_members))
    ]
    soa_containers = [
        "_".join([str(p) for p in perm]) for perm in permutations(range(n_members))
    ]

    aos_color = "#C00000"
    soa_color = "#EE8F00"
    other_color = "#164588"
    stddev_color = "#AB8AD1"

    plot_runtime_histogram(args.input, df, args.output, args.aggregate)
    plot_barplot(args.input, df, args.output, args.aggregate, ["time", "ANY_DATA_CACHE_FILLS_FROM_SYSTEM:LCL_L2:LOCAL_CCX:NEAR_CACHE_NEAR_FAR:DRAM_IO_NEAR:FAR_CACHE_NEAR_FAR:DRAM_IO_FAR:ALT_MEM_NEAR_FAR", "PAPI_L1_DCA"])
    plot_barplot(args.input, df, args.output, args.aggregate, ["PAPI_TOT_INS", "time"])
