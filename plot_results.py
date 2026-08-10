# !/usr/bin/env python3
#
# Script for plotting the results gathered using main.cpp
#

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import argparse
import os
import matplotlib
import matplotlib.ticker as ticker

from itertools import permutations
from pathlib import Path

###########
# HELPERS #
###########

def savefig(fig, output_file):
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")

def find_bin(edges, val):
    """
    Find the index of the bin corresponding to a given value.

    :param edges: Edges of the histogram bins
    :param val: Value to find the bin for
    :return: Index of the bin corresponding to the value
    """
    return np.clip(np.digitize(val, edges) - 1, 0, len(edges) - 2)

def map_benchmark_name(benchmark):
    """Map a benchmark name to the shorthand name in the paper."""
    if benchmark == "InvariantMassSequential":
        return "SIM"
    elif benchmark == "InvariantMassRandom":
        return "RIM"

    return benchmark

def map_arch_name(arch):
    """Map an architecture name to a short name for the paper."""
    if arch == "AMD Zen 2":
        return "Zen2"
    elif arch == "AMD Zen 4":
        return "Zen4"
    elif arch == "Intel Haswell":
        return "HSW"
    elif arch == "Intel Skylake":
        return "SKL"
    else:
        return arch

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

def get_val_from_partition(df, partition, val, aggregate):
    return get_agg_value(df[df["container"].str.contains(partition)], val, aggregate).iloc[0]

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
        annotations[idx].set(position=(new_x, new_y), zorder=100 + zorder)

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
                    annotations[i].xyann[0] + 0.05 * (j_textbox_xmax - j_textbox_xmin),
                    j_textbox_ymax + 2.6 * (i_textbox_ymax - i_textbox_ymin),
                    annotations[i].zorder,
                )

    # From top to bottom, increase the zorder so they are drawn later
    sorted_idx = np.argsort(-coords[:, 1])
    for ii, i in enumerate(sorted_idx):
        update_annotation_positions(
            i,
            annotations[i].xyann[0],
            annotations[i].xyann[1],
            len(annotations) + ii,
        )
        annotations[i].draggable()


def used_alone(partition, used):
    """Are the used members alone in their subsets?"""
    p_list = partition.split("_")
    # Find subset each used member belongs to
    used_idx = [i for u in used for i, p in enumerate(p_list) if u in p]
    return all(len(p_list[i]) == 1 for i in used_idx)


def used_not_mixed_with_unused(partition, used):
    """Are the used members isolated from the unused members?"""
    p_list = partition.split("_")

    # Find subset each used member belongs to
    used_idx = [i for u in used for i, p in enumerate(p_list) if u in p]

    # For each subset that contains used members, check if it only contains used members
    for ui in used_idx:
        n_used = sum(1 for member in p_list[ui] if member in used)
        if n_used != len(p_list[ui]):
            return False

    return True


def used_together(partition, used):
    """Are the used members all in the same partition?"""
    p_list = partition.split("_")
    used_idx = [i for u in used for i, p in enumerate(p_list) if u in p]
    return all(used_idx[0] == pi for pi in used_idx)


#############
# HISTOGRAM #
#############

def annotate_point(ax, x, y, text, color, annotations):
    """
    Annotate a bar in the histogram.

    :param ax: Matplotlib Axes object
    :param x: x-coordinate of the point to annotate
    :param y: y-coordinate of the point to annotate
    :param text: Text of the annotation
    :param color: Color of the annotation
    :param annotations: List to store the created annotations
    """
    ax_pixel_height = ax.transData.transform((0, ax.get_ylim()[1]))[1] - ax.transData.transform((0, ax.get_ylim()[0]))[1]
    pixel_y = ax.transData.transform((0, y))[1] + 0.15 * ax_pixel_height
    y_text = ax.transData.inverted().transform((0, pixel_y))[1]

    a = ax.annotate(
        text,
        xy=(x, y),
        xytext=(x, y_text),
        arrowprops=dict(arrowstyle="fancy", facecolor=color, alpha=0.8),
        bbox=dict(boxstyle="square", fc="w", alpha=0.8),
        fontsize=10,
        color=color,
        horizontalalignment="center",
    )

    annotations.append(a)


def annotate_minmax(ax, df_bp, heights, annotations, aggregate, verbose=True):
    """
    Annotate the minimum and maximum average runtimes in the histogram
    with their corresponding partition.

    :param ax: Matplotlib Axes object
    :param df_bp: DataFrame containing benchmark runtime data
    :param heights: Heights of the histogram bars
    :param annotations: List to store the created annotations
    :param aggregate: Metric to aggregate over (min, max, avg)
    :param verbose: Whether to include time and partition information in the annotation text
    """
    max_val = get_agg_value(df_bp, "time", aggregate).max()
    max_partition = get_partition_from_val(df_bp, max_val, aggregate)
    annotate_point(
        ax,
        max_val,
        heights[-1],
        (
            f"Max: {max_val:.2f}\n({''.join(filter(lambda x: not x.isalpha(), max_partition))})"
            if verbose
            else f"Max"
        ),
        "k",
        annotations,
    )

    min_val = get_agg_value(df_bp, "time", aggregate).min()
    min_partition = get_partition_from_val(df_bp, min_val, aggregate)
    annotate_point(
        ax,
        min_val,
        heights[0],
        (
            f"Min: {min_val:.2f}\n({''.join(filter(lambda x: not x.isalpha(), min_partition))})"
            if verbose
            else f"Min"
        ),
        "k",
        annotations,
    )


def annotate_common(ax, df, edges, annotations, aggregate, verbose=True):
    """
    Annotate the average runtimes of common partitioning schemes (AoS and SoA) in the histogram.
    Parts of the histogram that include AoS layouts with reordered data members are highlighted.

    :param ax: Matplotlib Axes object
    :param df: DataFrame containing benchmark runtime data
    :param edges: Edges of the histogram bins
    :param color: Color of the annotation
    :param annotations: List to store the created annotations
    :param aggregate: Metric to aggregate over (min, max, avg)
    :param verbose: Whether to include time and partition information in the annotation text
    """
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
        annotate_point(
            ax,
            aos_vals.iloc[0],
            heights[find_bin(edges, aos_vals.iloc[0])],
            (
                f"AoS: {aos_vals.iloc[0]:.2f}\n({aos_containers[0]})"
                if verbose
                else f"AoS"
            ),
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
        annotate_point(
            ax,
            soa_vals.iloc[0],
            heights[find_bin(edges, soa_vals.iloc[0])],
            (
                f"SoA: {soa_vals.iloc[0]:.2f}\n({soa_containers[0]})"
                if verbose
                else f"SoA"
            ),
            soa_color,
            annotations,
        )


def plot_histogram(ax, df, problem_size, benchmark, aggregate, annotations, annotate, verbose=True):
    """
    Plot a histogram of the given benchmark and problem size, with the average runtime of
    each partition on the x-axis and the frequency on the y-axis.

    :param ax: Matplotlib Axes object
    :param df: DataFrame containing benchmark runtime data
    :param problem_size: Problem size to filter the DataFrame by
    :param benchmark: Benchmark to filter the DataFrame by
    :param aggregate: Metric to aggregate over (min, max, avg)
    :param annotations: List to store the created annotations
    :param annotate: Type of annotations to include ("default" for min/max and common schemes,
                     "guidelines" for guidelines annotations)
    :param verbose: Whether to include time and partition information in the annotation text
    """

    df_bp = df[(df["benchmark"] == benchmark) & (df["problem_size"] == problem_size)]
    time_vals = get_agg_value(df_bp, "time", aggregate)
    heights, edges, _ = ax.hist(
        time_vals,
        bins="auto",
        color=other_color,
        align="mid",
        label="Other Layouts",
    )

    ax.set_ylabel("Frequency", fontsize=16)
    ax.set_yscale("symlog")
    y_minor = matplotlib.ticker.LogLocator(
        base=10.0, subs=np.arange(1, 10) * 0.1, numticks=10
    )
    ax.yaxis.set_minor_locator(y_minor)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(axis="y", alpha=0.75)

    ax.set_xlabel(f'Runtime ({df["time_unit"].iloc[0]})', fontsize=16)

    if annotate == "default":
        annotate_common(ax, df_bp, edges, annotations, aggregate, verbose)
    elif annotate == "guidelines":
        ut_containers = df_bp[
            df_bp["container"].isin(
                [
                    c
                    for c in df_bp["container"]
                    if used_together(
                        c.replace(container_base_name, ""), ["1", "2", "3", "4"]
                    )
                ]
            )
        ]
        ut_vals = get_agg_value(
            ut_containers,
            "time",
            aggregate,
        )
        ax.hist(
            ut_vals.values,
            bins=edges,
            color=ut_color,
            align="mid",
            label="Used Together",
        )

        unm_containers = df_bp[
            df_bp["container"].isin(
                [
                    c
                    for c in df_bp["container"]
                    if used_not_mixed_with_unused(
                        c.replace(container_base_name, ""), ["1", "2", "3", "4"]
                    )
                ]
            )
        ]
        unm_vals = get_agg_value(
            unm_containers,
            "time",
            aggregate,
        )
        ax.hist(
            unm_vals.values,
            bins=edges,
            color=unm_color,
            align="mid",
            label="Used not mixed with Unused",
        )

        median_unm = unm_vals.median()
        max_95_percentile_unm = np.quantile(np.sort(unm_vals), 0.94)
        median_partition_unm = find_bin(np.sort(time_vals), median_unm)
        max95_partition_unm = find_bin(np.sort(time_vals), max_95_percentile_unm)

        print(
            f"{benchmark} - Used not mixed with Unused:\n"
            f"\tmax95 ({max_95_percentile_unm}, {max95_partition_unm}) {max95_partition_unm / len(time_vals) * 100:.2f}%\n"
            f"\tmedian ({median_unm}, {median_partition_unm}) {median_partition_unm / len(time_vals) * 100:.2f}%"
        )

        ua_vals = get_agg_value(
            df_bp[
                df_bp["container"].isin(
                    [
                        c
                        for c in df_bp["container"]
                        if used_alone(
                            c.replace(container_base_name, ""), ["1", "2", "3", "4"]
                        )
                    ]
                )
            ],
            "time",
            aggregate,
        )
        ax.hist(
            ua_vals.values,
            bins=edges,
            color=ua_color,
            align="mid",
            label="SoA of Used Members",
        )
        max_95_percentile_ua = np.quantile(np.sort(ua_vals), 0.95)
        median_ua = ua_vals.median()
        max95_partition_ua = find_bin(np.sort(time_vals), max_95_percentile_ua)
        median_partition_ua = find_bin(np.sort(time_vals), median_ua)

    ax.legend(loc="best", fontsize=12, labelspacing=0.2, handletextpad=0.4, columnspacing=0.5)
    return heights, edges


def plot_runtime_histogram_max_psize_all_archs(
    files, dfs, output_dir, aggregate, annotate
):
    """
    Plot runtime histograms for each benchmark and problem size.

    :param df: Dictionary of DataFrames keyed by file name
    :param output_dir: Directory to save the output plots
    :param aggregate: Metric to aggregate over (min, max, avg)
    :param annotate: Type of annotations to include ("default" for min/max and common schemes,
                     "guidelines" for guidelines annotations)
    """
    fig = plt.figure(figsize=(12, 2.5 * len(files)))
    arch_names = [f"{Path(file).stem.replace('_', ' ')}" for file in files]

    for row, (file, df) in enumerate(zip(files, dfs)):
        problem_size = df["problem_size"].max()

        for ib, benchmark in enumerate(df["benchmark"].unique()):
            ax = fig.add_subplot(
                len(files),
                len(df["benchmark"].unique()),
                row * len(df["benchmark"].unique()) + ib + 1,
            )

            if row == 0:
                ax.set_title(r"\textbf{\LARGE{" + benchmark + r"}}", fontsize=16)

            plot_histogram(ax, df, problem_size, benchmark, aggregate, [], annotate)
            if ib == 0:
                arch_name = arch_names[row]
                ax.set_ylabel(
                    r"\textbf{\LARGE{" + arch_name + r"}}" + "\nFrequency", fontsize=16
                )
            ax.legend().set_visible(False)

    fig.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.03), handletextpad=0.4, columnspacing=0.5,
               ncol=len(labels), fontsize=12, markerscale=5)
    fig.subplots_adjust(hspace=0.3)

    output_file = os.path.join(
        output_dir,
        f"allarchs_maxpsize_{aggregate}_runtime_histogram_{annotate}.{fmt}",
    )
    print(f"Saving {output_file}...")
    savefig(fig, output_file)


def plot_runtime_histogram_all_minmax(files, dfs, output_dir, aggregate):
    def get_rank(val, sorted_vals):
        return int(find_bin(sorted_vals, val)) / len(sorted_vals) * 100

    fig = plt.figure(figsize=(14, 7))
    benchmarks = dfs[0]["benchmark"].unique()
    axs = fig.subplots(len(benchmarks), 2).flatten()
    arch_names = [f"{map_arch_name(Path(file).stem.replace('_', ' '))}" for file in files]
    minmax = {}

    # Compute the min and max partitions for each benchmark and problem size across all architectures
    # to annotate them in the histogram later
    for di, df in enumerate(dfs):
        for benchmark in benchmarks:
            for pi, ps in enumerate(df["problem_size"].unique()):
                df_bp = df[(df["benchmark"] == benchmark) & (df["problem_size"] == ps)]
                time_vals = get_agg_value(df_bp, "time", aggregate)
                min_partition = get_partition_from_val(df_bp, time_vals.min(), aggregate)
                max_partition = get_partition_from_val(df_bp, time_vals.max(), aggregate)
                if (benchmark, pi) not in minmax:
                    minmax[(benchmark, pi)] = [(arch_names[di], min_partition, max_partition)]
                else:
                    minmax[(benchmark, pi)].append((arch_names[di], min_partition, max_partition))

    # Plot the histogram for each benchmark and problem size of the first architecture,
    # and annotate the min and max partitions for each architecture
    df = dfs[0]
    for bi, benchmark in enumerate(benchmarks):
        for pi, problem_size in enumerate(df["problem_size"].unique()):
            ax = axs[bi * 2 + pi]
            annotations = []
            if bi == 0: ax.set_title(r"\textbf{\LARGE{" + ('Small' if pi == 0 else 'Large') + '} Problem Size', fontsize=16)
            heights, edges = plot_histogram(
                ax, df, problem_size, benchmark, aggregate, annotations, annotate="default", verbose=False
            )

            if pi == 0:
                ax.set_ylabel(
                    r"\textbf{\LARGE{" + benchmark + r"}}" + "\nFrequency", fontsize=16
                )

            df_bp = df[(df["benchmark"] == benchmark) & (df["problem_size"] == problem_size)]
            time_vals = np.sort(get_agg_value(df_bp, "time", aggregate))
            for item in minmax[(benchmark, pi)]:
                min_val = get_agg_value(df_bp[df_bp["container"] == item[1]], "time", aggregate).iloc[0]
                max_val = get_agg_value(df_bp[df_bp["container"] == item[2]], "time", aggregate).iloc[0]
                print(f"{item[0]} {benchmark}  {problem_size}:")
                print(f"\t min {min_val:.5f} - {item[1]} - {get_rank(min_val, time_vals):.3f}%")
                print(f"\tmax {max_val:.5f} - {item[2]} - {get_rank(max_val, time_vals):.3f}%")
                imin = min(find_bin(edges, min_val), len(heights) - 1)
                imax = min(find_bin(edges, max_val), len(heights) - 1)
                annotate_point(ax, min_val, heights[imin], f"{item[0]} Min", "k", annotations)
                annotate_point(ax, max_val, heights[imax], f"{item[0]} Max", "k", annotations)
            ax.set_ylim(bottom=0, top= ax.get_ylim()[1] * 10**2)
            adjust_annotations(ax, annotations)

    fig.tight_layout()
    output_file = os.path.join(
        output_dir,
        f"{Path(files[0]).stem}_{aggregate}_runtime_histogram_allminmax.{fmt}",
    )
    print(f"Saving {output_file}...")
    # plt.show() # Uncomment to display the plot to adjust the annotation positions manually if needed
    savefig(fig, output_file)
    fig.clf()  # Clear the figure to free memory for the next plot


############
# scatters#
############


def adjust_xticks(ax):
    """
    Adjust the xtick labels by lowering them if they are detected to be overlapping.

    :param ax: Matplotlib Axes object
    """
    sorted_xlabels = sorted(ax.get_xticklabels(), key=lambda x: x.get_position()[0])
    for i, xtick in enumerate(sorted_xlabels[1:], start=1):
        # If pixel difference is too small, we consider the labels to be overlapping and we lower the current label
        if (
            ax.transData.transform(xtick.get_position())[0]
            - ax.transData.transform(sorted_xlabels[i - 1].get_position())[0]
            < 10
        ):
            xtick.set_y(-0.08)


def scatter(ax, df, val, aggregate, aos, soa, sorted_indices=None):
    """
    Plot a scatter of the given value for each partition in the DataFrame, sorted by the given value.
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

    err_bars = val != "time" or len(sorted_vals) >= 1000

    # Plot scatter of all partitions
    ax.errorbar(
        sorted_containers,
        sorted_vals,
        yerr=sorted_stddev if err_bars else None,
        color=other_color,
        ecolor=stddev_color,
        ms=2,
        fmt="o",
        label="Other Layouts",
        rasterized=True,
    )

    # Overlap with bars for AoS partitions in red to highlight them
    aos_indices = [i for i, c in enumerate(sorted_containers) if c in aos]
    ax.scatter(
        sorted_containers[aos_indices],
        sorted_vals[aos_indices],
        color=aos_color,
        s=2,
        label="AoS (Reordered)",
        zorder=20,
        rasterized=True,
    )

    # Overlap with bars for SoA partitions in orange to highlight them
    soa_indices = [i for i, c in enumerate(sorted_containers) if c in soa]
    ax.scatter(
        sorted_containers[soa_indices],
        sorted_vals[soa_indices],
        color=soa_color,
        s=2,
        label="SoA",
        zorder=20,
        rasterized=True
    )

    # If there is a lot of data, we plot the standard deviation as a filled area instead of error bars
    if not err_bars:
        ax.fill_between(
            sorted_containers,
            sorted_vals - sorted_stddev,
            sorted_vals + sorted_stddev,
            facecolor="none",
            edgecolor=stddev_color,
            linewidth=0.01,
            alpha=0.3,
        )
        ax.add_patch(
            plt.Rectangle(
                (0, 0),
                0,
                0,
                alpha=0.3,
                edgecolor=stddev_color,
                facecolor="none",
                label="Standard Deviation",
            )
        )

    return sorted_indices, sorted_containers


def map_metric_name(metric, time_unit):
    """
    Map a hardware performance counter name to a more readable format.

    :param  metric: Metric name to map
    :param time_unit: Time unit to include in the label if the metric is "time"
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
        return "Unhalted Cycles"
    elif metric == "PAPI_TOT_INS":
        return "Retired Instructions"
    elif metric == "PAPI_VEC_INS":
        return "Vector Instructions"
    elif metric == "time":
        return f"Runtime ({time_unit})"
    else:
        return metric


def plot_scatter(file, df, output_dir, aggregate, metrics, sort_by=None, outside_legend=False):
    """
    Plot a scatter plot for each benchmark and problem size, for the given metrics.
    The bars are sorted by the first metric.

    :param file: File name corresponding to the DataFrame, used for naming the output file
    :param df: DataFrame containing benchmark runtime data
    :param output_dir: Directory to save the output plots
    :param aggregate: Aggregation function to use (e.g., "avg", "min", "max", "stddev")
    :param metrics: List of metrics to plot (e.g., ["time", "PAPI_TOT_INS"])
    :param sort_by: Metric to sort the containers by (e.g., "time" or a hardware performance counter)
    :param outside_legend: Whether to place the legend outside the plot
    """

    def plot_axes(ax, df, sorted_containers, metric):
        """
        Helper method to set the x-ticks, y-scale, and y-label for the runtime
        and hardware performance counter scatters.

        :param ax: Matplotlib Axes object
        :param df: DataFrame containing benchmark runtime data
        :param sorted_containers: List of container names sorted by the first metric
        :param metric: Metric being plotted (e.g., "time" or "PAPI_TOT_INS")
        """
        step = 1500
        xticks = np.concatenate(
            [
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
        adjust_xticks(ax)
        ax.set_xlabel("Layouts", fontsize=16)

        if ax.get_ylim()[0] >= 1e4 and metric != "time":
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.2)
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g}"))
        if ax.get_ylim()[1] / (ax.get_ylim()[0] + .00001) < 10:
            ax.yaxis.set_minor_formatter(ticker.StrMethodFormatter("{x:g}"))
        ax.set_ylabel(
            map_metric_name(metric, time_unit=df["time_unit"].iloc[0]), fontsize=16
        )

        annotations = []
        annotate_point(ax, aos_containers[0], get_val_from_partition(df, aos_containers[0], metric, aggregate),
                       "AoS", aos_color, annotations)
        annotate_point(ax, soa_containers[0], get_val_from_partition(df, soa_containers[0], metric, aggregate),
                       "SoA", soa_color, annotations)
        adjust_annotations(ax, annotations)

        if not outside_legend:
            ax.legend(loc="best", labelspacing=0.2, handletextpad=0.4, columnspacing=0.5, fontsize=12, markerscale=5)

    fig = plt.figure(figsize=(5 * len(df["benchmark"].unique()), 2 * len(metrics) + 1))
    axs = fig.subplots(len(metrics), len(df["benchmark"].unique())).flatten()
    problem_size = df["problem_size"].max()

    for ib, benchmark in enumerate(df["benchmark"].unique()):
        ax = axs[ib]
        ax.set_title(r"\textbf{\LARGE{" + benchmark + "}}", fontsize=16)

        df_bp = df[
            (df["benchmark"] == benchmark) & (df["problem_size"] == problem_size)
        ]

        # Plot the first metric
        if not sort_by:
            sorted_indices, sorted_containers = scatter(
                ax, df_bp, metrics[0], aggregate, aos_containers, soa_containers
            )
        else:
            sorted_indices = np.argsort(get_agg_value(df_bp, sort_by, aggregate))
            _, sorted_containers = scatter(
                ax,
                df_bp,
                metrics[0],
                aggregate,
                aos_containers,
                soa_containers,
                sorted_indices,
            )
        plot_axes(ax, df_bp, sorted_containers, metrics[0])
        if len(metrics) > 1: ax.xaxis.set_visible(False)

        # Plot scatters for all other metrics, with the same order of containers as the first metric
        for mi, metric in enumerate(metrics[1:], start=1):
            ax = axs[ib + mi * len(df["benchmark"].unique())]
            scatter(
                ax,
                df_bp,
                metric,
                aggregate,
                aos_containers,
                soa_containers,
                sorted_indices,
            )
            plot_axes(ax, df_bp, sorted_containers, metric)

            # Only show x-tick labels for the bottom row of plots
            if mi != len(metrics) - 1:
                ax.xaxis.set_visible(False)

    if outside_legend:
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.06), handletextpad=0.4, columnspacing=0.5,
                   ncol=len(labels), fontsize=12, markerscale=5)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.01)
    output_file = os.path.join(
        output_dir,
        f"{Path(file).stem}_{aggregate}_scatter_{'_'.join([m.split(':')[0] for m in metrics])}.{fmt}",
    )
    print(f"Saving {output_file}...")
    savefig(fig, output_file)
    fig.clf()  # Clear the figure to free memory for the next plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, help="Set output directory")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="<Required> Set input file",
        nargs="+",
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

    if not args.output:
        args.output = os.path.dirname(args.input[0])

    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "TeX Gyre Termes"]

    aos_color = "#C00000"
    soa_color = "#EE8F00"
    other_color = "#D5E4F0"
    stddev_color = "#A3A3A3DC"
    ut_color = "#85A0D6"
    unm_color = "#9A2CC9"
    ua_color = "#FFE11F"

    dpi = 1200
    fmt = "jpg"

    ########
    dfs = [ pd.read_csv(input) for input in args.input ]
    for df in dfs:
        df["benchmark"] = df["benchmark"].apply(lambda x: map_benchmark_name(x))

    for input, df in zip(args.input, dfs):
        container_base_name = (
            "PartitionedContainerContiguous"
            if "PartitionedContainerContiguous" in df["container"].iloc[0]
            else "PartitionedContainer"
        )

        n_members = np.max([int(c) for c in df["container"].iloc[0] if c.isdigit()]) + 1
        aos_containers = [
            "".join([str(p) for p in perm]) for perm in permutations(range(n_members))
        ]
        soa_containers = [
            "_".join([str(p) for p in perm]) for perm in permutations(range(n_members))
        ]

        plot_scatter(input, df, args.output, args.aggregate, ["time"])
        plot_scatter(input, df, args.output, args.aggregate, ["PAPI_TOT_INS", "time"], outside_legend=True)

        if "ANY_DATA_CACHE_FILLS_FROM_SYSTEM:LCL_L2:LOCAL_CCX:NEAR_CACHE_NEAR_FAR:DRAM_IO_NEAR:FAR_CACHE_NEAR_FAR:DRAM_IO_FAR:ALT_MEM_NEAR_FAR" in df.columns:
            plot_scatter(input, df, args.output, args.aggregate,
                        ["ANY_DATA_CACHE_FILLS_FROM_SYSTEM:LCL_L2:LOCAL_CCX:NEAR_CACHE_NEAR_FAR:DRAM_IO_NEAR:FAR_CACHE_NEAR_FAR:DRAM_IO_FAR:ALT_MEM_NEAR_FAR"],
                        sort_by="time", outside_legend=True)

    ########
    for _ in range(len(args.input)):
        plot_runtime_histogram_all_minmax(args.input, [ pd.read_csv(input) for input in args.input ], args.output, args.aggregate)
        args.input = np.roll(args.input, -1)

    plot_runtime_histogram_max_psize_all_archs(
        args.input,
        dfs,
        args.output,
        args.aggregate,
        annotate="guidelines",
    )
