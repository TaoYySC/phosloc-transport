# Panels: Figure 1(c,d,e); Supplementary Figure 1(a,b)


# Monorepo path constants (monorepo relative paths)
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PRECOMPUTED = PROJECT_ROOT / "data" / "precomputed"
TF_FAMILY_PATH = PROJECT_ROOT / "data" / "TF_family" / "TF_Information.txt"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox


positive_path = PROJECT_ROOT / "data" / "dataset_phos_site" / "TF_positive_phos_site_0608.csv"
tf_path = TF_FAMILY_PATH

output_root = PROJECT_ROOT / "results" / "0_dataset_description"
output_root.mkdir(parents=True, exist_ok=True)

dataset_out_dir = output_root
tf_family_fig_dir = output_root

IMPORT_LABEL = "Nuclear Import"
EXPORT_LABEL = "Nuclear Export"
DIRECTION_ORDER = [IMPORT_LABEL, EXPORT_LABEL]

IMPORT_COLOR = "#9a3f3f"
EXPORT_COLOR = "#4f7d95"
IMPORT_TEXT_COLOR = "#9a3f3f"
EXPORT_TEXT_COLOR = "#4f7d95"
ANNOTATION_LINE_COLOR = "#777777"

RESIDUE_COLORS = {
    "S": "#6292BE",
    "T": "#79C3EC",
    "Y": "#F79147",
}

ANNOTATE_MIN_COUNT = 4
KDE_BW_FACTOR = 1.6

MERGED_FIGSIZE = (8.4, 5.8)

FONT_FAMILY = "DejaVu Sans"
FONT_SIZE_BASE = 10
FONT_SIZE_TICK = 9
FONT_SIZE_LABEL = 10
FONT_SIZE_TITLE = 12
FONT_SIZE_PANEL = 12
FONT_SIZE_LEGEND = 9
FONT_SIZE_ANNOT = 8

LINEWIDTH_AXIS = 0.8
LINEWIDTH_MAIN = 1.2
LINEWIDTH_FILL_EDGE = 0.6

plt.rcParams["font.family"] = FONT_FAMILY
plt.rcParams["font.size"] = FONT_SIZE_BASE
plt.rcParams["axes.titlesize"] = FONT_SIZE_TITLE
plt.rcParams["axes.labelsize"] = FONT_SIZE_LABEL
plt.rcParams["xtick.labelsize"] = FONT_SIZE_TICK
plt.rcParams["ytick.labelsize"] = FONT_SIZE_TICK
plt.rcParams["legend.fontsize"] = FONT_SIZE_LEGEND
plt.rcParams["axes.linewidth"] = LINEWIDTH_AXIS
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["path.simplify"] = False
plt.rcParams["axes.unicode_minus"] = False


def count_autopct(values):
    total = np.sum(values)

    def _autopct(pct):
        count = int(round(pct * total / 100.0))
        return str(count) if count > 0 else ""
    return _autopct


def find_direction_column(df):
    candidates = [
        "Transport_Direction",
        "transport_direction",
        "DIRECTION",
        "LABEL",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"No transport direction column found. Expected one of: {candidates}")


def normalize_direction(x):
    x = str(x).strip().lower()

    if x in {"nuclear import", "import", "1", "in", "nls_import"}:
        return IMPORT_LABEL
    if x in {"nuclear export", "export", "0", "out", "nes_export"}:
        return EXPORT_LABEL

    if "import" in x:
        return IMPORT_LABEL
    if "export" in x:
        return EXPORT_LABEL

    return np.nan


def compute_relative_position_kde(values, x_min=0.0, x_max=1.0, n_points=300, bw_factor=1.0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return None, None, None

    if len(np.unique(values)) == 1:
        x_grid = np.linspace(x_min, x_max, n_points)
        y_grid = np.zeros_like(x_grid)
        idx = np.argmin(np.abs(x_grid - values[0]))
        y_grid[idx] = 1.0
        return None, x_grid, y_grid

    kde = gaussian_kde(values, bw_method=lambda s: s.scotts_factor() * bw_factor)
    x_grid = np.linspace(x_min, x_max, n_points)
    y_grid = kde(x_grid)
    return kde, x_grid, y_grid


def compute_kde(count_series):
    values = np.asarray(count_series.values, dtype=float)
    if len(values) == 0:
        return None, None, None
    if len(np.unique(values)) == 1:
        return None, None, None

    kde = gaussian_kde(values)
    x_grid = np.linspace(max(0.5, values.min() - 0.5), values.max() + 0.5, 200)
    y_grid = kde(x_grid)
    return kde, x_grid, y_grid


def build_exact_count_groups(import_counts, export_counts, annotate_threshold=4):
    import_df = pd.DataFrame({
        "TF": import_counts.index.astype(str),
        "count": import_counts.values.astype(int),
        "direction": "import"
    })

    export_df = pd.DataFrame({
        "TF": export_counts.index.astype(str),
        "count": export_counts.values.astype(int),
        "direction": "export"
    })

    merged = pd.concat([import_df, export_df], axis=0, ignore_index=True)
    merged = merged[merged["count"] >= annotate_threshold].copy()

    grouped = (
        merged.groupby(["count", "direction"])["TF"]
        .apply(lambda s: sorted(s.tolist()))
        .unstack()
        .reset_index()
        .sort_values("count")
    )

    if "import" not in grouped.columns:
        grouped["import"] = None
    if "export" not in grouped.columns:
        grouped["export"] = None

    grouped["import"] = grouped["import"].apply(lambda x: x if isinstance(x, list) else [])
    grouped["export"] = grouped["export"].apply(lambda x: x if isinstance(x, list) else [])

    return grouped


def build_density_lookup(kde_obj, x_values):
    if kde_obj is None:
        return {int(x): 0.0 for x in x_values}
    y = kde_obj(np.asarray(x_values, dtype=float))
    return {int(x): float(v) for x, v in zip(x_values, y)}


def annotate_site_groups_above_axes(
    ax,
    grouped_df,
    import_density_lookup,
    export_density_lookup,
    import_curve_y_grid=None,
    export_curve_y_grid=None,
    import_text_color=IMPORT_TEXT_COLOR,
    export_text_color=EXPORT_TEXT_COLOR,
    line_color=ANNOTATION_LINE_COLOR,
    name_fontsize=FONT_SIZE_ANNOT,
    name_step_axes=0.1,
    connector_linewidth=0.8,
    x_jitter_axes=0.014,
    y_padding_axes=0.008,
    curve_clearance_data=0.01,
    global_floor_extra_data=0.002,
    max_text_base_y_data=0.30,
    max_raise_iters=200,
    custom_label_y_offsets=None,
    connector_top_gap_axes=0.020,
    manual_label_positions=None,
):
    trans = ax.get_xaxis_transform()
    fig = ax.figure

    if custom_label_y_offsets is None:
        custom_label_y_offsets = {}

    if manual_label_positions is None:
        manual_label_positions = {}

    placed_boxes = []
    max_top = 0.98

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]

    def data_y_to_axes(y):
        if y_range <= 0:
            return 0.0
        return (y - ylim[0]) / y_range

    def axes_x_to_data(xa):
        return xlim[0] + xa * x_range

    def overlaps(box1, box2, pad=2.0):
        return not (
            box1.x1 + pad < box2.x0 or
            box1.x0 - pad > box2.x1 or
            box1.y1 + pad < box2.y0 or
            box1.y0 - pad > box2.y1
        )

    def get_label_offset(count_value, direction_value, label_value):
        return custom_label_y_offsets.get((count_value, direction_value, label_value), 0.0)

    def get_manual_position(count_value, direction_value, label_value):
        return manual_label_positions.get((count_value, direction_value, label_value), None)

    rows = grouped_df.sort_values("count").to_dict("records")

    for row in rows:
        x = int(row["count"])
        import_names = row["import"]
        export_names = row["export"]

        if not import_names and not export_names:
            continue

        labels = (
            [(name, "import", import_text_color) for name in import_names] +
            [(name, "export", export_text_color) for name in export_names]
        )

        import_curve_y = import_density_lookup.get(x, 0.0)
        export_curve_y = export_density_lookup.get(x, 0.0)
        local_curve_top_y_data = max(import_curve_y, export_curve_y)

        n_labels = len(labels)

        local_extra_lift_axes = max(0.0, (n_labels - 1) * name_step_axes * 0.10)
        local_extra_lift_data = local_extra_lift_axes * y_range

        y_anchor_data = local_curve_top_y_data + curve_clearance_data + local_extra_lift_data
        y_anchor_data = min(y_anchor_data, max_text_base_y_data)
        y_anchor = data_y_to_axes(y_anchor_data)

        x_offsets = [0.0, -x_jitter_axes, x_jitter_axes, -2 * x_jitter_axes, 2 * x_jitter_axes]

        placed = False
        chosen_x_axes = None
        chosen_y_axes = None
        final_block_bbox = None

        x_axes_center = (x - xlim[0]) / x_range

        manual_items = []
        auto_items = []

        for label, direction, color in labels:
            manual_pos = get_manual_position(x, direction, label)
            if manual_pos is None:
                auto_items.append((label, direction, color))
            else:
                manual_items.append((label, direction, color, manual_pos))

        for label, direction, color, (mx, my) in manual_items:
            ax.text(
                mx,
                my,
                label,
                fontsize=name_fontsize,
                color=color,
                ha="center",
                va="bottom",
                transform=ax.transAxes,
                clip_on=False,
            )

            connector_top_axes = max(0.0, my - connector_top_gap_axes)
            curve_y_data = max(import_curve_y, export_curve_y)
            curve_y_axes = data_y_to_axes(curve_y_data)
            mx_data = axes_x_to_data(mx)

            ax.plot(
                [x, mx_data],
                [curve_y_axes, connector_top_axes],
                color=line_color,
                linewidth=connector_linewidth,
                zorder=5,
                transform=trans,
                clip_on=False
            )

        if not auto_items:
            continue

        for x_offset in x_offsets:
            trial_x_axes = x_axes_center + x_offset
            trial_y_axes = y_anchor

            for _ in range(max_raise_iters):
                trial_texts = []

                for i, (label, direction, color) in enumerate(auto_items):
                    y_text = trial_y_axes + i * name_step_axes + get_label_offset(x, direction, label)
                    t = ax.text(
                        trial_x_axes,
                        y_text,
                        label,
                        fontsize=name_fontsize,
                        color=color,
                        ha="center",
                        va="bottom",
                        transform=ax.transAxes,
                        clip_on=False,
                        alpha=0.0,
                    )
                    trial_texts.append(t)

                fig.canvas.draw()

                bboxes = [t.get_window_extent(renderer=renderer) for t in trial_texts]
                x0 = min(b.x0 for b in bboxes)
                y0 = min(b.y0 for b in bboxes)
                x1 = max(b.x1 for b in bboxes)
                y1 = max(b.y1 for b in bboxes)
                block_bbox = Bbox.from_extents(x0, y0, x1, y1)

                has_overlap = any(overlaps(block_bbox, old_bbox) for old_bbox in placed_boxes)

                for t in trial_texts:
                    t.remove()

                if not has_overlap:
                    chosen_x_axes = trial_x_axes
                    chosen_y_axes = trial_y_axes
                    final_block_bbox = block_bbox
                    placed = True
                    break

                trial_y_axes += y_padding_axes

            if placed:
                break

        if not placed:
            chosen_x_axes = x_axes_center
            chosen_y_axes = y_anchor
            final_texts = []

            for i, (label, direction, color) in enumerate(auto_items):
                y_text = chosen_y_axes + i * name_step_axes + get_label_offset(x, direction, label)
                t = ax.text(
                    chosen_x_axes,
                    y_text,
                    label,
                    fontsize=name_fontsize,
                    color=color,
                    ha="center",
                    va="bottom",
                    transform=ax.transAxes,
                    clip_on=False,
                )
                final_texts.append(t)

            fig.canvas.draw()
            bboxes = [t.get_window_extent(renderer=renderer) for t in final_texts]
            x0 = min(b.x0 for b in bboxes)
            y0 = min(b.y0 for b in bboxes)
            x1 = max(b.x1 for b in bboxes)
            y1 = max(b.y1 for b in bboxes)
            final_block_bbox = Bbox.from_extents(x0, y0, x1, y1)
        else:
            for i, (label, direction, color) in enumerate(auto_items):
                y_text = chosen_y_axes + i * name_step_axes + get_label_offset(x, direction, label)
                ax.text(
                    chosen_x_axes,
                    y_text,
                    label,
                    fontsize=name_fontsize,
                    color=color,
                    ha="center",
                    va="bottom",
                    transform=ax.transAxes,
                    clip_on=False,
                )

        placed_boxes.append(final_block_bbox)

        connector_top_axes = max(0.0, chosen_y_axes - connector_top_gap_axes)
        curve_y_data = max(import_curve_y, export_curve_y)
        curve_y_axes = data_y_to_axes(curve_y_data)

        ax.plot(
            [x, x],
            [curve_y_axes, connector_top_axes],
            color=line_color,
            linewidth=connector_linewidth,
            zorder=5,
            transform=trans,
            clip_on=False
        )

        max_offset_here = max(get_label_offset(x, direction, label) for label, direction, _ in auto_items)
        block_top_axes = chosen_y_axes + len(auto_items) * name_step_axes + max_offset_here
        if block_top_axes > max_top:
            max_top = block_top_axes

    return max_top


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=FONT_SIZE_TICK, width=LINEWIDTH_AXIS)


def draw_panel_a(ax, summary_df):
    x = np.arange(len(DIRECTION_ORDER))
    bar_width = 0.28

    site_color = "#F8CBAD"
    protein_color = "#BDD7EE"

    bars_site = ax.bar(
        x - bar_width / 2,
        summary_df["Site_Count"].values,
        width=bar_width,
        color=site_color,
        edgecolor="white",
        linewidth=LINEWIDTH_FILL_EDGE,
        label="Site Count"
    )

    bars_protein = ax.bar(
        x + bar_width / 2,
        summary_df["Unique_Protein_Count"].values,
        width=bar_width,
        color=protein_color,
        edgecolor="white",
        linewidth=LINEWIDTH_FILL_EDGE,
        label="Protein Count"
    )

    max_count = max(
        summary_df["Site_Count"].max(),
        summary_df["Unique_Protein_Count"].max()
    )

    ax.set_ylim(0, max_count * 1.24)

    for bar in bars_site:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max_count * 0.025,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_BASE
        )

    for bar in bars_protein:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max_count * 0.025,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_BASE
        )

    ax.set_xticks(x)
    ax.set_xticklabels(DIRECTION_ORDER)
    ax.set_ylabel("Count")

    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.8, 1.1),
        ncol=1,
        handlelength=1.6,
        labelspacing=0.4,
        borderaxespad=0.2
    )

    style_axes(ax)


def draw_panel_b(ax_left, ax_right, sty_counts, legend_ax=None):
    pie_order = ["S", "T", "Y"]
    pie_titles = ["Nuclear Import", "Nuclear Export"]

    for ax, direction, pie_title in zip([ax_left, ax_right], DIRECTION_ORDER, pie_titles):
        values = sty_counts.loc[direction, pie_order].values
        colors = [RESIDUE_COLORS[r] for r in pie_order]

        ax.pie(
            values,
            labels=None,
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops={"edgecolor": "white", "linewidth": 0.8},
            textprops={"fontsize": FONT_SIZE_BASE, "color": "black"},
            autopct=count_autopct(values),
            pctdistance=0.62,
            radius=1.12
        )
        ax.set_aspect("equal")

        ax.text(
            0.5,
            -0.12,
            pie_title,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=FONT_SIZE_TICK
        )

    legend_handles = [
        Patch(facecolor=RESIDUE_COLORS[r], edgecolor="white", linewidth=0.8, label=r)
        for r in pie_order
    ]

    if legend_ax is not None:
        legend_ax.set_axis_off()
        legend_ax.legend(
            handles=legend_handles,
            title="Residue",
            loc="center left",
            frameon=False,
            fontsize=FONT_SIZE_LEGEND,
            title_fontsize=FONT_SIZE_LEGEND
        )
    else:
        ax_right.legend(
            handles=legend_handles,
            title="Residue",
            loc="upper right",
            bbox_to_anchor=(1.18, 1.08),
            frameon=False,
            fontsize=FONT_SIZE_LEGEND,
            title_fontsize=FONT_SIZE_LEGEND
        )


def draw_panel_c(ax, plot_df):
    import_values = plot_df.loc[plot_df["Transport_Direction"] == IMPORT_LABEL, "relative_position"].to_numpy(dtype=float)
    export_values = plot_df.loc[plot_df["Transport_Direction"] == EXPORT_LABEL, "relative_position"].to_numpy(dtype=float)

    kde_import, x_grid_import, y_grid_import = compute_relative_position_kde(
        import_values,
        x_min=0.0,
        x_max=1.0,
        bw_factor=KDE_BW_FACTOR
    )
    kde_export, x_grid_export, y_grid_export = compute_relative_position_kde(
        export_values,
        x_min=0.0,
        x_max=1.0,
        bw_factor=KDE_BW_FACTOR
    )

    if x_grid_import is not None and y_grid_import is not None:
        if kde_import is None and len(import_values) > 0:
            ax.axvline(import_values[0], color=IMPORT_COLOR, linewidth=LINEWIDTH_MAIN)
        else:
            ax.plot(x_grid_import, y_grid_import, color=IMPORT_COLOR, linewidth=LINEWIDTH_MAIN)
            ax.fill_between(x_grid_import, 0, y_grid_import, color=IMPORT_COLOR, alpha=0.22)

    if x_grid_export is not None and y_grid_export is not None:
        if kde_export is None and len(export_values) > 0:
            ax.axvline(export_values[0], color=EXPORT_COLOR, linewidth=LINEWIDTH_MAIN)
        else:
            ax.plot(x_grid_export, y_grid_export, color=EXPORT_COLOR, linewidth=LINEWIDTH_MAIN)
            ax.fill_between(x_grid_export, 0, y_grid_export, color=EXPORT_COLOR, alpha=0.22)

    ax.set_xlabel("Relative phosphosite position in protein")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)

    style_axes(ax)


def draw_panel_d(ax, family_direction_top):
    plot_df = family_direction_top.sort_values("Total", ascending=True)

    y_labels = [
        f"{family} ({int(plot_df.loc[family, 'Family_Total_TF'])})"
        for family in plot_df.index
    ]

    ax.barh(
        y_labels,
        plot_df[IMPORT_LABEL],
        color=IMPORT_COLOR,
        edgecolor="white",
        linewidth=LINEWIDTH_FILL_EDGE,
        label=IMPORT_LABEL
    )

    ax.barh(
        y_labels,
        plot_df[EXPORT_LABEL],
        left=plot_df[IMPORT_LABEL],
        color=EXPORT_COLOR,
        edgecolor="white",
        linewidth=LINEWIDTH_FILL_EDGE,
        label=EXPORT_LABEL
    )

    for i, family in enumerate(plot_df.index):
        import_count = int(plot_df.loc[family, IMPORT_LABEL])
        export_count = int(plot_df.loc[family, EXPORT_LABEL])
        import_tf_count = int(plot_df.loc[family, "Import_TF_Count"])
        export_tf_count = int(plot_df.loc[family, "Export_TF_Count"])

        if import_count > 0:
            ax.text(
                import_count / 2.0,
                i,
                f"{import_count}/{import_tf_count}",
                va="center",
                ha="center",
                fontsize=FONT_SIZE_ANNOT,
                color="white"
            )

        if export_count > 0:
            ax.text(
                import_count + export_count / 2.0,
                i,
                f"{export_count}/{export_tf_count}",
                va="center",
                ha="center",
                fontsize=FONT_SIZE_ANNOT,
                color="white"
            )

    ax.set_xlabel("Positive Phosphosite Count")
    ax.set_ylabel("TF Family")
    ax.legend(frameon=False, loc="lower right")

    style_axes(ax)


def draw_panel_e(ax, import_counts, export_counts):
    kde_import, x_grid_import, y_grid_import = compute_kde(import_counts)
    kde_export, x_grid_export, y_grid_export = compute_kde(export_counts)

    if kde_import is None:
        x0 = float(import_counts.iloc[0])
        ax.axvline(x0, color=IMPORT_COLOR, linewidth=LINEWIDTH_MAIN, label="Nuclear import")
    else:
        ax.plot(x_grid_import, y_grid_import, color=IMPORT_COLOR, linewidth=LINEWIDTH_MAIN, label="Nuclear import")
        ax.fill_between(x_grid_import, 0, y_grid_import, color=IMPORT_COLOR, alpha=0.28)

    if kde_export is None:
        x0 = float(export_counts.iloc[0])
        ax.axvline(x0, color=EXPORT_COLOR, linewidth=LINEWIDTH_MAIN, label="Nuclear export")
    else:
        ax.plot(x_grid_export, y_grid_export, color=EXPORT_COLOR, linewidth=LINEWIDTH_MAIN, label="Nuclear export")
        ax.fill_between(x_grid_export, 0, y_grid_export, color=EXPORT_COLOR, alpha=0.28)

    grouped_df = build_exact_count_groups(
        import_counts=import_counts,
        export_counts=export_counts,
        annotate_threshold=ANNOTATE_MIN_COUNT
    )

    site_x_values = grouped_df["count"].astype(int).tolist()
    import_density_lookup = build_density_lookup(kde_import, site_x_values)
    export_density_lookup = build_density_lookup(kde_export, site_x_values)

    max_count = int(
        max(
            import_counts.max() if len(import_counts) > 0 else 1,
            export_counts.max() if len(export_counts) > 0 else 1
        )
    )

    ymax = max(
        np.max(y_grid_import) if y_grid_import is not None else 0.0,
        np.max(y_grid_export) if y_grid_export is not None else 0.0,
        0.01
    )

    ax.set_xlim(0.5, max_count + 1.2)
    ax.set_ylim(0, ymax * 1.18)
    ax.set_xlabel("Positive site count per TF")
    ax.set_ylabel("Density")

    style_axes(ax)

    annotate_site_groups_above_axes(
        ax,
        grouped_df=grouped_df,
        import_density_lookup=import_density_lookup,
        export_density_lookup=export_density_lookup,
        import_curve_y_grid=y_grid_import,
        export_curve_y_grid=y_grid_export,
        import_text_color=IMPORT_TEXT_COLOR,
        export_text_color=EXPORT_TEXT_COLOR,
        line_color=ANNOTATION_LINE_COLOR,
        name_fontsize=FONT_SIZE_ANNOT,
        name_step_axes=0.04,
        connector_linewidth=0.8,
        x_jitter_axes=0.010,
        y_padding_axes=0.004,
        curve_clearance_data=0.012,
        global_floor_extra_data=0.000,
        max_text_base_y_data=0.13,
        max_raise_iters=200,
        connector_top_gap_axes=0.012,
        custom_label_y_offsets={},
        manual_label_positions={},
    )

    ax.legend(loc="upper right", frameon=False)


def save_relative_position_table(df, out_dir):
    if "POSITION" not in df.columns:
        raise ValueError("Missing required column: POSITION")

    direction_col = find_direction_column(df)

    if "LENGTH" in df.columns:
        protein_length = pd.to_numeric(df["LENGTH"], errors="coerce")
    elif "FULL_SEQUENCE" in df.columns:
        protein_length = df["FULL_SEQUENCE"].astype(str).str.len()
    else:
        raise ValueError("Neither LENGTH nor FULL_SEQUENCE column was found.")

    position = pd.to_numeric(df["POSITION"], errors="coerce")
    direction = df[direction_col].apply(normalize_direction)

    plot_df = pd.DataFrame({
        "POSITION": position,
        "LENGTH": protein_length,
        "Transport_Direction": direction,
    })

    plot_df = plot_df.dropna(subset=["POSITION", "LENGTH", "Transport_Direction"])
    plot_df = plot_df[(plot_df["POSITION"] > 0) & (plot_df["LENGTH"] > 0)]
    plot_df = plot_df[plot_df["POSITION"] <= plot_df["LENGTH"]]
    plot_df["relative_position"] = plot_df["POSITION"] / plot_df["LENGTH"]

    plot_df.to_csv(out_dir / "phos_site_relative_position_values_with_direction.csv", index=False)
    return plot_df


def save_merged_summary_figure(summary_df, sty_counts, plot_df, family_direction_top, import_counts, export_counts, out_dir):
    fig = plt.figure(figsize=MERGED_FIGSIZE)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        width_ratios=[1.0, 1.15],
        height_ratios=[1.0, 0.62, 1.22],
        left=0.06,
        right=0.985,
        top=0.97,
        bottom=0.06,
        wspace=0.28,
        hspace=0.26
    )

    ax_a = fig.add_subplot(gs[0, 0])

    gs_b = gs[0, 1].subgridspec(
        1,
        3,
        width_ratios=[1.2, 1.2, 0.55],
        wspace=0.02
    )
    ax_b1 = fig.add_subplot(gs_b[0, 0])
    ax_b2 = fig.add_subplot(gs_b[0, 1])
    ax_b_leg = fig.add_subplot(gs_b[0, 2])

    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1:, 0])
    ax_e = fig.add_subplot(gs[2, 1])

    draw_panel_a(ax_a, summary_df)
    draw_panel_b(ax_b1, ax_b2, sty_counts, legend_ax=ax_b_leg)
    draw_panel_c(ax_c, plot_df)
    draw_panel_d(ax_d, family_direction_top)
    draw_panel_e(ax_e, import_counts, export_counts)

    out_png = out_dir / "merged_dataset_tf_summary_5panel.png"
    out_pdf = out_dir / "merged_dataset_tf_summary_5panel.pdf"
    out_svg = out_dir / "merged_dataset_tf_summary_5panel.svg"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", transparent=False)
    fig.savefig(out_svg, format="svg", bbox_inches="tight", transparent=False)

    plt.close(fig)

    return out_png, out_pdf, out_svg


def main():
    positive_df = pd.read_csv(positive_path)

    if "Transport_Direction" not in positive_df.columns:
        raise ValueError("Column 'Transport_Direction' not found.")
    if "RESIDUE" not in positive_df.columns:
        raise ValueError("Column 'RESIDUE' not found.")
    if "ACC_ID" not in positive_df.columns:
        raise ValueError("Column 'ACC_ID' not found.")

    transport_df = positive_df[positive_df["Transport_Direction"].isin(DIRECTION_ORDER)].copy()

    direction_counts = (
        transport_df["Transport_Direction"]
        .value_counts()
        .reindex(DIRECTION_ORDER, fill_value=0)
    )

    site_counts = direction_counts.copy()

    protein_counts = (
        transport_df.groupby("Transport_Direction")["ACC_ID"]
        .nunique()
        .reindex(DIRECTION_ORDER, fill_value=0)
    )

    summary_df = pd.DataFrame({
        "Transport_Direction": DIRECTION_ORDER,
        "Site_Count": [site_counts[IMPORT_LABEL], site_counts[EXPORT_LABEL]],
        "Unique_Protein_Count": [protein_counts[IMPORT_LABEL], protein_counts[EXPORT_LABEL]],
    })
    summary_df.to_csv(dataset_out_dir / "transport_direction_site_protein_counts.csv", index=False)

    residue_order = ["S", "T", "Y"]
    sty_counts = (
        transport_df.groupby(["Transport_Direction", "RESIDUE"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=DIRECTION_ORDER, columns=residue_order, fill_value=0)
    )
    sty_counts.to_csv(dataset_out_dir / "import_export_sty_counts.csv")

    plot_df = save_relative_position_table(positive_df, dataset_out_dir)

    if "GENE" in positive_df.columns:
        tf_col = "GENE"
    elif "Gene_Name" in positive_df.columns:
        tf_col = "Gene_Name"
    elif "Protein_Name" in positive_df.columns:
        tf_col = "Protein_Name"
    elif "ACC_ID" in positive_df.columns:
        tf_col = "ACC_ID"
    else:
        raise ValueError("No suitable TF identifier column found.")

    import_counts = (
        transport_df[transport_df["Transport_Direction"] == IMPORT_LABEL]
        .groupby(tf_col)
        .size()
        .astype(int)
    )

    export_counts = (
        transport_df[transport_df["Transport_Direction"] == EXPORT_LABEL]
        .groupby(tf_col)
        .size()
        .astype(int)
    )

    import_counts.to_csv(dataset_out_dir / "tf_positive_site_count_import.csv", header=["site_count"])
    export_counts.to_csv(dataset_out_dir / "tf_positive_site_count_export.csv", header=["site_count"])

    tf_df = pd.read_csv(tf_path, sep="\t")
    positive_tf_df = positive_df.copy()

    if tf_col == "ACC_ID":
        raise ValueError("TF family mapping requires a gene-level column such as GENE or Gene_Name.")

    positive_tf_df["GENE_norm"] = positive_tf_df[tf_col].astype(str).str.strip().str.upper()

    tf_df = tf_df.copy()
    tf_df["TF_Species"] = tf_df["TF_Species"].astype(str).str.strip()
    tf_df["TF_Name"] = tf_df["TF_Name"].astype(str).str.strip()
    tf_df["Family_Name"] = tf_df["Family_Name"].astype(str).str.strip()
    tf_df["TF_Name_norm"] = tf_df["TF_Name"].str.upper()

    tf_df = tf_df[tf_df["TF_Species"] == "Homo_sapiens"].copy()

    dbid_cols = [c for c in tf_df.columns if str(c).startswith("DBID")]
    keep_cols = ["TF_Name", "TF_Name_norm", "Family_Name"]
    keep_cols.extend(dbid_cols)

    tf_ref = tf_df[keep_cols].drop_duplicates(subset=["TF_Name_norm", "Family_Name"]).copy()

    merged_df = positive_tf_df.merge(
        tf_ref,
        left_on="GENE_norm",
        right_on="TF_Name_norm",
        how="left"
    )

    matched_df = merged_df[merged_df["Family_Name"].notna()].copy()

    matched_df["Family_Name"] = matched_df["Family_Name"].astype(str).str.strip()
    family_upper = matched_df["Family_Name"].str.upper()

    matched_df = matched_df[
        (family_upper != "OTHERS") &
        (~family_upper.str.contains("UNKNOWN", na=False))
    ].copy()

    family_direction = (
        matched_df.groupby(["Family_Name", "Transport_Direction"])
        .size()
        .unstack(fill_value=0)
    )

    family_direction_tf = (
        matched_df.groupby(["Family_Name", "Transport_Direction"])["GENE_norm"]
        .nunique()
        .unstack(fill_value=0)
    )

    family_total_tf = (
        tf_ref.groupby("Family_Name")["TF_Name_norm"]
        .nunique()
    )

    family_direction = family_direction[
        (family_direction.index.astype(str).str.strip().str.upper() != "OTHERS") &
        (~family_direction.index.astype(str).str.strip().str.upper().str.contains("UNKNOWN", na=False))
    ].copy()

    if IMPORT_LABEL not in family_direction.columns:
        family_direction[IMPORT_LABEL] = 0
    if EXPORT_LABEL not in family_direction.columns:
        family_direction[EXPORT_LABEL] = 0

    if IMPORT_LABEL not in family_direction_tf.columns:
        family_direction_tf[IMPORT_LABEL] = 0
    if EXPORT_LABEL not in family_direction_tf.columns:
        family_direction_tf[EXPORT_LABEL] = 0

    family_direction["Import_TF_Count"] = family_direction_tf[IMPORT_LABEL].reindex(family_direction.index, fill_value=0).astype(int)
    family_direction["Export_TF_Count"] = family_direction_tf[EXPORT_LABEL].reindex(family_direction.index, fill_value=0).astype(int)
    family_direction["Family_Total_TF"] = family_total_tf.reindex(family_direction.index, fill_value=0).astype(int)

    family_direction["Total"] = family_direction[IMPORT_LABEL] + family_direction[EXPORT_LABEL]
    family_direction = family_direction.sort_values("Total", ascending=False)
    family_direction_top = family_direction.head(15).copy()
    family_direction_top.to_csv(tf_family_fig_dir / "tf_family_transport_direction_counts_no_others.csv")

    out_png, out_pdf, out_svg = save_merged_summary_figure(
        summary_df=summary_df,
        sty_counts=sty_counts,
        plot_df=plot_df,
        family_direction_top=family_direction_top,
        import_counts=import_counts,
        export_counts=export_counts,
        out_dir=output_root
    )

    print("Saved merged figure:")
    print(out_png)
    print(out_pdf)
    print(out_svg)


if __name__ == "__main__":
    main()