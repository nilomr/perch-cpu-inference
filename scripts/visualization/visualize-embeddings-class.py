#!/usr/bin/env python3
"""
UMAP visualization with well-spaced cursive labels, proper aspect ratio, and refined aesthetics.

Install:
    pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12 datashader scikit-learn adjustText
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from rich.panel import Panel
from rich.table import Table

import warnings

warnings.filterwarnings("ignore")

try:
    from cuml import UMAP
    import datashader as ds
    import datashader.transfer_functions as tf
    from scipy import ndimage
except ImportError as e:
    raise ImportError(
        f"Missing dependency: {e}\n"
        "Install: pip install --extra-index-url=https://pypi.nvidia.com "
        "cuml-cu12 datashader scikit-learn adjustText scipy"
    )

console = Console()

# ------------------------------ Config ------------------------------

EMBEDDINGS_DIR = "output-wytham-a4/embeddings_partitioned"
PREDICTIONS_DIR = "output-wytham-a4/predictions_partitioned"
OUTPUT_DIR = "umap_visualization"

MAX_FILES = None
N_WORKERS = 20

UMAP_N_NEIGHBORS = 80
UMAP_MIN_DIST = 0.15

N_TOP_CLASSES = 10
AX_PAD_FRAC = 0.20  # Extra padding for labels to sit in background
LABEL_PUSH = 0.15  # Push labels this far from centroid (fraction of plot range)

# Harmonious color palette
HARMONIOUS_COLORS = [
    "#8dd3c7",
    "#ffffb3",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
    "#d9d9d9",
    "#bc80bd",
]

# ------------------------------ IO ------------------------------


def _load_single_parquet(
    parquet_file: Path, emb_cols: List[str], metadata_cols: List[str]
) -> Optional[pd.DataFrame]:
    try:
        cols = metadata_cols + emb_cols
        return pd.read_parquet(parquet_file, engine="pyarrow", columns=cols)
    except Exception as e:
        console.print(f"[yellow]Skipping[/yellow] {parquet_file.name}: {e}")
        return None


def load_embeddings(
    embeddings_dir: str, max_files: Optional[int] = None, n_workers: int = 8
) -> pd.DataFrame:
    partition_dirs = sorted(Path(embeddings_dir).glob("*/"))
    all_files: List[Path] = []
    for d in partition_dirs:
        files = sorted(d.glob("*.parquet"))
        if max_files is not None:
            files = files[:max_files]
        all_files.extend(files)
    if not all_files:
        raise FileNotFoundError(f"No parquet files found in {embeddings_dir}")

    console.print(
        f"  Found {len(all_files)} parquet files across {len(partition_dirs)} partitions"
    )

    sample_df = pd.read_parquet(all_files[0], engine="pyarrow")
    emb_cols = sorted(
        [c for c in sample_df.columns if c.startswith("emb_")],
        key=lambda x: int(x.split("_")[1]),
    )
    metadata_cols = ["file", "file_path", "chunk_idx", "start_time", "end_time"]

    all_data: List[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {
            ex.submit(_load_single_parquet, f, emb_cols, metadata_cols): f
            for f in all_files
        }
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Loading parquet files...", total=len(futures))
            for fut in as_completed(futures):
                r = fut.result()
                if r is not None:
                    all_data.append(r)
                progress.update(task, advance=1)
    return pd.concat(all_data, ignore_index=True)


def load_predictions(predictions_dir: str) -> pd.DataFrame:
    files = sorted(Path(predictions_dir).glob("checkpoint_*.csv"))
    if not files:
        raise FileNotFoundError(f"No prediction CSVs found in {predictions_dir}")
    parts = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Loading prediction CSVs...", total=len(files))
        for f in files:
            parts.append(pd.read_csv(f))
            progress.update(task, advance=1)
    return pd.concat(parts, ignore_index=True)


# ------------------------------ Features ------------------------------


def extract_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    emb_cols = sorted(
        [c for c in df.columns if c.startswith("emb_")],
        key=lambda x: int(x.split("_")[1]),
    )
    embeddings = df[emb_cols].to_numpy(dtype=np.float32)
    predicted_classes = df["top1"].to_numpy()
    metadata_cols = ["file", "file_path", "chunk_idx", "start_time", "end_time"]
    metadata = df[metadata_cols].reset_index(drop=True)
    return embeddings, predicted_classes, metadata


def get_top_classes(
    predicted_classes: np.ndarray, n_top: int = 10
) -> Tuple[List[str], np.ndarray]:
    class_counts = pd.Series(predicted_classes).value_counts()
    top_classes = class_counts.head(n_top).index.tolist()
    class_indices = np.zeros(len(predicted_classes), dtype=int)
    for idx, cls in enumerate(top_classes, start=1):
        class_indices[predicted_classes == cls] = idx
    return top_classes, class_indices


def compute_robust_centroids(
    embeddings_2d: np.ndarray, class_indices: np.ndarray, n_classes: int
):
    """Median centroid for robust visual center."""
    centroids = []
    for cls_idx in range(1, n_classes + 1):
        mask = class_indices == cls_idx
        if mask.sum() == 0:
            centroids.append(np.array([np.nan, np.nan]))
            continue
        points = embeddings_2d[mask]
        centroid = np.median(points, axis=0)
        centroids.append(centroid)
    return np.array(centroids)


def _build_count_and_distance(embeddings_2d, class_indices, canvas_res, extent=None):
    # Datashader count grid
    df_plot = pd.DataFrame(
        {"x": embeddings_2d[:, 0], "y": embeddings_2d[:, 1], "class": class_indices}
    )
    canvas = ds.Canvas(plot_width=canvas_res, plot_height=canvas_res)
    agg = canvas.points(df_plot, "x", "y", agg=ds.count())
    counts = agg.values  # shape (H, W)

    # Extent from aggregation if not supplied
    if extent is None:
        extent = (agg.x_range[0], agg.x_range[1], agg.y_range[0], agg.y_range[1])

    # Binary obstacle mask: any data pixel is an obstacle
    obstacles = counts > 0

    # Pixel size in data units; use anisotropic sampling to get distances in data units
    xmin, xmax, ymin, ymax = extent
    H, W = counts.shape
    dx = (xmax - xmin) / max(W - 1, 1)
    dy = (ymax - ymin) / max(H - 1, 1)

    # Distance transform over background (~obstacles is background), distances in data units
    dist = ndimage.distance_transform_edt(~obstacles, sampling=(dy, dx))

    return counts, dist, extent


def _bilinear_sample(grid, x, y, extent):
    xmin, xmax, ymin, ymax = extent
    H, W = grid.shape
    # Map to fractional indices
    u = (x - xmin) / (xmax - xmin) * (W - 1)
    v = (y - ymin) / (ymax - ymin) * (H - 1)
    # Clamp
    u = np.clip(u, 0, W - 1 - 1e-6)
    v = np.clip(v, 0, H - 1 - 1e-6)
    i = np.floor(v).astype(int)
    j = np.floor(u).astype(int)
    du = u - j
    dv = v - i
    # Bilinear
    g00 = grid[i, j]
    g01 = grid[i, j + 1]
    g10 = grid[i + 1, j]
    g11 = grid[i + 1, j + 1]
    return (
        g00 * (1 - du) * (1 - dv)
        + g01 * (du) * (1 - dv)
        + g10 * (1 - du) * (dv)
        + g11 * (du) * (dv)
    )


def _pick_background_anchor(
    centroid,
    pushed_pos,
    dist_grid,
    extent,
    n_angles=48,
    n_r=12,
    r_min_frac=0.05,
    r_max_frac=0.5,
    prefer_pushed=True,
):
    # Search ring around centroid for maximum clearance
    xmin, xmax, ymin, ymax = extent
    plot_range = max(xmax - xmin, ymax - ymin)
    rads = np.linspace(r_min_frac * plot_range, r_max_frac * plot_range, n_r)
    thetas = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    # Optional: bias toward pushed direction
    bias_dir = (
        np.arctan2(pushed_pos[1] - centroid[1], pushed_pos[0] - centroid[0])
        if prefer_pushed
        else None
    )

    best_d = -np.inf
    best_xy = pushed_pos
    for r in rads:
        for th in thetas:
            x = centroid[0] + r * np.cos(th)
            y = centroid[1] + r * np.sin(th)
            # Skip if outside axes
            if not (xmin <= x <= xmax and ymin <= y <= ymax):
                continue
            d = float(_bilinear_sample(dist_grid, x, y, extent))
            # Tie-break: prefer angle near bias_dir
            if (
                prefer_pushed
                and np.isfinite(d)
                and d == best_d
                and bias_dir is not None
            ):
                delta = abs(np.arctan2(np.sin(th - bias_dir), np.cos(th - bias_dir)))
                delta_best = abs(
                    np.arctan2(np.sin(th - bias_dir), np.cos(th - bias_dir))
                )
                if delta < delta_best:
                    best_xy = (x, y)
            if d > best_d:
                best_d = d
                best_xy = (x, y)
    return np.array(best_xy), best_d


def _sample_obstacle_points(obstacles, extent, max_points=3000, rng=np.random):
    # Uniformly sample obstacle pixels to give adjustText a broad repulsion field
    rows, cols = np.where(obstacles)
    if len(rows) == 0:
        return np.array([]), np.array([])
    idx = np.arange(len(rows))
    if len(idx) > max_points:
        idx = rng.choice(idx, max_points, replace=False)
    rows = rows[idx]
    cols = cols[idx]
    xmin, xmax, ymin, ymax = extent
    H, W = obstacles.shape
    xs = xmin + cols / max(W - 1, 1) * (xmax - xmin)
    ys = ymin + rows / max(H - 1, 1) * (ymax - ymin)
    return xs, ys


def optimize_label_positions_radial(
    centroids, embeddings_2d, class_indices, n_classes, extent
):
    """Radial placement along rays from center, avoiding data clusters."""
    xmin, xmax, ymin, ymax = extent
    center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
    plot_range = max(xmax - xmin, ymax - ymin)

    # Build density grid for each class's own data
    grid_res = 150
    class_density_grids = {}
    for cls_idx in range(1, n_classes + 1):
        mask = class_indices == cls_idx
        if mask.sum() > 0:
            points = embeddings_2d[mask]
            density, x_edges, y_edges = np.histogram2d(
                points[:, 0],
                points[:, 1],
                bins=[grid_res, grid_res],
                range=[[xmin, xmax], [ymin, ymax]],
            )
            # Smooth and normalize
            from scipy.ndimage import gaussian_filter

            density = gaussian_filter(density, sigma=1.5)
            density = density / (density.max() + 1e-8)
            class_density_grids[cls_idx] = (density, x_edges, y_edges)
        else:
            class_density_grids[cls_idx] = None

    def sample_density_at_point(x, y, density_grid_data):
        """Sample density at a single point."""
        if density_grid_data is None:
            return 0.0
        density, x_edges, y_edges = density_grid_data
        # Find bin indices
        x_idx = np.searchsorted(x_edges, x) - 1
        y_idx = np.searchsorted(y_edges, y) - 1
        # Clamp to valid range
        x_idx = np.clip(x_idx, 0, grid_res - 1)
        y_idx = np.clip(y_idx, 0, grid_res - 1)
        return density[x_idx, y_idx]

    label_positions = []

    for cls_idx, centroid in enumerate(centroids, start=1):
        # Ray from center through centroid
        direction = centroid - center
        distance_to_centroid = np.linalg.norm(direction)
        if distance_to_centroid < 1e-6:
            # Centroid at center; pick arbitrary direction
            label_positions.append(centroid + np.array([0.15 * plot_range, 0]))
            continue

        direction_unit = direction / distance_to_centroid

        # Search along ray for lowest density position
        # Start at centroid and move outward
        search_distances = np.linspace(
            0.5 * plot_range,  # Minimum offset from centroid
            0.95 * plot_range,  # Maximum offset (don't go too far)
            20,  # Number of candidate positions
        )

        best_position = None
        best_score = np.inf

        density_grid_data = class_density_grids[cls_idx]

        for offset in search_distances:
            # Position along ray
            candidate = centroid + offset * direction_unit

            # Check if within bounds
            if not (xmin <= candidate[0] <= xmax and ymin <= candidate[1] <= ymax):
                continue

            # Score: density at this position + distance penalty
            density = sample_density_at_point(
                candidate[0], candidate[1], density_grid_data
            )

            # Prefer positions further from data but not too far from centroid
            # Balance: low density, reasonable distance
            distance_penalty = (
                offset / (0.3 * plot_range)
            ) ** 1.5  # Quadratic penalty for distance
            score = 3 * density + distance_penalty

            if score < best_score:
                best_score = score
                best_position = candidate

        if best_position is None:
            # Fallback: just push outward a fixed amount
            best_position = centroid + 0.35 * plot_range * direction_unit

        label_positions.append(best_position)

    return np.array(label_positions)


# ------------------------------ Plot ------------------------------


def create_class_plot(
    embeddings_2d: np.ndarray,
    class_indices: np.ndarray,
    top_classes: List[str],
    output_dir: str,
):
    # Use square figure for equal aspect ratio
    fig = plt.figure(figsize=(14, 14))
    fig.patch.set_facecolor("#1a1d21")

    # Main axes
    ax = plt.axes([0.05, 0.05, 0.9, 0.9])
    ax.set_facecolor("#1a1d21")
    ax.set_aspect("equal")  # Force equal aspect ratio for round points

    # Colors: 'Other' grey, then harmonious palette
    colors_hex = ["#3a3a3a"] + HARMONIOUS_COLORS[: len(top_classes)]

    # Datashader rendering with reduced alpha
    canvas_res = 2000
    df_plot = pd.DataFrame(
        {"x": embeddings_2d[:, 0], "y": embeddings_2d[:, 1], "class": class_indices}
    )
    canvas = ds.Canvas(plot_width=canvas_res, plot_height=canvas_res)
    agg = canvas.points(df_plot, "x", "y", agg=ds.mean("class"))
    color_key = {i: colors_hex[i] for i in range(len(colors_hex))}
    img = tf.shade(agg, color_key=color_key, how="linear", min_alpha=0)
    img = tf.spread(img, px=2)

    # Apply alpha to the entire image
    from PIL import Image

    pil_img = img.to_pil()
    pil_img = pil_img.convert("RGBA")
    pil_array = np.array(pil_img)
    # Reduce alpha channel to 0.7 where there are points
    alpha_mask = pil_array[:, :, 3] > 0
    pil_array[:, :, 3] = np.where(
        alpha_mask, (pil_array[:, :, 3] * 0.7).astype(np.uint8), 0
    )
    pil_img = Image.fromarray(pil_array)

    extent = (agg.x_range[0], agg.x_range[1], agg.y_range[0], agg.y_range[1])

    # Display with correct orientation
    ax.imshow(pil_img, extent=extent, origin="upper", aspect="auto")

    # Apply padding
    xpad = AX_PAD_FRAC * (extent[1] - extent[0])
    ypad = AX_PAD_FRAC * (extent[3] - extent[2])
    ax.set_xlim(extent[0] - xpad, extent[1] + xpad)
    ax.set_ylim(extent[2] - ypad, extent[3] + ypad)

    # Compute robust centroids
    centroids = compute_robust_centroids(embeddings_2d, class_indices, len(top_classes))

    # Sort labels by angle from center to minimize line crossings
    center_x = (extent[0] + extent[1]) / 2
    center_y = (extent[2] + extent[3]) / 2
    angles = np.arctan2(centroids[:, 1] - center_y, centroids[:, 0] - center_x)
    sort_indices = np.argsort(angles)

    # Reorder top_classes, centroids, and update class_indices
    top_classes_sorted = [top_classes[i] for i in sort_indices]
    centroids_sorted = centroids[sort_indices]
    colors_sorted = [HARMONIOUS_COLORS[i] for i in sort_indices]
    colors_hex = ["#3a3a3a"] + colors_sorted

    # Update class_indices to match new order
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(sort_indices)}
    new_class_indices = np.zeros_like(class_indices)
    for old_idx in range(1, len(top_classes) + 1):
        new_idx = old_to_new[old_idx - 1] + 1
        mask = class_indices == old_idx
        new_class_indices[mask] = new_idx
    class_indices = new_class_indices

    # Update variables
    top_classes = top_classes_sorted
    centroids = centroids_sorted

    # Colors: 'Other' grey, then harmonious palette
    # colors_hex already updated

    # Datashader rendering with reduced alpha
    canvas_res = 2000
    df_plot = pd.DataFrame(
        {"x": embeddings_2d[:, 0], "y": embeddings_2d[:, 1], "class": class_indices}
    )
    canvas = ds.Canvas(plot_width=canvas_res, plot_height=canvas_res)
    agg = canvas.points(df_plot, "x", "y", agg=ds.mean("class"))
    color_key = {i: colors_hex[i] for i in range(len(colors_hex))}
    img = tf.shade(agg, color_key=color_key, how="linear", min_alpha=0)
    img = tf.spread(img, px=2)

    # Apply alpha to the entire image
    from PIL import Image

    pil_img = img.to_pil()
    pil_img = pil_img.convert("RGBA")
    pil_array = np.array(pil_img)
    # Reduce alpha channel to 0.7 where there are points
    alpha_mask = pil_array[:, :, 3] > 0
    pil_array[:, :, 3] = np.where(
        alpha_mask, (pil_array[:, :, 3] * 0.7).astype(np.uint8), 0
    )
    pil_img = Image.fromarray(pil_array)

    extent = (agg.x_range[0], agg.x_range[1], agg.y_range[0], agg.y_range[1])

    # Display with correct orientation
    ax.imshow(pil_img, extent=extent, origin="upper", aspect="auto")

    # Apply padding
    xpad = AX_PAD_FRAC * (extent[1] - extent[0])
    ypad = AX_PAD_FRAC * (extent[3] - extent[2])
    ax.set_xlim(extent[0] - xpad, extent[1] + xpad)
    ax.set_ylim(extent[2] - ypad, extent[3] + ypad)

    # After computing centroids and sorting by angle...

    # Radial placement (avoids crossings, minimal distance)
    label_positions = optimize_label_positions_radial(
        centroids, embeddings_2d, class_indices, len(top_classes), extent
    )

    # Place labels at optimized positions
    texts = []
    for i, name in enumerate(top_classes):
        color = colors_hex[i + 1]
        display = name.replace("_", " ")

        ax.scatter(
            centroids[i, 0],
            centroids[i, 1],
            color=color,
            s=25,
            alpha=0.9,
            zorder=90,
            edgecolors="white",
            linewidths=0.5,
        )

        t = ax.text(
            label_positions[i, 0],
            label_positions[i, 1],
            display,
            fontsize=13,
            color=color,
            style="italic",
            ha="center",
            va="center",
            zorder=100,
        )
        texts.append(t)

        # Draw connector
        ax.plot(
            [centroids[i, 0], label_positions[i, 0]],
            [centroids[i, 1], label_positions[i, 1]],
            color=color,
            lw=1.0,
            alpha=0.7,
            zorder=95,
        )

    # Tidy axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for s in ax.spines.values():
        s.set_visible(False)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        f"{output_dir}/class_embeddings.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
        facecolor="#1a1d21",
        edgecolor="none",
    )
    plt.savefig(
        f"{output_dir}/class_embeddings.svg",
        bbox_inches="tight",
        pad_inches=0.1,
        facecolor="#1a1d21",
        edgecolor="none",
    )
    plt.close()

    console.print(
        f"\n✓ Class visualization saved as PNG and SVG: {len(embeddings_2d):,} points"
    )


# ------------------------------ Main ------------------------------


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    console.print(Panel.fit("→ Loading embeddings...", border_style="cyan"))
    df = load_embeddings(EMBEDDINGS_DIR, MAX_FILES, n_workers=N_WORKERS)
    console.print(f"  Loaded {len(df):,} vectors")

    console.print(Panel.fit("→ Loading predictions...", border_style="cyan"))
    df_pred = load_predictions(PREDICTIONS_DIR)
    console.print(f"  Loaded {len(df_pred):,} predictions")

    console.print(Panel.fit("→ Merging data...", border_style="cyan"))
    df = df.merge(
        df_pred, on=["file_path", "chunk_idx"], how="left", suffixes=("", "_pred")
    )
    console.print(f"  Merged, {len(df):,} rows")

    config_table = Table(title="Status", show_header=False)
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green bold")
    config_table.add_row("Embedding rows", f"{len(df):,}")
    config_table.add_row("Prediction rows", f"{len(df_pred):,}")
    config_table.add_row("UMAP neighbors", str(UMAP_N_NEIGHBORS))
    config_table.add_row("Top classes", str(N_TOP_CLASSES))
    config_table.add_row("Workers", str(N_WORKERS))
    console.print(config_table)

    console.print(Panel.fit("→ Extracting features...", border_style="cyan"))
    embeddings, predicted_classes, metadata = extract_features(df)
    console.print(f"  Processed {len(embeddings):,} samples")

    console.print(Panel.fit("→ Analyzing classes...", border_style="cyan"))
    top_classes, class_indices = get_top_classes(predicted_classes, N_TOP_CLASSES)
    n_other = int((class_indices == 0).sum())
    console.print(
        f"  Top {N_TOP_CLASSES} classes account for {len(predicted_classes) - n_other:,} points"
    )
    console.print(f"  Other classes: {n_other:,} points")

    console.print(Panel.fit("→ Running GPU-accelerated UMAP...", border_style="cyan"))
    umap = UMAP(
        n_components=2,
        metric="euclidean",
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        verbose=False,
    )
    embeddings_2d = umap.fit_transform(embeddings)
    if hasattr(embeddings_2d, "get"):
        embeddings_2d = embeddings_2d.get()

    console.print(Panel.fit("→ Creating visualization...", border_style="cyan"))
    create_class_plot(embeddings_2d, class_indices, top_classes, OUTPUT_DIR)

    results = metadata.copy()
    results["umap_x"] = embeddings_2d[:, 0]
    results["umap_y"] = embeddings_2d[:, 1]
    results["predicted_class"] = predicted_classes
    results["class_index"] = class_indices
    results.to_parquet(f"{OUTPUT_DIR}/umap_class_embeddings.parquet", index=False)

    console.print(
        Panel.fit(f"\n✓ Complete! Output: {OUTPUT_DIR}/", border_style="green")
    )


if __name__ == "__main__":
    main()
