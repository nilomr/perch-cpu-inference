#!/usr/bin/env python3
"""
GPU-Accelerated UMAP visualization of perch embeddings with temporal encoding.

Install:
    pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12 datashader scikit-learn
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, Normalize, LinearSegmentedColormap
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
    import matplotlib.cm as cm
except ImportError as e:
    raise ImportError(
        f"Missing dependency: {e}\n"
        "Install: pip install --extra-index-url=https://pypi.nvidia.com "
        "cuml-cu12 datashader scikit-learn"
    )

console = Console()

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

EMBEDDINGS_DIR = "output-wytham-a4/embeddings_partitioned"
PREDICTIONS_DIR = "output-wytham-a4/predictions_partitioned"
OUTPUT_DIR = "umap_visualization"

MAX_FILES = None
N_WORKERS = 20

UMAP_N_NEIGHBORS = 80
UMAP_MIN_DIST = 0.15

CBAR_FG = "#e0e0e0"


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------

def _load_single_parquet(
    parquet_file: Path, emb_cols: List[str], metadata_cols: List[str]
) -> Optional[pd.DataFrame]:
    """Load a single parquet file."""
    try:
        cols = metadata_cols + emb_cols
        df = pd.read_parquet(parquet_file, engine="pyarrow", columns=cols)
        return df
    except Exception as e:
        console.print(f"[yellow]Skipping[/yellow] {parquet_file.name}: {e}")
        return None


def load_embeddings(
    embeddings_dir: str, max_files: Optional[int] = None, n_workers: int = 8
) -> pd.DataFrame:
    """Load embeddings from partitioned parquet files in parallel."""
    partition_dirs = sorted(Path(embeddings_dir).glob("*/"))

    all_files: List[Path] = []
    for partition_dir in partition_dirs:
        files = sorted(partition_dir.glob("*.parquet"))
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
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_load_single_parquet, f, emb_cols, metadata_cols): f
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
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    all_data.append(result)
                progress.update(task, advance=1)

    return pd.concat(all_data, ignore_index=True)


def load_predictions(predictions_dir: str) -> pd.DataFrame:
    """Load prediction CSVs."""
    all_files = sorted(Path(predictions_dir).glob("checkpoint_*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No prediction CSVs found in {predictions_dir}")

    all_data: List[pd.DataFrame] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Loading prediction CSVs...", total=len(all_files))
        for f in all_files:
            df = pd.read_csv(f)
            all_data.append(df)
            progress.update(task, advance=1)

    return pd.concat(all_data, ignore_index=True)


# ---------------------------------------------------------------------
# Temporal feature extraction
# ---------------------------------------------------------------------

def extract_temporal_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray,
                                                         np.ndarray, pd.DataFrame, np.ndarray]:
    """Extract embeddings, time-of-day, day-of-year, and metadata."""
    emb_cols = sorted(
        [c for c in df.columns if c.startswith("emb_")],
        key=lambda x: int(x.split("_")[1]),
    )

    filename = df["file_path"].str.replace(" copy", "", regex=False)
    df_ts = df.copy()
    df_ts["timestamp_str"] = filename.str.extract(r"(\d{8}_\d{6})")[0]
    df_ts["datetime"] = pd.to_datetime(
        df_ts["timestamp_str"], format="%Y%m%d_%H%M%S", errors="coerce"
    )

    valid_mask = df_ts["datetime"].notna()
    df_valid = df_ts[valid_mask].copy()

    dt = df_valid["datetime"]
    df_valid["time_seconds"] = (
        dt.dt.hour * 3600 + dt.dt.minute * 60 + dt.dt.second
    )
    df_valid["day_of_year"] = dt.dt.dayofyear

    embeddings = df_valid[emb_cols].to_numpy(dtype=np.float32)
    times = df_valid["time_seconds"].to_numpy()
    days = df_valid["day_of_year"].to_numpy()

    metadata_cols = ["file", "file_path", "chunk_idx", "start_time", "end_time"]
    metadata = df_valid[metadata_cols].reset_index(drop=True)

    scores = df_valid["score1"].astype(float).to_numpy()

    return embeddings, times, days, metadata, scores


# ---------------------------------------------------------------------
# Time-window inference and colormap
# ---------------------------------------------------------------------

def infer_time_windows_from_hist(
    times: np.ndarray,
    bin_minutes: int = 10,
    occ_frac_threshold: float = 0.005,
):
    """Infer main morning and evening/night recording windows from time-of-day data."""
    hours = (times / 3600.0) % 24.0

    n_bins = int(24 * 60 / bin_minutes)
    hist, edges = np.histogram(hours, bins=n_bins, range=(0.0, 24.0))

    if hist.max() == 0:
        console.print("  No time data found; falling back to full-day window.")
        occupied = np.zeros(len(hist), dtype=bool)
        return hours, (0.0, 24.0), None, hist, edges, 0, occupied

    occ_threshold = max(1, int(hist.max() * occ_frac_threshold))
    occupied = hist >= occ_threshold

    runs = []
    i = 0
    while i < n_bins:
        if not occupied[i]:
            i += 1
            continue
        start = i
        while i + 1 < n_bins and occupied[i + 1]:
            i += 1
        end = i
        runs.append((start, end))
        i += 1

    if occupied[0] and occupied[-1] and len(runs) > 1:
        _, first_end = runs[0]
        last_start, _ = runs[-1]
        merged = (last_start, first_end)
        runs = [merged] + runs[1:-1]

    if not runs:
        console.print(
            "  No contiguous occupied runs found; falling back to full-day window."
        )
        occupied = hist >= 0
        return hours, (0.0, 24.0), None, hist, edges, occ_threshold, occupied

    segments = []
    for start_bin, end_bin in runs:
        start_h = edges[start_bin]
        end_h = edges[end_bin + 1]
        duration = end_h - start_h
        center = 0.5 * (start_h + end_h)
        segments.append(
            {"start": start_h, "end": end_h, "duration": duration, "center": center}
        )

    segments.sort(key=lambda s: s["duration"], reverse=True)
    if len(segments) == 1:
        morning_win = (segments[0]["start"], segments[0]["end"])
        evening_win = None
        console.print(
            f"  Single recording window: {morning_win[0]:.1f}h – {morning_win[1]:.1f}h"
        )
        return hours, morning_win, evening_win, hist, edges, occ_threshold, occupied

    seg_a, seg_b = segments[0], segments[1]
    if seg_a["center"] <= seg_b["center"]:
        morning_seg, evening_seg = seg_a, seg_b
    else:
        morning_seg, evening_seg = seg_b, seg_a

    morning_win = (morning_seg["start"], morning_seg["end"])
    evening_win = (evening_seg["start"], evening_seg["end"])

    console.print(
        f"  Inferred windows from histogram (threshold {occ_threshold} counts/bin):"
    )
    console.print(f"    Morning (dawn): {morning_win[0]:.1f}h – {morning_win[1]:.1f}h")
    console.print(f"    Evening/night:  {evening_win[0]:.1f}h – {evening_win[1]:.1f}h")

    return hours, morning_win, evening_win, hist, edges, occ_threshold, occupied


def build_24h_colormap(
    hours: np.ndarray,
    morning_win: Tuple[float, float],
    evening_win: Optional[Tuple[float, float]],
    n_colors: int = 288,
):
    """Build a 24h colormap with dawn/dusk colors and transparent gaps."""
    cmap_array = np.ones((n_colors, 4))
    cmap_array[:, :3] = 0.5  # Will be transparent anyway
    cmap_array[:, 3] = 0.15  # Transparent by default

    morning_palette = cm.get_cmap("viridis")
    evening_palette = cm.get_cmap("Reds")

    def hour_from_index(i: int) -> float:
        return 24.0 * i / (n_colors - 1)

    m_start, m_end = morning_win
    e_start, e_end = (evening_win if evening_win is not None else (None, None))

    def in_evening(h: float) -> bool:
        if e_start is None or e_end is None:
            return False
        if e_start <= e_end:
            return e_start <= h <= e_end
        return (h >= e_start) or (h <= e_end)

    morning_span = max(m_end - m_start, 1e-6)
    if evening_win is not None and e_start is not None and e_end is not None:
        if e_start <= e_end:
            evening_span = max(e_end - e_start, 1e-6)
        else:
            evening_span = max((24.0 - e_start) + e_end, 1e-6)
    else:
        evening_span = 1.0

    for i in range(n_colors):
        h = hour_from_index(i)

        if m_start <= h <= m_end:
            frac = (h - m_start) / morning_span
            cmap_array[i] = morning_palette(np.clip(frac, 0.0, 1.0))
        elif evening_win is not None and in_evening(h):
            if e_start is not None and e_end is not None:
                if e_start <= e_end:
                    frac = (h - e_start) / evening_span
                else:
                    if h >= e_start:
                        frac = (h - e_start) / evening_span
                    else:
                        frac = ((24.0 - e_start) + h) / evening_span
                cmap_array[i] = evening_palette(np.clip(frac, 0.0, 1.0))

    cmap = ListedColormap(cmap_array)
    norm_times = hours / 24.0

    tick_hours = [0, 3, 6, 9, 12, 15, 18, 21, 24]
    tick_positions = [h / 24.0 for h in tick_hours]
    tick_labels = [f"{h}h" for h in tick_hours]
    tick_info = list(zip(tick_positions, tick_labels))

    return cmap, norm_times, tick_info


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def create_temporal_plots(
    embeddings_2d: np.ndarray,
    times: np.ndarray,
    days: np.ndarray,
    output_dir: str,
    scores: np.ndarray,
):
    """Create temporal UMAP visualizations with dark gradient background."""
    hours, morning_win, evening_win, *_ = infer_time_windows_from_hist(times)
    time_cmap, norm_times, time_ticks = build_24h_colormap(
        hours, morning_win, evening_win
    )

    fig = plt.figure(figsize=(18, 6))
    fig.patch.set_facecolor('#1a1d21')
    
    # Background gradient with noise to reduce banding
    bg_ax = plt.axes([0, 0, 1, 1], frameon=False)
    bg_ax.set_zorder(-1)
    
    # Vertical gradient with high resolution and noise
    gradient_colors = ['#1a1d21', '#25282c']
    n_grad = 512
    grad_cmap = LinearSegmentedColormap.from_list('bg_gradient', gradient_colors, N=n_grad)
    
    # Create smooth vertical gradient
    gradient = np.linspace(0, 1, 1024).reshape(-1, 1)
    gradient = np.repeat(gradient, 100, axis=1)
    
    # Add subtle noise to reduce banding
    np.random.seed(42)
    noise = np.random.normal(0, 0.003, gradient.shape)
    gradient = np.clip(gradient + noise, 0, 1)
    
    bg_ax.imshow(
        gradient,
        extent=[0, 1, 0, 1],
        transform=bg_ax.transAxes,
        aspect='auto',
        interpolation='bilinear',
        cmap=grad_cmap,
        vmin=0,
        vmax=1,
        zorder=-1
    )
    bg_ax.set_xlim(0, 1)
    bg_ax.set_ylim(0, 1)
    bg_ax.axis('off')
    
    gs = GridSpec(1, 3, figure=fig, wspace=0.25, left=0.05, right=0.95, top=0.95, bottom=0.15)
    canvas_res = 2000

    # Use actual day range (not normalized from 0)
    min_day = int(days.min())
    max_day = int(days.max())
    day_range = max(max_day - min_day, 1)
    norm_days = (days - min_day) / day_range

    min_score = float(scores.min())
    max_score = float(scores.max())

    panels = [
        (norm_times, time_cmap, "Time of day", time_ticks, None),
        (norm_days, cm.get_cmap("viridis"), "Day of year", None, (min_day, max_day)),
        (scores, cm.get_cmap("plasma"), "Top logit score", None, (min_score, max_score)),
    ]

    for idx, (values, cmap, label, ticks, val_range) in enumerate(panels):
        ax = fig.add_subplot(gs[idx])
        ax.patch.set_alpha(0.0)

        df_plot = pd.DataFrame(
            {"x": embeddings_2d[:, 0], "y": embeddings_2d[:, 1], "c": values}
        )

        canvas = ds.Canvas(plot_width=canvas_res, plot_height=canvas_res)
        agg = canvas.points(df_plot, "x", "y", agg=ds.mean("c"))
        img = tf.shade(agg, cmap=cmap, how="linear")
        img = tf.spread(img, px=2)

        extent = (
            agg.x_range[0],
            agg.x_range[1],
            agg.y_range[0],
            agg.y_range[1],
        )

        ax.imshow(
            img.to_pil(),
            extent=extent,
            origin="lower",
            aspect="auto",
        )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Create colorbar with proper normalization
        if val_range:
            norm = Normalize(vmin=val_range[0], vmax=val_range[1])
        else:
            norm = Normalize(vmin=0.0, vmax=1.0)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = plt.colorbar(
            sm, 
            ax=ax, 
            orientation="horizontal", 
            fraction=0.015,  # Narrower
            pad=0.06,
            aspect=30
        )
        
        cbar.set_label(label, fontsize=9, color=CBAR_FG, labelpad=4)
        cbar.ax.set_facecolor('#25282c')
        
        # Remove outline
        cbar.outline.set_visible(False)

        if ticks:
            pos, lab = zip(*ticks)
            cbar.set_ticks(pos)
            cbar.set_ticklabels(lab, fontsize=7)
        elif idx == 1:
            # Day of year: show actual day values
            day_ticks_norm = np.linspace(0.0, 1.0, 5)
            day_labels = [f"{int(min_day + t * day_range)}" for t in day_ticks_norm]
            cbar.set_ticks(list(day_ticks_norm * day_range + min_day))
            cbar.set_ticklabels(day_labels)

        # Remove tick lines, keep labels
        cbar.ax.tick_params(
            labelsize=8, 
            colors=CBAR_FG, 
            length=0,  # No tick lines
            pad=3
        )

    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(
        f"{output_dir}/temporal_embeddings.png",
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.1,
        facecolor='#1a1d21',
        edgecolor='none',
    )
    plt.close()

    console.print(f"✓ Visualizations saved: {len(embeddings_2d):,} points")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

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
    config_table.add_row("Workers", str(N_WORKERS))
    console.print(config_table)

    console.print(Panel.fit("→ Extracting temporal features...", border_style="cyan"))
    embeddings, times, days, metadata, scores = extract_temporal_features(df)
    console.print(f"  Processed {len(embeddings):,} valid samples")
    console.print(
        f"  Time: {times.min() / 3600:.1f}h – {times.max() / 3600:.1f}h, "
        f"Days: {days.min()} – {days.max()}"
    )

    # Filter to days with sufficient data (>500 embeddings)
    day_counts = pd.Series(days).value_counts().sort_index()
    threshold = 500
    valid_days = day_counts[day_counts > threshold].index
    if not valid_days.empty:
        first_valid_day = valid_days.min()
        mask = days >= first_valid_day
        embeddings = embeddings[mask]
        times = times[mask]
        days = days[mask]
        metadata = metadata[mask]
        scores = scores[mask]
        console.print(f"  Filtered to days >= {first_valid_day}, {len(embeddings):,} samples remaining")
    else:
        console.print("  No days with >500 embeddings found, using all data")

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

    console.print(Panel.fit("→ Creating visualizations...", border_style="cyan"))
    create_temporal_plots(embeddings_2d, times, days, OUTPUT_DIR, scores)

    results = metadata.copy()
    results["umap_x"] = embeddings_2d[:, 0]
    results["umap_y"] = embeddings_2d[:, 1]
    results["time_seconds"] = times
    results["day_of_year"] = days
    results.to_parquet(f"{OUTPUT_DIR}/umap_embeddings.parquet", index=False)

    console.print(
        Panel.fit(f"\n✓ Complete! Output: {OUTPUT_DIR}/", border_style="green")
    )


if __name__ == "__main__":
    main()
