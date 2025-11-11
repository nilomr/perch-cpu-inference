#!/usr/bin/env python3
"""
Visualize model output for a single file.
"""

import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.collections import LineCollection
import librosa
from opensoundscape import Audio


def load_predictions(csv_path):
    return pd.read_csv(csv_path)


def parse_chunk_idx(filename_stemmed):
    m = re.search(r"_(\d+)$", filename_stemmed)
    return int(m.group(1)) if m else 0


def logits_to_relative_confidence(scores):
    """
    Convert logits to relative confidence scores.
    Since we only have top-5 logits (not full distribution), we can't compute
    true probabilities. Instead, show relative strength as percentage of max.
    """
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0 or not np.any(np.isfinite(scores)):
        return np.array([])

    # Normalize to 0-100 scale based on relative logit values
    # The difference in logits tells us relative confidence
    max_score = np.max(scores)
    min_score = np.min(scores)

    if max_score == min_score:
        return np.ones_like(scores) * 50.0

    # Map linearly: highest logit = 100%, lowest in top-5 = scaled down
    # This preserves the relative differences in the logits
    normalized = ((scores - min_score) / (max_score - min_score)) * 100.0

    return normalized


def get_confidence_color(conf):
    """Return color based on confidence level with professional palette."""
    if conf >= 75:
        return "#10b981"  # Emerald green
    elif conf >= 50:
        return "#f59e0b"  # Amber
    elif conf >= 25:
        return "#ef4444"  # Red
    else:
        return "#6b7280"  # Gray


def truncate_species_name(name, max_len=22):
    """Smart truncation of species names."""
    if len(name) <= max_len:
        return name
    # Try to truncate at word boundary
    truncated = name[: max_len - 3]
    last_space = truncated.rfind(" ")
    if last_space > max_len * 0.6:
        return truncated[:last_space] + "..."
    return truncated + "..."


def create_spectrogram(audio_path, max_freq=12000):
    """Generate clean spectrogram optimized for bird vocalizations."""
    # Load audio efficiently
    audio = Audio.from_file(audio_path, sample_rate=32000)
    y = audio.samples
    sr = audio.sample_rate
    duration = len(y) / sr

    # Optimized STFT parameters for speed and clarity
    n_fft = 2048
    hop_length = 512

    # Compute STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Frequency masking for relevant range
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    freq_mask = freqs <= max_freq

    return S_db[freq_mask, :], freqs[freq_mask], duration, sr, hop_length


def extract_predictions(predictions_df, audio_stem, duration):
    """Extract and organize predictions for the audio file."""
    chunk_duration = 5.0
    preds = {}

    for _, row in predictions_df.iterrows():
        file_field = str(row.get("file", ""))
        file_path_field = str(row.get("file_path", ""))

        # Match audio file
        matched = False
        if file_path_field:
            try:
                if Path(file_path_field).stem == audio_stem:
                    matched = True
            except Exception:
                pass

        if not matched and file_field:
            m = re.match(r"^(.*?)(?:_(\d+))?$", file_field)
            prefix = m.group(1) if m else file_field
            if prefix == audio_stem or file_field.startswith(audio_stem):
                matched = True

        if not matched:
            continue

        idx = parse_chunk_idx(file_field)
        start_time = row.get("start_time", idx * chunk_duration)
        end_time = row.get("end_time", min((idx + 1) * chunk_duration, duration))

        # Extract top 5 scores and classes
        scores = [row.get(f"score{i + 1}", np.nan) for i in range(5)]
        confs = logits_to_relative_confidence(scores)
        classes = [str(row.get(f"top{i + 1}", "")) for i in range(5)]

        # Store valid predictions only
        valid_preds = [
            (cls, conf)
            for cls, conf in zip(classes, confs)
            if cls and cls != "nan" and np.isfinite(conf)
        ]

        if valid_preds:
            preds[idx] = (start_time, end_time, valid_preds[:3])

    return preds


def plot_visualization(audio_path, predictions_df, output_path=None):
    """Create reasonably ok visualization."""

    print(f"Processing {audio_path.name}...")

    # Generate spectrogram
    S_db, freqs, duration, sr, hop_length = create_spectrogram(audio_path)
    times = librosa.frames_to_time(
        np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length
    )

    # Extract predictions
    preds = extract_predictions(predictions_df, audio_path.stem, duration)

    if not preds:
        print(f"No predictions found for {audio_path.name}")
        return

    # Adaptive figure sizing: wider for longer recordings
    width = max(16, min(30, duration / 3.0))
    height = 10

    fig = plt.figure(figsize=(width, height), facecolor="white")

    # Create layout: predictions table on top, spectrogram below
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 2.0], hspace=0.15)

    # Top panel: Prediction table
    ax_preds = fig.add_subplot(gs[0])
    # Bottom panel: Spectrogram
    ax_spec = fig.add_subplot(gs[1])

    # ===== SPECTROGRAM =====
    vmin, vmax = np.percentile(S_db, [5, 99])
    ax_spec.pcolormesh(
        times,
        freqs,
        S_db,
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        shading="gouraud",
        rasterized=True,
    )

    ax_spec.set_ylabel("Frequency (kHz)", fontsize=11, fontweight="500")
    ax_spec.set_xlabel("Time (seconds)", fontsize=11, fontweight="500")
    ax_spec.set_yticks(np.arange(0, 13000, 2000))
    ax_spec.set_yticklabels([f"{f / 1000:.0f}" for f in np.arange(0, 13000, 2000)])
    ax_spec.tick_params(axis="both", labelsize=9)
    ax_spec.set_ylim(0, freqs[-1])
    ax_spec.set_xlim(0, duration)

    # ===== PREDICTIONS TABLE =====
    ax_preds.set_xlim(0, duration)
    ax_preds.set_ylim(0, 3.5)
    ax_preds.axis("off")

    chunk_duration = 5.0
    n_chunks = int(np.ceil(duration / chunk_duration))

    # Table headers
    ax_preds.text(
        -0.5, 3.2, "Rank", fontsize=9, fontweight="700", color="#374151", ha="right"
    )

    for idx in range(n_chunks):
        start_t = idx * chunk_duration
        end_t = min((idx + 1) * chunk_duration, duration)
        mid_t = (start_t + end_t) / 2

        # Column header: time range
        ax_preds.text(
            mid_t,
            3.2,
            f"{int(start_t)}-{int(end_t)}s",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="700",
            color="#374151",
        )

        # Column separator
        if idx < n_chunks - 1:
            ax_preds.axvline(end_t, color="#e5e7eb", linewidth=1.5, zorder=1)

        # Draw predictions for this chunk
        if idx in preds:
            _, _, pred_list = preds[idx]

            for rank, (species, conf) in enumerate(pred_list):
                y_pos = 2.4 - rank * 0.9
                color = get_confidence_color(conf)

                # Species name (truncated if needed)
                species_label = truncate_species_name(species, 22)

                # Background box for better readability
                box_width = end_t - start_t
                box = FancyBboxPatch(
                    (start_t, y_pos - 0.35),
                    box_width,
                    0.7,
                    boxstyle="round,pad=0.05",
                    facecolor="white" if rank == 0 else "#f9fafb",
                    edgecolor=color if rank == 0 else "#e5e7eb",
                    linewidth=2 if rank == 0 else 1,
                    alpha=0.95,
                    zorder=5,
                )
                ax_preds.add_patch(box)

                # Species name
                ax_preds.text(
                    mid_t,
                    y_pos + 0.08,
                    species_label,
                    ha="center",
                    va="center",
                    fontsize=8 if rank == 0 else 7.5,
                    fontweight="600" if rank == 0 else "500",
                    color="#111827" if rank == 0 else "#4b5563",
                    zorder=10,
                )

                # Confidence percentage
                ax_preds.text(
                    mid_t,
                    y_pos - 0.15,
                    f"{conf:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8 if rank == 0 else 7,
                    fontweight="700" if rank == 0 else "600",
                    color=color,
                    zorder=10,
                )
        else:
            # No detection - show placeholder
            y_pos = 1.5
            ax_preds.text(
                mid_t,
                y_pos,
                "No detection",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="500",
                color="#9ca3af",
                style="italic",
                zorder=10,
            )

    # Row labels (1st, 2nd, 3rd)
    for rank in range(3):
        y_pos = 2.4 - rank * 0.9
        label = ["1st", "2nd", "3rd"][rank]
        ax_preds.text(
            -0.5,
            y_pos,
            label,
            ha="right",
            va="center",
            fontsize=8,
            fontweight="600",
            color="#6b7280",
            zorder=10,
        )

    # Horizontal separators between rows
    for rank in range(1, 3):
        y_pos = 2.4 - rank * 0.9 + 0.45
        ax_preds.axhline(y_pos, color="#f3f4f6", linewidth=1, zorder=1)

    # ===== TITLE AND LEGEND =====
    fig.suptitle(
        f"{audio_path.stem}",
        fontsize=15,
        fontweight="700",
        color="#111827",
        x=0.5,
        y=0.98,
        ha="center",
    )

    # Add legend to spectrogram panel
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#10b981", label="High (≥75%)"),
        Patch(facecolor="#f59e0b", label="Medium (50-74%)"),
        Patch(facecolor="#ef4444", label="Low (25-49%)"),
        Patch(facecolor="#6b7280", label="Very Low (<25%)"),
    ]
    ax_spec.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=8,
        framealpha=0.95,
        title="Confidence",
        title_fontsize=8,
        edgecolor="#e5e7eb",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"✓ Saved: {output_path}")
        print(
            f"  Duration: {duration:.1f}s | Predictions: {len(preds)} chunks | Size: {width:.1f}×{height} in"
        )
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Perch v2 output visualization")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("predictions_csv", help="Path to predictions CSV")
    parser.add_argument("--output", help="Output image path (optional)")

    args = parser.parse_args()

    audio_path = Path(args.audio_file)
    csv_path = Path(args.predictions_csv)

    if not audio_path.exists():
        print(f"✗ Audio file not found: {audio_path}")
        return
    if not csv_path.exists():
        print(f"✗ Predictions CSV not found: {csv_path}")
        return

    predictions = load_predictions(csv_path)
    plot_visualization(audio_path, predictions, args.output)


if __name__ == "__main__":
    main()
