"""
Perch v2 ONNX - Production version - parallel inference on CPU
"""

from pathlib import Path
import json
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import warnings
import os
import tempfile
import shutil
from typing import Dict
from datetime import datetime

import numpy as np
import pandas as pd
import typer
from contexttimer import Timer
import soundfile as sf

try:
    import onnxruntime as ort
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
    from rich.console import Console
    from rich.progress import (
        Progress,
        SpinnerColumn,
        BarColumn,
        TextColumn,
        TimeRemainingColumn,
        TimeElapsedColumn,
        MofNCompleteColumn,
    )
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
except ImportError:
    raise ImportError("Install: pip install onnxruntime pyarrow rich")

app = typer.Typer(no_args_is_help=True)
console = Console()

SAMPLE_RATE = 32000
CHUNK_SIZE = 160000  # 5 seconds
CHECKPOINT_INTERVAL = 100  # Save every N batches

warnings.filterwarnings("ignore")

_session = None


def atomic_write_json(filepath: Path, data: dict):
    """Atomically write JSON to prevent corruption."""
    temp_fd, temp_path = tempfile.mkstemp(
        dir=filepath.parent, prefix=".tmp_", suffix=".json"
    )
    try:
        with os.fdopen(temp_fd, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        shutil.move(temp_path, filepath)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def load_checkpoint(checkpoint_path: Path) -> Dict:
    """Load checkpoint and return processed file info."""
    if not checkpoint_path.exists():
        return {"processed_files": set(), "total_batches": 0, "checkpoint_id": 0}

    try:
        with open(checkpoint_path, "r") as f:
            data = json.load(f)
            processed = set(data.get("processed_files", []))
            batches = data.get("total_batches", 0)
            checkpoint_id = data.get("checkpoint_id", 0)
            if processed:
                console.print(
                    f"[yellow]Checkpoint: {len(processed)} files processed "
                    f"({batches} batches, checkpoint_id={checkpoint_id})[/yellow]"
                )
            return {
                "processed_files": processed,
                "total_batches": batches,
                "checkpoint_id": checkpoint_id,
            }
    except (json.JSONDecodeError, IOError):
        console.print("[yellow]Warning: Invalid checkpoint, starting fresh[/yellow]")
        return {"processed_files": set(), "total_batches": 0, "checkpoint_id": 0}


def save_checkpoint(
    checkpoint_path: Path, processed_files: set, total_batches: int, checkpoint_id: int
):
    """Save checkpoint atomically."""
    data = {
        "processed_files": sorted(list(processed_files)),
        "total_processed": len(processed_files),
        "total_batches": total_batches,
        "checkpoint_id": checkpoint_id,
        "last_updated": datetime.now().isoformat(),
    }
    atomic_write_json(checkpoint_path, data)


def write_embeddings_partitioned(
    output_dir: Path, df: pd.DataFrame, checkpoint_id: int, use_float16: bool = False
):
    """
    Write embeddings to partitioned parquet dataset.
    Files are partitioned by checkpoint_id to keep files manageable.
    """
    emb_array = np.stack(df["embeddings"].values)

    # Optional: quantize to float16 for 50% size reduction
    if use_float16:
        emb_array = emb_array.astype(np.float16)

    # Create dataframe with embeddings
    emb_df = df[["file", "file_path", "chunk_idx", "start_time", "end_time"]].copy()
    emb_cols = {f"emb_{i}": emb_array[:, i] for i in range(emb_array.shape[1])}
    emb_df = emb_df.assign(**emb_cols)
    emb_df["checkpoint_id"] = checkpoint_id

    # Convert to PyArrow Table
    table = pa.Table.from_pandas(emb_df)

    # Write to partitioned dataset with proper format specification
    pq_format = ds.ParquetFileFormat()
    ds.write_dataset(
        table,
        output_dir / "embeddings_partitioned",
        format=pq_format,
        partitioning=["checkpoint_id"],
        existing_data_behavior="overwrite_or_ignore",
        file_options=pq_format.make_write_options(compression="snappy"),
    )


def write_predictions_partitioned(
    output_dir: Path, df: pd.DataFrame, classes: list, checkpoint_id: int
):
    """Write predictions to partitioned dataset (CSV for easy inspection)."""
    top10_rows = []
    for _, row in df.iterrows():
        top10_indices = np.argsort(row["logits"])[-10:][::-1]
        top10_rows.append(
            {
                "file": row["file"],
                "file_path": row["file_path"],
                "chunk_idx": row["chunk_idx"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "checkpoint_id": checkpoint_id,
                **{f"top{i + 1}": classes[idx] for i, idx in enumerate(top10_indices)},
                **{
                    f"score{i + 1}": float(row["logits"][idx])
                    for i, idx in enumerate(top10_indices)
                },
            }
        )

    pred_df = pd.DataFrame(top10_rows)

    # Save to checkpoint-specific CSV file
    pred_dir = output_dir / "predictions_partitioned"
    pred_dir.mkdir(exist_ok=True)
    pred_file = pred_dir / f"checkpoint_{checkpoint_id:04d}.csv"
    pred_df.to_csv(pred_file, index=False)


def init_worker(model_path: str, num_threads: int):
    """Initialize optimized ONNX session per worker."""
    global _session

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = num_threads
    sess_options.inter_op_num_threads = 1
    sess_options.enable_mem_pattern = True
    sess_options.enable_cpu_mem_arena = True
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    _session = ort.InferenceSession(
        model_path, sess_options, providers=["CPUExecutionProvider"]
    )


def load_single_file(file_path: str):
    """Load and chunk a single file."""
    try:
        y, sr = sf.read(file_path, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa

            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)

        chunks = []
        duration = len(y) / SAMPLE_RATE

        for i in range(0, len(y), CHUNK_SIZE):
            chunk_idx = i // CHUNK_SIZE
            start_time = chunk_idx * 5.0
            end_time = min((chunk_idx + 1) * 5.0, duration)
            chunks.append(
                (file_path, chunk_idx, y[i : i + CHUNK_SIZE], start_time, end_time)
            )

        return chunks
    except Exception:
        return []


def process_batch(batch_chunks: list) -> list:
    """Process a batch through ONNX."""
    audio_batch, metadata = [], []

    for file_path, chunk_idx, chunk, start_time, end_time in batch_chunks:
        if len(chunk) < CHUNK_SIZE:
            chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)))

        max_val = np.abs(chunk).max()
        if max_val > 1e-10:
            chunk = chunk / max_val * 0.25

        audio_batch.append(chunk)
        metadata.append(
            {
                "file_path": file_path,
                "chunk_idx": chunk_idx,
                "start_time": start_time,
                "end_time": end_time,
            }
        )

    input_name = _session.get_inputs()[0].name
    emb_name = next(o.name for o in _session.get_outputs() if o.name == "embedding")
    logit_name = next(o.name for o in _session.get_outputs() if o.name == "label")

    embeddings, logits = _session.run(
        [emb_name, logit_name], {input_name: np.array(audio_batch, dtype=np.float32)}
    )

    return [
        {"embeddings": emb, "logits": log, **meta}
        for emb, log, meta in zip(embeddings, logits, metadata)
    ]


@app.callback(invoke_without_command=True)
def main(
    audio_dir: Path = typer.Option(..., "--audio-dir", "-d", help="Audio directory"),
    model_path: Path = typer.Option(
        "./models/perch_v2/perch_v2.onnx", help="ONNX model"
    ),
    classes_json: Path = typer.Option(
        "./models/perch_v2/classes.json", help="Classes JSON"
    ),
    output_dir: Path = typer.Option("./output-onnx", help="Output directory"),
    batch_size: int = typer.Option(16, "--batch-size", "-b", help="Batch size"),
    workers: int = typer.Option(4, "--workers", "-w", help="Inference workers"),
    loader_threads: int = typer.Option(
        8, "--loader-threads", "-l", help="File loader threads"
    ),
    checkpoint_interval: int = typer.Option(
        CHECKPOINT_INTERVAL,
        "--checkpoint-interval",
        "-c",
        help="Save results every N batches",
    ),
    use_float16: bool = typer.Option(
        False, "--float16/--float32", help="Use float16 for embeddings (50% smaller)"
    ),
    resume: bool = typer.Option(
        True, "--resume/--no-resume", help="Resume from checkpoint"
    ),
):
    """
    High-performance ONNX inference with optimized storage

    - Partitioned parquet datasets (efficient for massive datasets)
    - Optional float16 compression (50% size reduction)
    - Top 10 species predictions
    - Incremental checkpointing

    Storage improvements:
    - Embeddings: Partitioned parquet (no CSV)
    - Predictions: Partitioned CSV files
    - Target file size: ~500MB per partition

    Example:
        python script.py --audio-dir ./audio --output-dir ./output --workers 16 --float16
    """
    output_dir.mkdir(exist_ok=True)
    checkpoint_path = output_dir / ".checkpoint.json"

    # Load checkpoint
    checkpoint_data = (
        load_checkpoint(checkpoint_path)
        if resume
        else {"processed_files": set(), "total_batches": 0, "checkpoint_id": 0}
    )
    processed_files = checkpoint_data["processed_files"]
    current_checkpoint_id = checkpoint_data["checkpoint_id"]

    # Find all files
    all_files = sorted([str(f) for p in ["*.wav", "*.WAV"] for f in audio_dir.glob(p)])
    if not all_files:
        console.print("[red]x No files found[/red]")
        return

    # Filter unprocessed files
    files = [f for f in all_files if f not in processed_files]

    if not files:
        console.print("[green]v All files already processed![/green]")
        return

    if processed_files:
        console.print(
            f"[green]v Resuming: {len(processed_files)} done, {len(files)} remaining[/green]"
        )

    # Config
    total_cpus = os.cpu_count() or 8
    threads_per_worker = max(1, total_cpus // workers)

    config_table = Table(title="Configuration", box=box.ROUNDED, show_header=False)
    config_table.add_column("Setting", style="cyan", width=30)
    config_table.add_column("Value", style="green bold")
    config_table.add_row("Total Files", f"{len(all_files):,}")
    config_table.add_row("Already Processed", f"{len(processed_files):,}")
    config_table.add_row("To Process", f"{len(files):,}")
    config_table.add_row("Loader Threads", str(loader_threads))
    config_table.add_row("Inference Workers", str(workers))
    config_table.add_row("Batch Size", str(batch_size))
    config_table.add_row("Checkpoint Interval", f"Every {checkpoint_interval} batches")
    config_table.add_row("Embedding Format", "float16" if use_float16 else "float32")
    config_table.add_row("Storage Format", "Partitioned Parquet")
    config_table.add_row("Top Species", "10")
    config_table.add_row("CPUs", str(total_cpus))
    console.print(config_table)
    console.print()

    # Storage info
    console.print(
        Panel.fit(
            "[bold cyan]Storage Optimization[/bold cyan]\n"
            f"[dim]Embeddings: Partitioned by checkpoint_id (parquet only)[/dim]\n"
            f"[dim]Predictions: Partitioned CSV files[/dim]\n"
            f"[dim]Compression: {'float16 (50% smaller)' if use_float16 else 'float32 (standard)'}[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    # STEP 1: Parallel file loading
    console.print(
        Panel.fit(
            f"[bold cyan]Step 1: Loading {len(files):,} files[/bold cyan]",
            border_style="cyan",
        )
    )

    all_chunks = []
    file_to_chunks = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Loading...", total=len(files))

        with ThreadPoolExecutor(max_workers=loader_threads) as executor:
            future_to_file = {executor.submit(load_single_file, f): f for f in files}

            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                chunks = future.result()

                if chunks:
                    start_idx = len(all_chunks)
                    all_chunks.extend(chunks)
                    end_idx = len(all_chunks)
                    file_to_chunks[file_path] = (start_idx, end_idx)

                progress.update(task, advance=1)

    console.print(f"[green]v[/green] Loaded {len(all_chunks):,} chunks\n")

    if not all_chunks:
        console.print("[red]x No chunks to process[/red]")
        return

    # STEP 2: Create batches
    batches = [
        all_chunks[i : i + batch_size] for i in range(0, len(all_chunks), batch_size)
    ]

    console.print(
        Panel.fit(
            f"[bold green]Step 2: Running inference ({len(batches):,} batches)[/bold green]\n"
            f"[dim]{workers} workers, checkpoints every {checkpoint_interval} batches[/dim]",
            border_style="green",
        )
    )
    console.print()

    # Load class names
    classes = json.load(open(classes_json))

    # STEP 3: Parallel inference with optimized storage
    init_fn = partial(init_worker, str(model_path), threads_per_worker)
    chunksize = max(1, len(batches) // (workers * 4))

    batch_buffer = []
    processed_chunk_indices = set()
    batch_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[bold green]Processing ({workers} workers)", total=len(batches)
        )

        with Timer() as t, mp.Pool(workers, initializer=init_fn) as pool:
            for batch_results in pool.imap(process_batch, batches, chunksize=chunksize):
                batch_buffer.extend(batch_results)
                batch_count += 1

                for res in batch_results:
                    processed_chunk_indices.add((res["file_path"], res["chunk_idx"]))

                progress.update(task, advance=1)

                # Save incrementally
                if batch_count % checkpoint_interval == 0 and batch_buffer:
                    rows = [
                        {
                            "file": f"{Path(res['file_path']).stem}_{res['chunk_idx']}",
                            **res,
                        }
                        for res in batch_buffer
                    ]
                    df = pd.DataFrame(rows)

                    # Write embeddings to partitioned dataset
                    write_embeddings_partitioned(
                        output_dir, df, current_checkpoint_id, use_float16
                    )

                    # Write predictions to partitioned files
                    write_predictions_partitioned(
                        output_dir, df, classes, current_checkpoint_id
                    )

                    # Determine completed files
                    newly_completed = set()
                    for file_path, (start_idx, end_idx) in file_to_chunks.items():
                        num_chunks = end_idx - start_idx
                        completed = sum(
                            1 for fp, _ in processed_chunk_indices if fp == file_path
                        )
                        if completed == num_chunks:
                            newly_completed.add(file_path)

                    if newly_completed:
                        processed_files.update(newly_completed)

                    # Save checkpoint
                    current_checkpoint_id += 1
                    save_checkpoint(
                        checkpoint_path,
                        processed_files,
                        batch_count,
                        current_checkpoint_id,
                    )

                    batch_buffer = []

                    console.print(
                        f"\n[dim]Checkpoint {current_checkpoint_id}: "
                        f"{len(processed_files)}/{len(all_files)} files, "
                        f"{batch_count}/{len(batches)} batches[/dim]"
                    )

    # Save remaining buffer
    if batch_buffer:
        rows = [
            {"file": f"{Path(res['file_path']).stem}_{res['chunk_idx']}", **res}
            for res in batch_buffer
        ]
        df = pd.DataFrame(rows)

        write_embeddings_partitioned(output_dir, df, current_checkpoint_id, use_float16)
        write_predictions_partitioned(output_dir, df, classes, current_checkpoint_id)

    # Final checkpoint
    processed_files.update(files)
    save_checkpoint(
        checkpoint_path, processed_files, batch_count, current_checkpoint_id
    )

    console.print(f"\n[cyan]v Results saved to partitioned datasets[/cyan]")

    # Show storage info
    emb_dir = output_dir / "embeddings_partitioned"
    pred_dir = output_dir / "predictions_partitioned"

    if emb_dir.exists():
        emb_size = sum(f.stat().st_size for f in emb_dir.rglob("*.parquet"))
        console.print(f"[dim]Embeddings size: {emb_size / 1e9:.2f} GB[/dim]")

    if pred_dir.exists():
        pred_size = sum(f.stat().st_size for f in pred_dir.rglob("*.csv"))
        console.print(f"[dim]Predictions size: {pred_size / 1e6:.2f} MB[/dim]")

    # Clean up if done
    if len(processed_files) == len(all_files):
        console.print("[green]v All complete! Removing checkpoint[/green]")
        try:
            checkpoint_path.unlink()
        except OSError:
            pass

    # Results
    num_files = len(files)
    num_chunks = len(all_chunks)
    speed = num_files * 60 / t.elapsed

    results_table = Table(
        title="Performance Results", box=box.DOUBLE_EDGE, title_style="bold magenta"
    )
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green bold", justify="right")

    results_table.add_row("Files Processed", f"{num_files:,}")
    results_table.add_row("Total Completed", f"{len(processed_files):,}")
    results_table.add_row("Chunks Processed", f"{num_chunks:,}")
    results_table.add_row("Processing Time", f"{t.elapsed:.2f}s")
    results_table.add_row("Speed", f"{speed:.1f}x realtime")
    results_table.add_row("Throughput", f"{num_files / t.elapsed:.2f} files/sec")
    results_table.add_row("Partitions Created", str(current_checkpoint_id))

    console.print(results_table)

    # Projection
    files_in_10tb = 10_000_000
    hours_for_10tb = (files_in_10tb / (num_files / t.elapsed)) / 3600
    days_for_10tb = hours_for_10tb / 24

    console.print()
    projection_panel = Panel(
        f"[yellow]10TB Dataset (~10M files):[/yellow]\n"
        f"[bold white]{hours_for_10tb:.1f} hours ({days_for_10tb:.2f} days)[/bold white]",
        title="Projection",
        border_style="yellow",
    )
    console.print(projection_panel)

    console.print(
        f"\n[dim]Reading embeddings: dataset = pq.ParquetDataset('{emb_dir}')[/dim]"
    )


if __name__ == "__main__":
    app()
