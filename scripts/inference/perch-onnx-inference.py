"""
Perch v2 ONNX - Production version - Memory-aware parallel inference on CPU
"""

import sys
import os


# CRITICAL: Parse max_cpus BEFORE any other imports
def _parse_max_cpus():
    """Extract --max-cpus from command line before imports."""
    for i, arg in enumerate(sys.argv):
        if arg == "--max-cpus" and i + 1 < len(sys.argv):
            try:
                return int(sys.argv[i + 1])
            except ValueError:
                pass
    return None


# Set thread limits BEFORE importing numpy/onnx
_max_cpus_override = _parse_max_cpus()
if _max_cpus_override is not None:
    os.environ["OMP_NUM_THREADS"] = str(_max_cpus_override)
    os.environ["MKL_NUM_THREADS"] = str(_max_cpus_override)
    os.environ["OPENBLAS_NUM_THREADS"] = str(_max_cpus_override)
    os.environ["NUMEXPR_NUM_THREADS"] = str(_max_cpus_override)

# NOW import everything else
from pathlib import Path
import json
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import warnings
import tempfile
import shutil
from typing import Dict, List, Tuple
from datetime import datetime
import gc

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
CHUNK_MEMORY_MB = (CHUNK_SIZE * 4) / (1024 * 1024)  # float32 = 4 bytes per sample
MEMORY_OVERHEAD_FACTOR = 1.5  # Account for metadata, copies, etc.

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
    """Write embeddings to partitioned parquet dataset."""
    emb_array = np.stack(df["embeddings"].values)

    if use_float16:
        emb_array = emb_array.astype(np.float16)

    emb_df = df[["file", "file_path", "chunk_idx", "start_time", "end_time"]].copy()
    emb_cols = {f"emb_{i}": emb_array[:, i] for i in range(emb_array.shape[1])}
    emb_df = emb_df.assign(**emb_cols)
    emb_df["checkpoint_id"] = checkpoint_id

    table = pa.Table.from_pandas(emb_df)

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
    """Write predictions to partitioned dataset."""
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


def estimate_avg_chunks_per_file(
    sample_files: List[str], max_samples: int = 10
) -> float:
    """Estimate average chunks per file from a sample."""
    total_chunks = 0
    valid_files = 0

    for file_path in sample_files[:max_samples]:
        try:
            info = sf.info(file_path)
            duration = info.duration
            chunks = int(np.ceil(duration / 5.0))  # 5 seconds per chunk
            total_chunks += chunks
            valid_files += 1
        except Exception:
            continue

    if valid_files == 0:
        return 12  # Default: assume 60-second files = 12 chunks

    return total_chunks / valid_files


def calculate_file_group_size(max_ram_gb: float, avg_chunks_per_file: float) -> int:
    """Calculate how many files can fit in RAM budget."""
    max_ram_mb = max_ram_gb * 1024
    memory_per_chunk_mb = CHUNK_MEMORY_MB * MEMORY_OVERHEAD_FACTOR
    memory_per_file_mb = memory_per_chunk_mb * avg_chunks_per_file

    files_per_group = int(max_ram_mb / memory_per_file_mb)

    # Safety margin: use 80% of calculated capacity
    files_per_group = max(1, int(files_per_group * 0.8))

    return files_per_group


@app.callback(invoke_without_command=True)
def main(
    audio_dir: Path = typer.Option(..., "--audio-dir", "-d", help="Audio directory"),
    model_path: Path = typer.Option("models/perch_v2/perch_v2.onnx", help="ONNX model"),
    classes_json: Path = typer.Option(
        "models/perch_v2/classes.json", help="Classes JSON"
    ),
    output_dir: Path = typer.Option("output", help="Output directory"),
    batch_size: int = typer.Option(16, "--batch-size", "-b", help="Batch size"),
    workers: int = typer.Option(4, "--workers", "-w", help="Inference workers"),
    loader_threads: int = typer.Option(
        8, "--loader-threads", "-l", help="File loader threads"
    ),
    checkpoint_interval: int = typer.Option(
        200, "--checkpoint-interval", "-c", help="Save results every N batches"
    ),
    use_float16: bool = typer.Option(
        False, "--float16/--float32", help="Use float16 for embeddings"
    ),
    resume: bool = typer.Option(
        True, "--resume/--no-resume", help="Resume from checkpoint"
    ),
    max_cpus: int = typer.Option(
        None, "--max-cpus", help="Maximum total CPUs to use (default: all available)"
    ),
    max_ram_gb: float = typer.Option(
        20.0, "--max-ram-gb", help="Maximum RAM budget in GB (default: 20)"
    ),
):
    """
    Memory-aware ONNX inference with CPU control.

    Example:
        # Use 20GB RAM, 16 CPUs total
        python scripts/inference/perch-onnx-inference.py --audio-dir ./audio --workers 4 --max-cpus 16 --max-ram-gb 20

        # Use 50GB RAM, all CPUs
        python scripts/inference/perch-onnx-inference.py --audio-dir ./audio --workers 8 --max-ram-gb 50
    """
    output_dir.mkdir(exist_ok=True)
    checkpoint_path = output_dir / ".checkpoint.json"

    checkpoint_data = (
        load_checkpoint(checkpoint_path)
        if resume
        else {"processed_files": set(), "total_batches": 0, "checkpoint_id": 0}
    )
    processed_files = checkpoint_data["processed_files"]
    current_checkpoint_id = checkpoint_data["checkpoint_id"]
    total_batches = checkpoint_data["total_batches"]

    all_files = sorted([str(f) for p in ["*.wav", "*.WAV"] for f in audio_dir.glob(p)])
    if not all_files:
        console.print("[red]x No files found[/red]")
        return

    files = [f for f in all_files if f not in processed_files]

    if not files:
        console.print("[green]v All files already processed![/green]")
        return

    if processed_files:
        console.print(
            f"[green]v Resuming: {len(processed_files)} done, {len(files)} remaining[/green]"
        )

    # Estimate memory requirements
    console.print("[cyan]Estimating memory requirements...[/cyan]")
    avg_chunks = estimate_avg_chunks_per_file(files, max_samples=min(10, len(files)))
    files_per_group = calculate_file_group_size(max_ram_gb, avg_chunks)

    # CPU configuration
    system_cpus = os.cpu_count() or 8
    total_cpus = max_cpus if max_cpus is not None else system_cpus
    total_cpus = min(total_cpus, system_cpus)
    threads_per_worker = max(1, total_cpus // workers)

    # Calculate file groups
    file_groups = [
        files[i : i + files_per_group] for i in range(0, len(files), files_per_group)
    ]

    config_table = Table(title="Configuration", box=box.ROUNDED, show_header=False)
    config_table.add_column("Setting", style="cyan", width=30)
    config_table.add_column("Value", style="green bold")
    config_table.add_row("Total Files", f"{len(all_files):,}")
    config_table.add_row("Already Processed", f"{len(processed_files):,}")
    config_table.add_row("To Process", f"{len(files):,}")
    config_table.add_row("Avg Chunks/File", f"{avg_chunks:.1f}")
    config_table.add_row("RAM Budget", f"{max_ram_gb:.1f} GB")
    config_table.add_row("Files per Group", f"{files_per_group:,}")
    config_table.add_row("Total Groups", f"{len(file_groups):,}")
    config_table.add_row("System CPUs", str(system_cpus))
    config_table.add_row("Max CPUs (limit)", str(total_cpus))
    config_table.add_row("Inference Workers", str(workers))
    config_table.add_row("Threads per Worker", str(threads_per_worker))
    config_table.add_row("Loader Threads", str(loader_threads))
    config_table.add_row("Batch Size", str(batch_size))
    config_table.add_row("Checkpoint Interval", f"Every {checkpoint_interval} batches")
    config_table.add_row("Embedding Format", "float16" if use_float16 else "float32")
    console.print(config_table)
    console.print()

    # Verify thread limits
    env_threads = os.environ.get("OMP_NUM_THREADS", "unset")
    if env_threads != "unset":
        console.print(
            Panel.fit(
                f"[bold green]CPU Limiting Active[/bold green]\n"
                f"[dim]OMP_NUM_THREADS={env_threads}[/dim]\n"
                f"[dim]Each worker will use max {threads_per_worker} threads[/dim]",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel.fit(
                "[bold yellow]Warning: No CPU limit set[/bold yellow]\n"
                "[dim]Workers may use all available cores[/dim]",
                border_style="yellow",
            )
        )
    console.print()

    classes = json.load(open(classes_json))

    # Initialize worker pool once
    init_fn = partial(init_worker, str(model_path), threads_per_worker)

    total_files_processed = 0
    overall_timer = Timer()
    overall_timer.__enter__()

    # Process file groups sequentially
    for group_idx, file_group in enumerate(file_groups):
        console.print(
            Panel.fit(
                f"[bold cyan]Processing Group {group_idx + 1}/{len(file_groups)}[/bold cyan]\n"
                f"[dim]Files: {len(file_group):,} | "
                f"Est. RAM: {len(file_group) * avg_chunks * CHUNK_MEMORY_MB * MEMORY_OVERHEAD_FACTOR:.1f} MB[/dim]",
                border_style="cyan",
            )
        )

        # STEP 1: Load files in this group
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
            task = progress.add_task(
                f"[cyan]Loading group {group_idx + 1}...", total=len(file_group)
            )

            with ThreadPoolExecutor(max_workers=loader_threads) as executor:
                future_to_file = {
                    executor.submit(load_single_file, f): f for f in file_group
                }

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
            console.print("[yellow]! No chunks in this group, skipping[/yellow]\n")
            continue

        # STEP 2: Create batches
        batches = [
            all_chunks[i : i + batch_size]
            for i in range(0, len(all_chunks), batch_size)
        ]

        console.print(
            f"[green]Processing {len(batches):,} batches with {workers} workers[/green]\n"
        )

        # STEP 3: Parallel inference
        chunksize = max(1, min(10, len(batches) // (workers * 2)))

        batch_buffer = []
        processed_chunk_indices = set()
        group_batch_count = 0

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
                f"[bold green]Inference ({workers} workers)", total=len(batches)
            )

            with mp.Pool(workers, initializer=init_fn) as pool:
                for batch_results in pool.imap(
                    process_batch, batches, chunksize=chunksize
                ):
                    batch_buffer.extend(batch_results)
                    group_batch_count += 1
                    total_batches += 1

                    for res in batch_results:
                        processed_chunk_indices.add(
                            (res["file_path"], res["chunk_idx"])
                        )

                    progress.update(task, advance=1)

                    # Save incrementally
                    if group_batch_count % checkpoint_interval == 0 and batch_buffer:
                        rows = [
                            {
                                "file": f"{Path(res['file_path']).stem}_{res['chunk_idx']}",
                                **res,
                            }
                            for res in batch_buffer
                        ]
                        df = pd.DataFrame(rows)

                        write_embeddings_partitioned(
                            output_dir, df, current_checkpoint_id, use_float16
                        )
                        write_predictions_partitioned(
                            output_dir, df, classes, current_checkpoint_id
                        )

                        # Check which files are complete
                        newly_completed = set()
                        for file_path, (start_idx, end_idx) in file_to_chunks.items():
                            num_chunks = end_idx - start_idx
                            completed = sum(
                                1
                                for fp, _ in processed_chunk_indices
                                if fp == file_path
                            )
                            if completed == num_chunks:
                                newly_completed.add(file_path)

                        if newly_completed:
                            processed_files.update(newly_completed)

                        current_checkpoint_id += 1
                        save_checkpoint(
                            checkpoint_path,
                            processed_files,
                            total_batches,
                            current_checkpoint_id,
                        )

                        batch_buffer = []

                        console.print(
                            f"[dim]Checkpoint {current_checkpoint_id}: "
                            f"{len(processed_files)}/{len(all_files)} files complete[/dim]"
                        )

        # Save remaining buffer for this group
        if batch_buffer:
            rows = [
                {
                    "file": f"{Path(res['file_path']).stem}_{res['chunk_idx']}",
                    **res,
                }
                for res in batch_buffer
            ]
            df = pd.DataFrame(rows)

            write_embeddings_partitioned(
                output_dir, df, current_checkpoint_id, use_float16
            )
            write_predictions_partitioned(
                output_dir, df, classes, current_checkpoint_id
            )

            batch_buffer = []

        # Mark all files in this group as complete
        processed_files.update(file_group)
        current_checkpoint_id += 1
        save_checkpoint(
            checkpoint_path, processed_files, total_batches, current_checkpoint_id
        )

        total_files_processed += len(file_group)

        # Free memory before next group
        del all_chunks
        del file_to_chunks
        del processed_chunk_indices
        gc.collect()

        console.print(
            f"[green]v Group {group_idx + 1} complete[/green] "
            f"[dim]({total_files_processed}/{len(files)} files)[/dim]\n"
        )

    overall_timer.__exit__(None, None, None)

    console.print("\n[cyan]v Results saved to partitioned datasets[/cyan]")

    # Show storage info
    emb_dir = output_dir / "embeddings_partitioned"
    pred_dir = output_dir / "predictions_partitioned"

    if emb_dir.exists():
        emb_size = sum(f.stat().st_size for f in emb_dir.rglob("*.parquet"))
        console.print(f"[dim]Embeddings size: {emb_size / 1e9:.2f} GB[/dim]")

    if pred_dir.exists():
        pred_size = sum(f.stat().st_size for f in pred_dir.rglob("*.csv"))
        console.print(f"[dim]Predictions size: {pred_size / 1e6:.2f} MB[/dim]")

    # Clean up checkpoint if done
    if len(processed_files) == len(all_files):
        console.print("[green]v All complete! Removing checkpoint[/green]")
        try:
            checkpoint_path.unlink()
        except OSError:
            pass

    # Results
    speed = len(files) * 60 / overall_timer.elapsed

    results_table = Table(
        title="Performance Results", box=box.DOUBLE_EDGE, title_style="bold magenta"
    )
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green bold", justify="right")

    results_table.add_row("Files Processed", f"{len(files):,}")
    results_table.add_row("Total Completed", f"{len(processed_files):,}")
    results_table.add_row("Total Running Time", f"{overall_timer.elapsed:.2f}s")
    results_table.add_row("Speed", f"{speed:.1f}x realtime")
    results_table.add_row(
        "Throughput", f"{len(files) / overall_timer.elapsed:.2f} files/sec"
    )
    results_table.add_row(
        "CPU Efficiency", f"{(workers * threads_per_worker) / system_cpus * 100:.1f}%"
    )
    results_table.add_row("Peak RAM Used", f"~{max_ram_gb:.1f} GB (budget)")

    console.print(results_table)

    console.print(
        f"\n[dim]Reading embeddings: dataset = pq.ParquetDataset('{emb_dir}')[/dim]"
    )


if __name__ == "__main__":
    app()
