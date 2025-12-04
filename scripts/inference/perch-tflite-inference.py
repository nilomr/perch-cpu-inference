"""
Perch v2 TFLite parallel inference on CPU
"""

from pathlib import Path
import json
import multiprocessing as mp
from functools import partial
from typing import List, Optional, Dict, Tuple
import warnings
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import typer
from contexttimer import Timer
from tqdm import tqdm

try:
    import soundfile as sf
except ImportError:
    raise ImportError("Install soundfile: pip install soundfile")

try:
    import bioacoustics_model_zoo as bmz
except ImportError:
    raise ImportError(
        "Install bioacoustics_model_zoo: pip install bioacoustics-model-zoo"
    )

app = typer.Typer()

# Constants
SAMPLE_RATE = 32000
CHUNK_DURATION = 5
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_DURATION

# Suppress warnings and logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Force spawn instead of fork (safer for TF)
if sys.platform != "win32":
    mp.set_start_method("spawn", force=True)


class InterpreterPool:
    """Persistent interpreter for a worker process."""

    _interpreter = None
    _model_path = None

    @classmethod
    def get_interpreter(cls, model_path: str):
        """Get or create interpreter (one per worker process)."""
        if cls._interpreter is None or cls._model_path != model_path:
            cls._model_path = model_path
            cls._interpreter = tf.lite.Interpreter(
                model_path=model_path,
                num_threads=4,
            )
            cls._interpreter.allocate_tensors()
        return cls._interpreter


def compile_perch_v2_to_tflite(save_path: Path, use_cpu_model: bool = True) -> Path:
    """Download and compile Perch v2 to TFLite."""
    print("Loading Perch v2 model...")

    import torch

    original_cuda_state = torch.cuda.is_available
    if use_cpu_model:
        torch.cuda.is_available = lambda: False

    try:
        perch = bmz.Perch2()
        print(f"Loaded Perch v2 (system: {perch.system}, version: {perch.version})")

        tf_model = perch.tf_model

        print("Converting to TFLite format...")
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_model_dir = Path(temp_dir) / "temp_saved_model"
            concrete_func = tf_model.signatures["serving_default"]

            tf.saved_model.save(
                tf_model,
                str(temp_model_dir),
                signatures={"serving_default": concrete_func},
            )

            converter = tf.lite.TFLiteConverter.from_saved_model(
                str(temp_model_dir), signature_keys=["serving_default"]
            )

            converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            converter.allow_custom_ops = True

            print("Performing conversion...")
            tflite_model = converter.convert()

        save_path = Path(save_path).expanduser()
        save_path.mkdir(parents=True, exist_ok=True)
        model_file = save_path / "perch_v2.tflite"

        with open(model_file, "wb") as f:
            f.write(tflite_model)

        print(f"✓ Saved TFLite model: {model_file}")
        print(f"  Size: {model_file.stat().st_size / (1024**2):.2f} MB")

        metadata = {
            "version": perch.version,
            "system": perch.system,
            "embedding_size": perch.embedding_size,
            "sample_duration": perch.sample_duration,
            "sample_rate": SAMPLE_RATE,
            "num_classes": len(perch.classes),
        }

        with open(save_path / "perch_v2_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        with open(save_path / "ebird_codes.json", "w") as f:
            json.dump(perch.ebird_codes, f, indent=2)
        with open(save_path / "classes.json", "w") as f:
            json.dump(perch.classes, f, indent=2)

        print(f"✓ Saved metadata to: {save_path}")
        return model_file

    finally:
        if use_cpu_model:
            torch.cuda.is_available = original_cuda_state


def load_audio_fast(file_path: str) -> np.ndarray:
    """Fast audio loading - avoid librosa when possible."""
    try:
        info = sf.info(file_path)

        # If already 32kHz, use direct soundfile (10x faster)
        if info.samplerate == SAMPLE_RATE:
            y, _ = sf.read(file_path, dtype="float32", always_2d=False)
            if y.ndim > 1:
                y = y.mean(axis=1)
            return y

        # Only use librosa if resampling needed
        import librosa

        y, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        return y.astype(np.float32)

    except Exception as e:
        warnings.warn(f"Error loading {file_path}: {e}")
        return np.zeros(CHUNK_SAMPLES, dtype=np.float32)


def normalize_chunk(chunk: np.ndarray) -> np.ndarray:
    """Normalize audio chunk to Perch v2 specs."""
    max_val = np.abs(chunk).max()
    if max_val > 1e-10:
        chunk = chunk / max_val
    return chunk * 0.25


def run_inference_single(
    audio_chunk: np.ndarray,
    interpreter: tf.lite.Interpreter,
) -> Dict[str, np.ndarray]:
    """Run inference on single chunk."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    audio_batch = audio_chunk.reshape(1, -1).astype(np.float32)
    interpreter.set_tensor(input_details[0]["index"], audio_batch)
    interpreter.invoke()

    outputs = {}
    for detail in output_details:
        shape = detail["shape"]
        data = interpreter.get_tensor(detail["index"])

        if len(shape) == 2 and shape[1] > 10000:
            outputs["logits"] = data.squeeze(0)
        elif len(shape) == 2 and shape[1] == 1536:
            outputs["embeddings"] = data.squeeze(0)

    return outputs


def process_file_optimized(
    file_path: str,
    model_path: str,
) -> Tuple[str, Optional[List[Dict]]]:
    """Process single file with pooled interpreter."""
    try:
        # Get persistent interpreter for this worker
        interpreter = InterpreterPool.get_interpreter(model_path)

        # Fast audio loading
        y = load_audio_fast(file_path)

        num_chunks = int(np.ceil(len(y) / CHUNK_SAMPLES))
        results = []

        for chunk_idx in range(num_chunks):
            start_sample = chunk_idx * CHUNK_SAMPLES
            end_sample = min(start_sample + CHUNK_SAMPLES, len(y))
            start_time = chunk_idx * CHUNK_DURATION
            end_time = end_sample / SAMPLE_RATE

            chunk = y[start_sample:end_sample]

            # Pad if needed
            if len(chunk) < CHUNK_SAMPLES:
                padded = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
                padded[: len(chunk)] = chunk
                chunk = padded

            chunk = normalize_chunk(chunk)
            outputs = run_inference_single(chunk, interpreter)

            results.append(
                {
                    "chunk_idx": chunk_idx,
                    "start_time": start_time,
                    "end_time": end_time,
                    "logits": outputs.get("logits"),
                    "embeddings": outputs.get("embeddings"),
                }
            )

        return (file_path, results)

    except Exception as e:
        warnings.warn(f"Error processing {file_path}: {e}")
        return (file_path, None)


def parallel_inference_optimized(
    audio_files: List[str],
    model_path: Path,
    num_workers: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
    chunksize: int = 1,
) -> pd.DataFrame:
    """Run optimized parallel inference."""
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 2)

    print(f"Processing {len(audio_files)} files with {num_workers} workers")

    processed_files = set()
    if checkpoint_path and checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint_df = pd.read_parquet(checkpoint_path)
        processed_files = set(checkpoint_df["file_path"].unique())
        print(f"  Resuming: {len(processed_files)} files already processed")

    files_to_process = [f for f in audio_files if f not in processed_files]

    if not files_to_process:
        print("All files already processed!")
        return pd.read_parquet(checkpoint_path) if checkpoint_path else pd.DataFrame()

    worker_fn = partial(process_file_optimized, model_path=str(model_path))

    all_results = []

    # Use spawn context explicitly
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        with tqdm(total=len(files_to_process), desc="Processing", unit="file") as pbar:
            for file_path, results in pool.imap_unordered(
                worker_fn, files_to_process, chunksize=chunksize
            ):
                if results is not None:
                    for result in results:
                        all_results.append(
                            {
                                "file": f"{Path(file_path).stem}_{result['chunk_idx']}",
                                "file_path": file_path,
                                "chunk_idx": result["chunk_idx"],
                                "start_time": result["start_time"],
                                "end_time": result["end_time"],
                                "logits": result["logits"],
                                "embeddings": result["embeddings"],
                            }
                        )

                pbar.update(1)

                if checkpoint_path and len(all_results) > 0 and pbar.n % 100 == 0:
                    temp_df = pd.DataFrame(all_results)
                    temp_df.to_parquet(checkpoint_path, compression="snappy")

    if not all_results:
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results)

    if checkpoint_path:
        results_df.to_parquet(checkpoint_path, compression="snappy")
        print(f"✓ Checkpoint saved: {checkpoint_path}")

    return results_df


@app.command()
def compile_model(
    save_path: Path = typer.Option("models/perch_v2", help="Model save directory"),
    use_cpu: bool = typer.Option(True, help="Use CPU model"),
):
    """Compile Perch v2 to TFLite."""
    try:
        model_path = compile_perch_v2_to_tflite(save_path, use_cpu)
        print(f"\n✓ Complete: {model_path}")
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        raise


@app.command()
def run_inference(
    audio_dir: Path = typer.Option(..., help="Audio directory"),
    model_path: Path = typer.Option(
        "models/perch_v2/perch_v2.tflite", help="Model path"
    ),
    output_dir: Path = typer.Option("output", help="Output directory"),
    num_workers: Optional[int] = typer.Option(None, help="Number of workers"),
    file_pattern: str = typer.Option("*.wav", help="File pattern"),
    save_embeddings: bool = typer.Option(True, help="Save embeddings"),
    save_top5: bool = typer.Option(True, help="Save top-5 predictions"),
    checkpoint: bool = typer.Option(True, help="Enable checkpointing"),
):
    """Run optimized parallel inference."""

    audio_files = sorted(
        set(
            [
                str(f)
                for pattern in [file_pattern, file_pattern.upper()]
                for f in Path(audio_dir).expanduser().glob(pattern)
            ]
        )
    )

    if not audio_files:
        print(f"No files found in {audio_dir} matching {file_pattern}")
        return

    print(f"Found {len(audio_files)} files")

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "checkpoint.parquet" if checkpoint else None

    with Timer() as timer:
        results_df = parallel_inference_optimized(
            audio_files=audio_files,
            model_path=model_path,
            num_workers=num_workers,
            checkpoint_path=checkpoint_path,
        )

    if results_df.empty:
        print("No results generated")
        return

    if save_embeddings and "embeddings" in results_df.columns:
        emb_df = results_df[
            ["file", "file_path", "chunk_idx", "start_time", "end_time"]
        ].copy()
        emb_array = np.stack(results_df["embeddings"].values)
        for i in range(emb_array.shape[1]):
            emb_df[f"emb_{i}"] = emb_array[:, i]
        emb_path = output_dir / "embeddings.parquet"
        emb_df.to_parquet(emb_path, compression="snappy")
        print(f"✓ Saved embeddings: {emb_path}")

    if save_top5 and "logits" in results_df.columns:
        classes_path = Path(model_path).parent / "classes.json"
        with open(classes_path) as f:
            classes = json.load(f)

        top5_rows = []
        for _, row in results_df.iterrows():
            logits = row["logits"]
            top5_idx = np.argsort(logits)[-5:][::-1]
            top5_rows.append(
                {
                    "file": row["file"],
                    "file_path": row["file_path"],
                    "chunk_idx": row["chunk_idx"],
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    **{f"top{i + 1}": classes[idx] for i, idx in enumerate(top5_idx)},
                    **{f"score{i + 1}": logits[idx] for i, idx in enumerate(top5_idx)},
                }
            )

        top5_df = pd.DataFrame(top5_rows)
        top5_path = output_dir / "top5_classes.csv"
        top5_df.to_csv(top5_path, index=False)
        print(f"✓ Saved top-5: {top5_path}")

    total_audio_duration = len(audio_files) * 60
    print(f"\n✓ Complete!")
    print(f"  Files: {len(audio_files)} ({total_audio_duration / 3600:.1f} hours)")
    print(f"  Time: {timer.elapsed:.1f}s ({timer.elapsed / 60:.1f} min)")
    print(f"  Speed: {total_audio_duration / timer.elapsed:.1f}× realtime")
    print(f"  Throughput: {len(audio_files) / timer.elapsed:.2f} files/sec")


@app.command()
def benchmark(
    audio_dir: Path = typer.Option(..., help="Test audio directory"),
    model_path: Path = typer.Option(
        "models/perch_v2/perch_v2.tflite", help="Model path"
    ),
    num_files: int = typer.Option(76, help="Files to benchmark"),
    worker_counts: str = typer.Option("8,16,24,32,40", help="Worker counts"),
):
    """Benchmark different configurations."""

    audio_files = sorted(
        set(
            [
                str(f)
                for pattern in ["*.wav", "*.WAV"]
                for f in Path(audio_dir).expanduser().glob(pattern)
            ]
        )
    )[:num_files]

    if not audio_files:
        print(f"No files found in {audio_dir}")
        return

    print(f"Benchmarking with {len(audio_files)} files\n")

    results = []
    for workers in [int(w) for w in worker_counts.split(",")]:
        print(f"Testing: {workers} workers")

        with Timer() as timer:
            _ = parallel_inference_optimized(
                audio_files=audio_files,
                model_path=model_path,
                num_workers=workers,
                checkpoint_path=None,
            )

        speed = len(audio_files) / timer.elapsed
        results.append(
            {
                "workers": workers,
                "time_sec": timer.elapsed,
                "files_per_sec": speed,
                "realtime_factor": (len(audio_files) * 60) / timer.elapsed,
            }
        )
        print(
            f"  {timer.elapsed:.1f}s | {speed:.2f} files/s | {results[-1]['realtime_factor']:.1f}× realtime\n"
        )

    df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)

    # Find optimal
    best = df.loc[df["files_per_sec"].idxmax()]
    print(
        f"\n✓ Optimal: {int(best['workers'])} workers → {best['files_per_sec']:.2f} files/s"
    )


if __name__ == "__main__":
    app()
