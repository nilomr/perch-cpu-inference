"""
Benchmark script for Perch v2 ONNX inference
Saves all test data to JSON with detailed statistics and analysis
"""

from pathlib import Path
import json
import subprocess
import shutil
import tempfile
import time
import statistics
import sys
import platform
from typing import List, Dict, Optional
from datetime import datetime
import psutil
import threading
from queue import Queue, Empty

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box

app = typer.Typer()
console = Console()


class ResourceMonitor:
    """Monitor CPU and memory usage of a subprocess"""

    def __init__(self, pid: int, interval: float = 0.5):
        self.pid = pid
        self.interval = interval
        self.cpu_samples = []
        self.memory_samples = []
        self.monitoring = False
        self.thread = None

    def start(self):
        """Start monitoring in a background thread"""
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def _monitor(self):
        """Monitor loop"""
        try:
            process = psutil.Process(self.pid)
            process.cpu_percent(interval=None)
            time.sleep(0.1)

            while self.monitoring:
                try:
                    cpu = process.cpu_percent(interval=None)
                    mem_info = process.memory_info()
                    memory_mb = mem_info.rss / (1024 * 1024)

                    children = process.children(recursive=True)
                    for child in children:
                        try:
                            cpu += child.cpu_percent(interval=None)
                            memory_mb += child.memory_info().rss / (1024 * 1024)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass

                    if cpu > 0:
                        self.cpu_samples.append(cpu)
                    self.memory_samples.append(memory_mb)

                    time.sleep(self.interval)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
        except psutil.NoSuchProcess:
            pass

    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        if not self.cpu_samples or not self.memory_samples:
            return {
                "cpu_mean": 0.0,
                "cpu_median": 0.0,
                "cpu_max": 0.0,
                "cpu_min": 0.0,
                "cpu_stdev": 0.0,
                "cpu_samples": [],
                "memory_mean_mb": 0.0,
                "memory_median_mb": 0.0,
                "memory_max_mb": 0.0,
                "memory_min_mb": 0.0,
                "memory_stdev_mb": 0.0,
                "memory_samples": [],
            }

        return {
            "cpu_mean": statistics.mean(self.cpu_samples),
            "cpu_median": statistics.median(self.cpu_samples),
            "cpu_max": max(self.cpu_samples),
            "cpu_min": min(self.cpu_samples),
            "cpu_stdev": statistics.stdev(self.cpu_samples)
            if len(self.cpu_samples) > 1
            else 0.0,
            "cpu_samples": self.cpu_samples.copy(),
            "memory_mean_mb": statistics.mean(self.memory_samples),
            "memory_median_mb": statistics.median(self.memory_samples),
            "memory_max_mb": max(self.memory_samples),
            "memory_min_mb": min(self.memory_samples),
            "memory_stdev_mb": statistics.stdev(self.memory_samples)
            if len(self.memory_samples) > 1
            else 0.0,
            "memory_samples": self.memory_samples.copy(),
        }


def get_system_info() -> Dict:
    """Collect system information for the benchmark report"""
    cpu_freq = psutil.cpu_freq()
    mem = psutil.virtual_memory()

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_freq_current_mhz": cpu_freq.current if cpu_freq else None,
        "cpu_freq_max_mhz": cpu_freq.max if cpu_freq else None,
        "memory_total_gb": mem.total / (1024**3),
        "memory_available_gb": mem.available / (1024**3),
        "timestamp": datetime.now().isoformat(),
    }


def calculate_statistics(values: List[float]) -> Dict:
    """Calculate comprehensive statistics for a list of values"""
    if not values:
        return {}

    n = len(values)
    mean_val = statistics.mean(values)

    stats = {
        "count": n,
        "mean": mean_val,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
    }

    if n > 1:
        stdev = statistics.stdev(values)
        stats["stdev"] = stdev
        stats["variance"] = statistics.variance(values)
        # Coefficient of variation (relative variability)
        stats["coefficient_of_variation"] = (
            (stdev / mean_val * 100) if mean_val > 0 else 0.0
        )

        # Percentiles
        sorted_vals = sorted(values)
        stats["percentile_25"] = sorted_vals[int(n * 0.25)]
        stats["percentile_75"] = sorted_vals[int(n * 0.75)]
        stats["percentile_95"] = (
            sorted_vals[int(n * 0.95)] if n >= 20 else sorted_vals[-1]
        )

    return stats


def copy_test_files(source_dir: Path, num_files: int) -> Path:
    """Copy test files to temporary directory"""
    if not source_dir.exists():
        raise FileNotFoundError(f"Test data directory not found: {source_dir}")

    audio_files = sorted(
        list(source_dir.glob("*.wav")) + list(source_dir.glob("*.WAV"))
    )

    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {source_dir}")

    files_to_copy = audio_files[:num_files]
    temp_dir = Path(tempfile.mkdtemp(prefix="perch_benchmark_"))

    console.print(f"[cyan]Copying {len(files_to_copy)} files to {temp_dir}...[/cyan]")

    for file in files_to_copy:
        shutil.copy2(file, temp_dir / file.name)

    return temp_dir


def run_inference(
    script_path: Path,
    audio_dir: Path,
    model_path: Path,
    classes_json: Path,
    output_dir: Path,
    batch_size: int,
    workers: int,
    loader_threads: int,
    use_float16: bool,
    timeout: int = 600,
) -> Dict:
    """Run inference and collect comprehensive metrics"""

    cmd = [
        sys.executable,
        str(script_path),
        "--audio-dir",
        str(audio_dir),
        "--model-path",
        str(model_path),
        "--classes-json",
        str(classes_json),
        "--output-dir",
        str(output_dir),
        "--batch-size",
        str(batch_size),
        "--workers",
        str(workers),
        "--loader-threads",
        str(loader_threads),
        "--checkpoint-interval",
        "10000",
        "--no-resume",
    ]

    if use_float16:
        cmd.append("--float16")

    console.print(f"[dim]Running inference...[/dim]")

    start_time = time.perf_counter()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    console.print(f"[cyan]Started PID: {process.pid}[/cyan]")

    monitor = ResourceMonitor(process.pid, interval=0.5)
    monitor.start()

    stdout_lines = []
    stderr_lines = []

    start_wait = time.time()
    last_output_time = time.time()

    while process.poll() is None:
        if time.time() - start_wait > timeout:
            console.print(f"\n[red]Timeout after {timeout}s[/red]")
            process.kill()
            monitor.stop()
            return {
                "elapsed_time": time.perf_counter() - start_time,
                "returncode": -1,
                "error": f"Timeout after {timeout}s",
            }

        if time.time() - last_output_time > 30:
            console.print(f"\n[yellow]No output for 30s...[/yellow]")

        try:
            import select

            if hasattr(select, "select"):
                ready, _, _ = select.select(
                    [process.stdout, process.stderr], [], [], 0.1
                )
                for stream in ready:
                    line = stream.readline()
                    if line:
                        last_output_time = time.time()
                        if stream == process.stdout:
                            stdout_lines.append(line)
                            if any(
                                word in line
                                for word in [
                                    "Loading",
                                    "Processing",
                                    "complete",
                                    "Speed",
                                ]
                            ):
                                console.print(f"[dim]{line.rstrip()}[/dim]")
                        else:
                            stderr_lines.append(line)
            else:
                time.sleep(0.5)
        except Exception as e:
            console.print(f"[yellow]Read error: {e}[/yellow]")
            time.sleep(0.1)

    remaining_stdout, remaining_stderr = process.communicate(timeout=5)
    if remaining_stdout:
        stdout_lines.append(remaining_stdout)
    if remaining_stderr:
        stderr_lines.append(remaining_stderr)

    end_time = time.perf_counter()
    monitor.stop()

    elapsed = end_time - start_time
    stdout = "".join(stdout_lines)
    stderr = "".join(stderr_lines)

    console.print(
        f"[green]Completed in {elapsed:.1f}s (returncode: {process.returncode})[/green]"
    )

    # Parse metrics
    metrics = {
        "elapsed_time": elapsed,
        "returncode": process.returncode,
        "stdout_snippet": stdout[:500] if stdout else "",  # Store snippet only
        "stderr_snippet": stderr[:500] if stderr else "",
    }

    # Extract metrics from stdout
    for line in stdout.split("\n"):
        if "Files Processed" in line and "│" in line:
            try:
                parts = line.split("│")
                if len(parts) >= 2:
                    value = parts[-1].strip().replace(",", "")
                    metrics["files_processed"] = int(value)
            except (IndexError, ValueError):
                pass
        elif "Chunks Processed" in line and "│" in line:
            try:
                parts = line.split("│")
                if len(parts) >= 2:
                    value = parts[-1].strip().replace(",", "")
                    metrics["chunks_processed"] = int(value)
            except (IndexError, ValueError):
                pass
        elif "Throughput" in line and "files/sec" in line and "│" in line:
            try:
                parts = line.split("│")
                if len(parts) >= 2:
                    throughput_str = parts[-1].strip().split("files/sec")[0].strip()
                    metrics["throughput_files_per_sec"] = float(throughput_str)
            except (IndexError, ValueError):
                pass

    # Add resource stats
    metrics.update(monitor.get_stats())

    return metrics


def run_benchmark_config(
    script_path: Path,
    audio_dir: Path,
    model_path: Path,
    classes_json: Path,
    config: Dict,
    iterations: int,
    timeout: int,
) -> Dict:
    """Run benchmark for a specific configuration and return ALL raw data"""

    results = []

    console.print(
        f"\n[cyan]Config: workers={config['workers']}, batch={config['batch_size']}[/cyan]"
    )

    for i in range(iterations):
        output_dir = Path(tempfile.mkdtemp(prefix="perch_output_"))

        try:
            console.print(f"[yellow]Run {i + 1}/{iterations}[/yellow]")

            metrics = run_inference(
                script_path=script_path,
                audio_dir=audio_dir,
                model_path=model_path,
                classes_json=classes_json,
                output_dir=output_dir,
                batch_size=config["batch_size"],
                workers=config["workers"],
                loader_threads=config["loader_threads"],
                use_float16=config["use_float16"],
                timeout=timeout,
            )

            if metrics["returncode"] == 0:
                results.append(metrics)
                console.print(
                    f"[green]✓ Run {i + 1}: {metrics['elapsed_time']:.1f}s[/green]"
                )
            else:
                error_msg = metrics.get("error", f"Exit code: {metrics['returncode']}")
                console.print(f"[yellow]⚠ Run {i + 1} failed: {error_msg}[/yellow]")

        finally:
            if output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)

    if not results:
        console.print(f"[red]✗ All runs failed[/red]")
        return None

    # Calculate comprehensive statistics
    elapsed_times = [r["elapsed_time"] for r in results]
    throughputs = [
        r.get("throughput_files_per_sec", 0)
        for r in results
        if "throughput_files_per_sec" in r
    ]
    cpu_means = [r.get("cpu_mean", 0) for r in results]
    memory_means = [r.get("memory_mean_mb", 0) for r in results]

    return {
        "config": config,
        "iterations_successful": len(results),
        "iterations_requested": iterations,
        # Store ALL raw results
        "raw_results": results,
        # Elapsed time statistics
        "elapsed_time_stats": calculate_statistics(elapsed_times),
        # Throughput statistics
        "throughput_stats": calculate_statistics(throughputs) if throughputs else {},
        # CPU statistics
        "cpu_stats": calculate_statistics(cpu_means),
        # Memory statistics
        "memory_stats": calculate_statistics(memory_means),
        # Other metrics from first run
        "files_processed": results[0].get("files_processed", 0),
        "chunks_processed": results[0].get("chunks_processed", 0),
    }


def calculate_speedup_analysis(all_results: List[Dict]) -> Dict:
    """Calculate speedup and relative performance between configurations"""
    if len(all_results) < 2:
        return {}

    # Find baseline (slowest configuration)
    baseline = max(
        all_results,
        key=lambda x: x["elapsed_time_stats"].get("mean", float("inf")),
    )
    baseline_time = baseline["elapsed_time_stats"]["mean"]

    speedup_analysis = {
        "baseline_config": baseline["config"],
        "baseline_time": baseline_time,
        "comparisons": [],
    }

    for result in all_results:
        current_time = result["elapsed_time_stats"]["mean"]
        speedup = baseline_time / current_time if current_time > 0 else 0

        speedup_analysis["comparisons"].append(
            {
                "config": result["config"],
                "mean_time": current_time,
                "speedup": speedup,
                "percent_faster": (speedup - 1) * 100,
                "time_saved_per_run": baseline_time - current_time,
            }
        )

    # Sort by speedup
    speedup_analysis["comparisons"].sort(key=lambda x: x["speedup"], reverse=True)

    # Find best config
    best = speedup_analysis["comparisons"][0]
    speedup_analysis["best_config"] = best["config"]
    speedup_analysis["best_speedup"] = best["speedup"]

    return speedup_analysis


@app.command()
def benchmark(
    script_path: Path = typer.Option(
        "perch-onnx-inference.py",
        "--script",
        "-s",
        help="Path to inference script",
    ),
    test_dir: Path = typer.Option(
        "test-data/am-files",
        "--test-dir",
        "-t",
        help="Directory containing test audio files",
    ),
    model_path: Path = typer.Option(
        "./models/perch_v2/perch_v2.onnx",
        "--model",
        "-m",
        help="Path to ONNX model",
    ),
    classes_json: Path = typer.Option(
        "./models/perch_v2/classes.json",
        "--classes",
        "-c",
        help="Path to classes JSON",
    ),
    num_files: int = typer.Option(
        10,
        "--num-files",
        "-n",
        help="Number of test files to use",
    ),
    iterations: int = typer.Option(
        3,
        "--iterations",
        "-i",
        help="Number of iterations per configuration",
    ),
    test_workers: Optional[List[int]] = typer.Option(
        None,
        "--workers",
        "-w",
        help="Worker counts to test",
    ),
    test_batch_sizes: Optional[List[int]] = typer.Option(
        None,
        "--batch-sizes",
        "-b",
        help="Batch sizes to test",
    ),
    timeout: int = typer.Option(
        600,
        "--timeout",
        help="Timeout per run in seconds",
    ),
    skip_test: bool = typer.Option(
        False,
        "--skip-test",
        help="Skip initial test of inference script",
    ),
):
    """
    Benchmark Perch v2 ONNX inference with comprehensive JSON output.

    Saves detailed results to benchmark_results_TIMESTAMP.json in current directory.
    """

    # Validate inputs
    if not script_path.exists():
        console.print(f"[red]Error: Script not found: {script_path}[/red]")
        raise typer.Exit(1)

    if not model_path.exists():
        console.print(f"[red]Error: Model not found: {model_path}[/red]")
        raise typer.Exit(1)

    if not classes_json.exists():
        console.print(f"[red]Error: Classes JSON not found: {classes_json}[/red]")
        raise typer.Exit(1)

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = Path.cwd() / f"benchmark_results_{timestamp}.json"

    console.print()
    console.print(
        Panel.fit(
            "[bold magenta]Perch v2 ONNX Inference Benchmark[/bold magenta]\n"
            f"[dim]Output: {output_json.name}[/dim]\n"
            f"[dim]Test files: {num_files} | Iterations: {iterations}[/dim]",
            border_style="magenta",
        )
    )

    # Collect system info
    system_info = get_system_info()

    # Setup test configurations
    if test_workers is None:
        test_workers = [2, 4]

    if test_batch_sizes is None:
        test_batch_sizes = [8, 16]

    console.print(f"\n[cyan]Testing configurations:[/cyan]")
    console.print(f"  Workers: {test_workers}")
    console.print(f"  Batch sizes: {test_batch_sizes}\n")

    # Copy test files
    try:
        temp_audio_dir = copy_test_files(test_dir, num_files)
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]✓ Test files ready in {temp_audio_dir}[/green]")

    # Generate test configurations
    configs = []
    for workers in test_workers:
        for batch_size in test_batch_sizes:
            configs.append(
                {
                    "workers": workers,
                    "batch_size": batch_size,
                    "loader_threads": 8,
                    "use_float16": False,
                }
            )

    console.print(
        Panel.fit(
            f"[bold cyan]Running {len(configs)} configurations × {iterations} iterations[/bold cyan]",
            border_style="cyan",
        )
    )

    all_results = []
    start_benchmark = time.perf_counter()

    for config in configs:
        result = run_benchmark_config(
            script_path=script_path,
            audio_dir=temp_audio_dir,
            model_path=model_path,
            classes_json=classes_json,
            config=config,
            iterations=iterations,
            timeout=timeout,
        )

        if result:
            all_results.append(result)

    end_benchmark = time.perf_counter()
    total_benchmark_time = end_benchmark - start_benchmark

    # Clean up temp directory
    if temp_audio_dir.exists():
        shutil.rmtree(temp_audio_dir, ignore_errors=True)

    if not all_results:
        console.print("[red]No successful runs to report[/red]")
        raise typer.Exit(1)

    # Calculate speedup analysis
    speedup_analysis = calculate_speedup_analysis(all_results)

    # Create comprehensive JSON report
    report = {
        "benchmark_metadata": {
            "timestamp": datetime.now().isoformat(),
            "script_path": str(script_path),
            "model_path": str(model_path),
            "num_test_files": num_files,
            "iterations_per_config": iterations,
            "total_benchmark_time_seconds": total_benchmark_time,
            "configurations_tested": len(configs),
            "successful_configurations": len(all_results),
        },
        "system_info": system_info,
        "test_parameters": {
            "workers_tested": test_workers,
            "batch_sizes_tested": test_batch_sizes,
            "timeout_seconds": timeout,
        },
        "results": all_results,
        "speedup_analysis": speedup_analysis,
    }

    # Save JSON report
    with open(output_json, "w") as f:
        json.dump(report, f, indent=2, default=str)

    console.print("\n" + "=" * 60)
    console.print(
        Panel.fit(
            "[bold green]Benchmark Complete[/bold green]",
            border_style="green",
        )
    )
    console.print()

    # Display summary table
    table = Table(
        title="Performance Results Summary",
        box=box.DOUBLE_EDGE,
        title_style="bold magenta",
    )

    table.add_column("Workers", style="cyan", justify="center")
    table.add_column("Batch", style="cyan", justify="center")
    table.add_column("Time (s)", style="green", justify="right")
    table.add_column("CV %", style="yellow", justify="right")
    table.add_column("Speedup", style="blue", justify="right")
    table.add_column("CPU %", style="magenta", justify="right")
    table.add_column("Runs", style="dim", justify="right")

    for result in all_results:
        cfg = result["config"]
        time_stats = result["elapsed_time_stats"]
        cpu_stats = result["cpu_stats"]

        # Find speedup for this config
        speedup_val = 1.0
        if speedup_analysis and "comparisons" in speedup_analysis:
            for comp in speedup_analysis["comparisons"]:
                if comp["config"] == cfg:
                    speedup_val = comp["speedup"]
                    break

        table.add_row(
            str(cfg["workers"]),
            str(cfg["batch_size"]),
            f"{time_stats['mean']:.1f}±{time_stats.get('stdev', 0):.1f}",
            f"{time_stats.get('coefficient_of_variation', 0):.1f}",
            f"{speedup_val:.2f}x",
            f"{cpu_stats.get('mean', 0):.0f}",
            f"{result['iterations_successful']}/{result['iterations_requested']}",
        )

    console.print(table)
    console.print()

    # Display speedup summary
    if speedup_analysis and "best_config" in speedup_analysis:
        best_cfg = speedup_analysis["best_config"]
        console.print(
            Panel.fit(
                f"[bold yellow]Best Configuration[/bold yellow]\n\n"
                f"[green]Workers:[/green] {best_cfg['workers']}\n"
                f"[green]Batch size:[/green] {best_cfg['batch_size']}\n"
                f"[green]Speedup:[/green] {speedup_analysis['best_speedup']:.2f}x faster than baseline\n"
                f"[green]Performance gain:[/green] {(speedup_analysis['best_speedup'] - 1) * 100:.1f}% faster",
                border_style="yellow",
            )
        )
        console.print()

    console.print(
        f"[green]✓ Detailed results saved to:[/green] [bold]{output_json}[/bold]"
    )
    console.print(f"[cyan]Total benchmark time: {total_benchmark_time:.1f}s[/cyan]\n")


if __name__ == "__main__":
    app()
