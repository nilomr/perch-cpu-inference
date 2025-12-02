#!/usr/bin/env python3
"""
Benchmark Results Analysis Script

Analyzes benchmark results to calculate how much faster than real-time
the different configurations perform, with statistics table.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box


def load_benchmark_results(filepath: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def calculate_speedup_factors(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Calculate speedup factors for each configuration.

    Real-time processing: Files are 60 seconds of audio, so real-time means
    processing 1 file per 60 seconds = 0.0167 files/second
    """
    real_time_rate = (
        1.0 / 60.0
    )  # files per second for real-time processing (60-second files)
    speedup_results = []

    for result in results["results"]:
        config = result["config"]
        throughput_stats = result["throughput_stats"]

        # Calculate speedup using mean throughput
        mean_throughput = throughput_stats["mean"]
        speedup = mean_throughput / real_time_rate

        speedup_results.append(
            {
                "config": config,
                "mean_throughput": mean_throughput,
                "speedup_factor": speedup,
                "throughput_stats": throughput_stats,
                "elapsed_time_stats": result.get("elapsed_time_stats", {}),
                "cpu_stats": result.get("cpu_stats", {}),
                "memory_stats": result.get("memory_stats", {}),
                "iterations_successful": result.get("iterations_successful", 0),
                "iterations_requested": result.get("iterations_requested", 0),
            }
        )

    # Sort by speedup factor (highest first)
    speedup_results.sort(key=lambda x: x["speedup_factor"], reverse=True)

    return speedup_results


def print_detailed_analysis(speedup_results: List[Dict[str, Any]]):
    """Print comprehensive analysis table with all statistics."""
    console = Console()

    # Header
    console.print("\n" + "=" * 120)
    console.print("COMPREHENSIVE BENCHMARK ANALYSIS".center(120))
    console.print("=" * 120)
    console.print(
        "Real-time processing rate: 0.0167 files/second (60-second audio files at 1:1 speed)"
    )
    console.print()

    # Main results table
    table = Table(
        title="Detailed Performance Analysis (Ranked by Speedup)",
        box=box.DOUBLE_EDGE,
        title_style="bold magenta",
        show_header=True,
        header_style="bold cyan",
    )

    # Add columns
    table.add_column("Rank", style="yellow", justify="center", width=4)
    table.add_column("Workers", style="cyan", justify="center", width=7)
    table.add_column("Batch", style="cyan", justify="center", width=5)
    table.add_column("Time (s)", style="green", justify="right", width=12)
    table.add_column("CV %", style="yellow", justify="right", width=6)
    table.add_column("Throughput\n(files/sec)", style="blue", justify="right", width=12)
    table.add_column("Speedup\n(x real-time)", style="red", justify="right", width=12)
    table.add_column("CPU %", style="magenta", justify="right", width=8)
    table.add_column("Memory\n(MB)", style="cyan", justify="right", width=10)
    table.add_column("Runs", style="dim", justify="center", width=6)

    for i, result in enumerate(speedup_results, 1):
        config = result["config"]
        time_stats = result["elapsed_time_stats"]
        throughput_stats = result["throughput_stats"]
        cpu_stats = result["cpu_stats"]
        memory_stats = result["memory_stats"]

        # Format time with std dev
        time_mean = time_stats.get("mean", 0)
        time_stdev = time_stats.get("stdev", 0)
        time_str = f"{time_mean:.1f}±{time_stdev:.1f}"

        # Coefficient of variation for time
        cv_time = time_stats.get("coefficient_of_variation", 0)

        # Throughput
        throughput_mean = throughput_stats.get("mean", 0)
        throughput_str = f"{throughput_mean:.2f}"

        # Speedup
        speedup = result["speedup_factor"]
        speedup_str = f"{speedup:.1f}"

        # CPU usage
        cpu_mean = cpu_stats.get("mean", 0)
        cpu_str = f"{cpu_mean:.0f}"

        # Memory usage
        memory_mean = memory_stats.get("mean", 0)
        memory_str = f"{memory_mean:.0f}"

        # Runs
        runs_successful = result["iterations_successful"]
        runs_requested = result["iterations_requested"]
        runs_str = f"{runs_successful}/{runs_requested}"

        table.add_row(
            str(i),
            str(config["workers"]),
            str(config["batch_size"]),
            time_str,
            f"{cv_time:.1f}",
            throughput_str,
            speedup_str,
            cpu_str,
            memory_str,
            runs_str,
        )

    console.print(table)

    # Additional statistics panel
    if speedup_results:
        best_result = speedup_results[0]
        worst_result = speedup_results[-1]

        console.print("\n")
        console.print(
            Panel.fit(
                "[bold green]Performance Summary[/bold green]\n\n"
                f"[green]Best Configuration:[/green] {best_result['config']['workers']} workers, {best_result['config']['batch_size']} batch\n"
                f"[green]Best Speedup:[/green] {best_result['speedup_factor']:.1f}x real-time ({best_result['mean_throughput']:.2f} files/sec)\n"
                f"[green]Worst Configuration:[/green] {worst_result['config']['workers']} workers, {worst_result['config']['batch_size']} batch\n"
                f"[green]Worst Speedup:[/green] {worst_result['speedup_factor']:.1f}x real-time ({worst_result['mean_throughput']:.2f} files/sec)\n"
                f"[green]Performance Range:[/green] {best_result['speedup_factor'] / worst_result['speedup_factor']:.1f}x difference",
                border_style="green",
            )
        )


def print_summary_stats(speedup_results: List[Dict[str, Any]]):
    """Print summary statistics."""
    if not speedup_results:
        return

    speedups = [r["speedup_factor"] for r in speedup_results]
    throughputs = [r["mean_throughput"] for r in speedup_results]
    times = [r["elapsed_time_stats"].get("mean", 0) for r in speedup_results]
    cpus = [r["cpu_stats"].get("mean", 0) for r in speedup_results]
    memories = [r["memory_stats"].get("mean", 0) for r in speedup_results]

    console = Console()

    console.print("\n" + "=" * 60)
    console.print("SUMMARY STATISTICS".center(60))
    console.print("=" * 60)

    # Create summary table
    summary_table = Table(box=box.SIMPLE, show_header=False)
    summary_table.add_column("Metric", style="cyan", width=20)
    summary_table.add_column("Min", style="red", justify="right", width=10)
    summary_table.add_column("Max", style="green", justify="right", width=10)
    summary_table.add_column("Avg", style="yellow", justify="right", width=10)
    summary_table.add_column("Range", style="blue", justify="right", width=12)

    summary_table.add_row(
        "Speedup (x real-time)",
        f"{min(speedups):.1f}",
        f"{max(speedups):.1f}",
        f"{sum(speedups) / len(speedups):.1f}",
        f"{max(speedups) - min(speedups):.1f}",
    )
    summary_table.add_row(
        "Throughput (files/sec)",
        f"{min(throughputs):.2f}",
        f"{max(throughputs):.2f}",
        f"{sum(throughputs) / len(throughputs):.2f}",
        f"{max(throughputs) - min(throughputs):.2f}",
    )
    summary_table.add_row(
        "Time (seconds)",
        f"{min(times):.1f}",
        f"{max(times):.1f}",
        f"{sum(times) / len(times):.1f}",
        f"{max(times) - min(times):.1f}",
    )
    summary_table.add_row(
        "CPU Usage (%)",
        f"{min(cpus):.0f}",
        f"{max(cpus):.0f}",
        f"{sum(cpus) / len(cpus):.0f}",
        f"{max(cpus) - min(cpus):.0f}",
    )
    summary_table.add_row(
        "Memory (MB)",
        f"{min(memories):.0f}",
        f"{max(memories):.0f}",
        f"{sum(memories) / len(memories):.0f}",
        f"{max(memories) - min(memories):.0f}",
    )

    console.print(summary_table)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results for speedup factors and detailed statistics"
    )
    parser.add_argument("results_file", help="Path to benchmark results JSON file")
    parser.add_argument("--output", "-o", help="Output file for results (optional)")

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.results_file).exists():
        print(f"Error: Results file '{args.results_file}' not found.")
        sys.exit(1)

    # Load and analyze results
    try:
        results = load_benchmark_results(args.results_file)
        speedup_results = calculate_speedup_factors(results)

        # Print detailed analysis
        print_detailed_analysis(speedup_results)
        print_summary_stats(speedup_results)

        # Save to file if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(
                    {
                        "speedup_analysis": speedup_results,
                        "real_time_rate": 1.0 / 60.0,
                        "summary": {
                            "max_speedup": max(
                                r["speedup_factor"] for r in speedup_results
                            ),
                            "min_speedup": min(
                                r["speedup_factor"] for r in speedup_results
                            ),
                            "avg_speedup": sum(
                                r["speedup_factor"] for r in speedup_results
                            )
                            / len(speedup_results),
                        },
                    },
                    f,
                    indent=2,
                )
            print(f"\nResults saved to: {args.output}")

    except Exception as e:
        print(f"Error processing results: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
