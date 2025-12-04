# Batch processing

Run large-scale bird species classification on multiple directories of audio data. Processes directories sequentially in the background, allowing you to safely close your terminal or SSH session.

## Running batch inference

```bash
# Run with custom paths
./tools/run_inference.sh /path/to/audio/directories /path/to/output

# Example: Process multiple site directories
./tools/run_inference.sh /data/acoustics/2025 ./output/2025
```

The script will process each subdirectory in the input path sequentially.

## Monitoring progress

```bash
./tools/monitor.sh          # Follow log live (Ctrl+C to stop)
./tools/monitor.sh status   # Quick progress summary
./tools/monitor.sh log      # Last 50 lines
```

## Stopping inference

```bash
./tools/monitor.sh kill
```

## Output structure

```
output/
├── site1/
│   ├── embeddings_partitioned/   # Parquet: 1536-dim embeddings per 5s chunk
│   └── predictions_partitioned/  # CSV: top-10 species predictions per chunk
├── site2/
└── ...
```

Each site directory will contain its own embeddings and predictions.

## Notes

- Sites are processed one at a time to manage system resources
- The process runs in the background and survives SSH/terminal disconnections
- View `inference.log` for detailed progress and any errors
- Process ID is saved to `inference.pid` for monitoring and control
- Resume functionality allows restarting failed runs from checkpoints

## Parsing inference logs

Use `parse_inference_log` to analyze completed inference runs:

```bash
# Print summary and create CSV alongside the log
./tools/parse_inference_log.sh /path/to/inference.log

# Write results to a specific CSV file
./tools/parse_inference_log.sh /path/to/inference.log /tmp/results.csv
```

Prerequisites: python3 available on PATH. Uses only the Python standard library.

## Real-world performance example

Below are statistics from processing the 2025 Wytham Woods acoustics dataset:

### Execution details

| Metric | Value |
|--------|-------|
| **Total jobs run** | 160 |
| **Total files processed** | 1,076,405 |
| **Total duration of audio files** | 747.5 days (17,940 hours) |
| **Total running time** | 40.85 hours (147,073 seconds) |
| **Average throughput** | 7.23 files/second |
| **Average speed** | 433.6x realtime |

### System configuration

| Parameter | Value |
|-----------|-------|
| **Inference workers** | 32 |
| **Threads per worker** | 3 |
| **Batch size** | 4 |
| **RAM budget** | 10.0 GB |
| **Embedding format** | float32 |

### Performance range

| Metric | Value |
|--------|-------|
| **Throughput** | 5.41 - 14.94 files/sec |
| **Speed** | 324.9x - 896.3x realtime |

### Output sizes

| Output | Size |
|--------|------|
| **Total embeddings** | 118.33 GB |
| **Total predictions** | 5,936.32 MB (5.80 GB) |
