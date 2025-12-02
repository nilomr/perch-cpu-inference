# Batch Inference Runner

Run large-scale bird species classification on multiple directories of audio data. Processes directories sequentially in the background, allowing you to safely close your terminal or VS Code session.

## Start Inference

```bash
# Run with custom paths
./tools/run_inference.sh /path/to/audio/directories /path/to/output

# Example: Process multiple site directories
./tools/run_inference.sh /data/acoustics/2025 ./output/2025
```

The script will process each subdirectory in the input path sequentially.

## Monitor Progress

```bash
./tools/monitor.sh          # Follow log live (Ctrl+C to stop)
./tools/monitor.sh status   # Quick progress summary
./tools/monitor.sh log      # Last 50 lines
```

## Stop Inference

```bash
./tools/monitor.sh kill
```

## Output Structure

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