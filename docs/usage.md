# Usage guide

Detailed instructions for running Perch v2 inference, configuring options, and visualizing results.

## Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Key packages:
- `onnxruntime` - ONNX inference engine
- `tensorflow` - TFLite inference support
- `pyarrow` - Efficient Parquet file storage
- `librosa` - Audio file processing and resampling
- `soundfile` - Audio file I/O
- `matplotlib` - Visualization
- `rich` - Progress bars and console output
- `typer` - Command-line interface

### 2. Download or compile models

**ONNX Model (recommended):**
```bash
wget https://huggingface.co/justinchuby/Perch-onnx/resolve/main/perch_v2.onnx -P models/perch_v2/
```

**TFLite Model:**
```bash
python scripts/inference/perch-tflite-inference.py compile-model
```

Models are saved to `models/perch_v2/`.

## Running inference

### ONNX inference (recommended)

```bash
python scripts/inference/perch-onnx-inference.py --audio-dir ./data/test-data --output-dir ./output
```

### TFLite inference

```bash
python scripts/inference/perch-tflite-inference.py run-inference --audio-dir ./data/test-data
```

### Advanced options

```bash
# ONNX with custom settings
python scripts/inference/perch-onnx-inference.py \
  --audio-dir /path/to/audio \
  --output-dir ./output \
  --workers 32 \
  --batch-size 32 \
  --float16 \
  --checkpoint-interval 50
```

## Output formats

### Embeddings (Parquet)

- Partitioned by checkpoint_id
- Columns: `file`, `chunk_idx`, `start_time`, `end_time`, `emb_0`..`emb_1535`

### Predictions (CSV)

- Top-10 species per 5-second chunk
- Columns: `file`, `start_time`, `end_time`, `top1`..`top10`, `score1`..`score10`

### Output structure

```
output/
├── embeddings_partitioned/   # Parquet: 1536-dim embeddings per 5s chunk
└── predictions_partitioned/  # CSV: top-10 species predictions per chunk
```

## Visualization

### Spectrogram with predictions

```bash
python scripts/visualization/visualize.py \
  data/test-data/wren-test.wav \
  output/predictions_partitioned/checkpoint_0000.csv \
  --output visualization.png
```

Creates a spectrogram (0-12kHz) with time-aligned predictions table.

### Detection time series

Analyze prediction outputs to create time series plots of species detections:

```bash
python scripts/visualization/plot_detection_timeseries.py \
  /path/to/predictions_dir \
  --output plot.png \
  --top-n 10 \
  --logit-threshold 11.0
```

Features:
- Aggregates detections by day from AudioMoth filename timestamps
- Plots top N species by total detection count
- Configurable logit threshold filtering (raw logit scores, can be negative)
- Counts ALL detections above threshold, not just top predictions
- Automatically includes related site deployments (e.g., A4-2, A4-3 when A4 is specified)

> **Note**: This is useful for quick exploratory analysis, but should not be used for anything serious without validation and proper modelling.

![Time series plot example](B4-detections.png)

## Benchmarking

```bash
# Test different worker counts
python scripts/inference/perch-tflite-inference.py benchmark ./data/test-data
```

## Performance benchmarks

### ONNX performance

*Benchmark conducted with 100 60-second audio files across 3 runs per configuration*

| Rank | Workers | Batch Size | Time (s) | Throughput (files/sec) | **Speedup** | CPU % | Memory (MB) |
|:----:|:-------:|:----------:|:--------:|:----------------------:|:-----------:|:-----:|:-----------:|
| 1 | 32 | 4 | 11.7±0.1 | 12.54 | **752.4x** | 292% | 34,071 |
| 2 | 32 | 8 | 12.4±0.2 | 11.66 | **699.8x** | 285% | 36,271 |
| 3 | 32 | 16 | 14.4±0.3 | 9.34 | **560.4x** | 256% | 45,292 |
| 4 | 16 | 8 | 16.6±0.9 | 7.81 | **468.8x** | 201% | 23,167 |
| 5 | 16 | 16 | 17.3±0.4 | 7.39 | **443.6x** | 199% | 26,842 |
| 6 | 16 | 4 | 17.4±0.3 | 7.33 | **440.0x** | 204% | 21,186 |
| 7 | 8 | 8 | 19.4±0.5 | 6.38 | **383.0x** | 192% | 12,866 |
| 8 | 8 | 4 | 20.1±1.0 | 6.08 | **365.0x** | 195% | 11,839 |
| 9 | 8 | 16 | 21.6±1.3 | 5.62 | **337.0x** | 169% | 15,674 |
| 10 | 2 | 4 | 43.1±2.6 | 2.53 | **152.0x** | 165% | 4,395 |
| 11 | 2 | 8 | 45.2±6.2 | 2.45 | **146.8x** | 142% | 4,752 |
| 12 | 2 | 16 | 44.8±1.5 | 2.43 | **145.6x** | 135% | 5,544 |
