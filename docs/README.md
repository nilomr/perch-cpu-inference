# PhenoScale Bioacoustics - Perch v2 Inference

High-performance bird species classification using Google's Perch v2 model for large-scale bioacoustics inference on CPUs. Work in progress.

## Features

- **ONNX Inference**: Optimized CPU inference
- **TFLite Inference**: Lightweight deployment with parallel processing

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download or compile Models

**ONNX Model:**
Download the pre-optimized ONNX model:
```bash
wget https://huggingface.co/justinchuby/Perch-onnx/resolve/main/perch_v2.onnx -P models/perch_v2/
```

**TFLite Model:**
Compile the model from TensorFlow Hub:
```bash
python perch-tflite-inference.py compile-model
```

### 3. Run ONNX inference

```bash
python perch-onnx-inference.py --audio-dir ./data/test-data --output-dir ./output
```

### 4. Run TFLite inference

After compiling the model (step 2), run inference:

```bash
python perch-tflite-inference.py run-inference --audio-dir ./data/test-data
```

### 5. Visualize Results

```bash
python scripts/visualization/visualize.py data/test-data/wren-test.wav output/predictions_partitioned/checkpoint_0000.csv --output visualization.png
```

## Project structure

```
├── scripts/
│   ├── inference/
│   │   ├── perch-onnx-inference.py      # ONNX inference script
│   │   └── perch-tflite-inference.py    # TFLite inference script
│   ├── benchmark/
│   │   ├── perch-onnx-benchmark.py
│   │   └── analyze_benchmark_speedup.py
│   └── visualization/
│       ├── visualize.py                 # Visualization tool
│       ├── visualize-embeddings-class.py
│       ├── visualize-embeddings-time.py
│       ├── data-report-plots.py
│       └── data-report.py
├── docs/
│   ├── README.md                        # Detailed usage
│   └── INFERENCE_README.md              # Inference runner guide
├── data/
│   └── test-data/                       # Sample audio files
├── models/
│   └── perch_v2/                        # Model directory (models downloaded separately)
│       ├── perch_v2.onnx               # ONNX model (~750MB, download separately)
│       ├── perch_v2.tflite             # TFLite model (compile using script)
│       ├── classes.json                # Species classes
│       └── metadata files
├── output/                              # Inference outputs
├── results/                             # Benchmark results
├── tools/
│   ├── run_inference.sh                 # Batch inference runner
│   └── monitor.sh                       # Process monitor
└── requirements.txt                     # Python dependencies
```

## Model details

- **Perch v2**: Google's classifier pre-trained on a multi-taxa dataset ([Paper](https://arxiv.org/abs/2508.04665))
- **Input**: 5-second audio chunks at 32kHz
- **Output**: Top-10 (configurable) species predictions with logits. These can be used to, e.g., estimate call density (see [Navine & Denton et. al., 2024](https://arxiv.org/html/2402.15360v1)).
- **Embeddings**: 1536-dimensional feature vectors

## Performance

### ONNX

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

## Usage examples

### Batch processing large datasets

```bash
# ONNX with 32 workers, float16 compression
python perch-onnx-inference.py \
  --audio-dir /path/to/audio \
  --output-dir ./output \
  --workers 32 \
  --batch-size 32 \
  --float16 \
  --checkpoint-interval 50
```

### Benchmarking

```bash
# Test different worker counts
python perch-tflite-inference.py benchmark ./data/test-data
```

## Output formats

### Embeddings (Parquet)
- Partitioned by checkpoint_id
- Columns: file, chunk_idx, start_time, end_time, emb_0..emb_1535

### Predictions (CSV)
- Top-10 species per 5-second chunk
- Columns: file, start_time, end_time, top1..top10, score1..score10

## Visualization

- Spectrogram (0-12kHz)
- Time-aligned predictions table

## Dependencies

Key packages:

- `onnxruntime` - ONNX inference engine
- `tensorflow` - TFLite inference support
- `pyarrow` - Efficient Parquet file storage
- `librosa` - Audio file processing and resampling
- `soundfile` - Audio file I/O
- `matplotlib` - Visualization
- `rich` - Progress bars and console output
- `typer` - Command-line interface

See `requirements.txt` for the complete list.

## Model sources

- ONNX model: Manually optimized onnx version https://huggingface.co/justinchuby/Perch-onnx/tree/main by [Justin Chu](https://github.com/justinchuby)
- TFLite model: Compiled from [TensorFlow Hub](https://github.com/kitzeslab/bioacoustics-model-zoo/blob/main/bioacoustics_model_zoo/perch_v2.py) in the bioacoustics model zoo by Lapp, S., and Kitzes, J., 2025 ("Bioacoustics Model Zoo version 0.12.2". https://github.com/kitzeslab/bioacoustics-model-zoo).

## Contributing

This is a research tool for bioacoustics analysis. For issues or improvements, please open a GitHub issue.

## License

See LICENSE file for details.