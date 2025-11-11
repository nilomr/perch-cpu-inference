# Phenoscale Bioacoustics - Perch v2 Inference

High-performance bird species classification using Google's Perch v2 model for large-scale bioacoustics inference on CPUs. Work in progress.

## Features

- **ONNX Inference**: Optimized CPU inference
- **TFLite Inference**: Lightweight deployment with parallel processing

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run ONNX Inference

```bash
python perch-onnx-inference.py --audio-dir ./test-data --output-dir ./output-onnx
```

### 3. Run TFLite Inference

First, compile the model (downloads and converts Perch v2):

```bash
python perch-tflite-inference.py compile-model
```

Then run inference:

```bash
python perch-tflite-inference.py run-inference ./test-data
```

### 4. Visualize Results

```bash
python visualize.py test-data/wren-test.wav output-onnx/predictions_partitioned/checkpoint_0000.csv --output visualization.png
```

## Project Structure

```
├── perch-onnx-inference.py      # ONNX inference script
├── perch-tflite-inference.py    # TFLite inference script
├── visualize.py                 # Visualization tool
├── models/
│   └── perch_v2/               # Pre-trained models
│       ├── perch_v2.onnx       # ONNX model
│       ├── perch_v2.tflite     # TFLite model
│       ├── classes.json        # Species classes
│       └── metadata files
├── test-data/                  # Sample audio files
└── requirements.txt            # Python dependencies
```

## Model Details

- **Perch v2**: Google's bird species classifier trained on millions of audio clips
- **Input**: 5-second audio chunks at 32kHz
- **Output**: Top-10 (configurable) species predictions with logits
- **Embeddings**: 1536-dimensional feature vectors

## Performance

- **ONNX**: ~550x real time using 24 workers. Benchmark coming.

## Usage Examples

### Batch Processing Large Datasets

```bash
# ONNX with 32 workers, float16 compression
python perch-onnx-inference.py \
  --audio-dir /path/to/audio \
  --output-dir ./results \
  --workers 32 \
  --batch-size 32 \
  --float16 \
  --checkpoint-interval 50
```

### Benchmarking

```bash
# Test different worker counts
python perch-tflite-inference.py benchmark ./test-data
```

## Output Formats

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

Key packages, loosely:

- `onnxruntime` - ONNX inference
- `tensorflow` - TFLite inference
- `pyarrow` - Parquet storage
- `librosa` - Audio processing
- `matplotlib` - Visualization
- `rich` - Progress bars
- `typer` - CLI interface

## Model Sources

- ONNX model: Manually optimized onnx version https://huggingface.co/justinchuby/Perch-onnx/tree/main by [Justin Chu](https://github.com/justinchuby)
- TFLite model: Compiled from [TensorFlow Hub](https://github.com/kitzeslab/bioacoustics-model-zoo/blob/main/bioacoustics_model_zoo/perch_v2.py) in the bioacoustics mdoel zoo, compiled by Lapp, S., and Kitzes, J., 2025 ("Bioacoustics Model Zoo version 0.12.2". https://github.com/kitzeslab/bioacoustics-model-zoo).

## Contributing

This is a research tool for bioacoustics analysis. For issues or improvements, please open a GitHub issue.

## License

See LICENSE file for details.