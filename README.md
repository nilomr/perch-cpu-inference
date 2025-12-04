# Perch v2 CPU Inference

High-performance acoustic embedding extraction and species classification using Google's Perch v2 model for large-scale bioacoustics inference on CPUs.

![Example of 2D projection of resulting embeddings](./docs/temporal_embeddings.jpg)

## Features

- **High-performance CPU inference** — Up to 750x realtime with ONNX or TFLite
- **Batch processing** — Large-scale dataset processing with checkpointing and resume
- **Visualization** — Spectrogram overlays and detection time series charts

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download model
wget https://huggingface.co/justinchuby/Perch-onnx/resolve/main/perch_v2.onnx -P models/perch_v2/

# Run inference
python scripts/inference/perch-onnx-inference.py --audio-dir ./data/test-data --output-dir ./output
```

## Documentation

| Guide | Description |
|-------|-------------|
| **[Usage Guide](docs/usage.md)** | Installation, running inference, visualization, output formats, benchmarks |
| **[Batch Processing](docs/batch-processing.md)** | Large-scale inference, monitoring, log parsing |

## Model

- **Perch v2**: Google's classifier pre-trained on a multi-taxa dataset ([Paper](https://arxiv.org/abs/2508.04665))
- **Input**: 5-second audio chunks at 32kHz
- **Output**: Species predictions and 1536-dimensional embeddings

### Credits

| Format | Source | Prepared by |
|--------|--------|-------------|
| ONNX | [Hugging Face](https://huggingface.co/justinchuby/Perch-onnx) | Justin Chu ([@justinchuby](https://github.com/justinchuby)) |
| TFLite | [Bioacoustics Model Zoo](https://github.com/kitzeslab/bioacoustics-model-zoo) | Lapp, S., and Kitzes, J. (2025) |

## Project structure

```
├── scripts/inference/         # Inference scripts
│   ├── perch-onnx-inference.py    # Main ONNX inference script
│   └── perch-tflite-inference.py  # TFLite inference script
├── scripts/                   # Benchmarking and visualization tools
├── tools/                     # Shell scripts for batch processing
├── docs/                      # Documentation
├── models/                    # Model files (downloaded separately)
└── data/                      # Test data
```

## License

See [LICENSE](LICENSE) file for details.

---

<small>Developed as part of the PhenoScale project, UKRI Frontiers grant EP/X024520/1 awarded to Ben Sheldon, University of Oxford.</small>