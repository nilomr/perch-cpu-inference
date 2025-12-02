# High-performance bioacoustics inference

_Very_ high-performance acoustic data embedding extraction and classification using Google's Perch v2 model for large-scale bioacoustics inference on CPUs.

This repository provides optimized inference tools for running Perch v2 on bioacoustics data, with support for both ONNX and TFLite formats, very efficient parallel processing, and some basic visualization.

![Example of 2D projection of resulting embeddings](./docs/temporal_embeddings.jpg)

<small>Developed as part of the PhenoScale project, UKRI Frontiers grant EP/X024520/1 awarded to Ben Sheldon, University of Oxford.</small>

## Features

- **High-performance inference**: Optimized CPU inference with parallel processing; ONNX and TFLite model support
- **Batch processing**: Large-scale dataset processing with checkpointing and resume functionality
- **Visualization**: Spectrogram and prediction overlays for validating results
- **Benchmarking**: Performance analysis tools
- **Monitoring**: Background process management for long-running jobs

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download or compile Models

**ONNX Model:**
```bash
wget https://huggingface.co/justinchuby/Perch-onnx/resolve/main/perch_v2.onnx -P models/perch_v2/
```

**TFLite Model:** (optional, not tested as much)
```bash
python perch-tflite-inference.py compile-model
```

Note: Models will be saved to `models/perch_v2/` directory.

### 3. Run inference

```bash
# ONNX inference (recommended for performance)
python perch-onnx-inference.py --audio-dir ./data/test-data --output-dir ./output

# TFLite inference
python perch-tflite-inference.py run-inference --audio-dir ./data/test-data
```

### 4. Visualize results (optional)

```bash
python scripts/visualization/visualize.py data/test-data/wren-test.wav output/predictions_partitioned/checkpoint_0000.csv --output visualization.png
```

## Project structure

```
├── scripts/                    # Python scripts organized by function
│   ├── inference/             # Inference scripts
│   ├── benchmark/             # Benchmarking tools
│   └── visualization/         # Visualization and reporting
├── docs/                      # Documentation
│   ├── README.md             # Detailed usage guide
│   └── INFERENCE_README.md   # Inference runner guide
├── data/                      # Test and sample data
├── models/                    # Model files
├── output/                    # Inference outputs
├── results/                   # Benchmark results
├── tools/                     # Shell scripts and utilities
└── requirements.txt           # Python dependencies
```

## Documentation

- **[Detailed use guide](docs/README.md)**: Complete installation, usage, and performance information
- **[Inference runner guide](docs/INFERENCE_README.md)**: Instructions for batch processing large datasets

## Model details

- **Perch v2**: Google's classifier pre-trained on a multi-taxa dataset ([Paper](https://arxiv.org/abs/2508.04665))
- **Input**: 5-second audio chunks at 32kHz
- **Output**: Species predictions and 1536-dimensional embeddings

## Model sources & credits

This project uses optimized versions of Google's Perch v2 model prepared by the following contributors:

### ONNX model
- **Prepared by**: Justin Chu ([@justinchuby](https://github.com/justinchuby))
- **Source**: [Hugging Face - Perch-onnx](https://huggingface.co/justinchuby/Perch-onnx/tree/main)
- **Description**: Manually optimized ONNX version for improved CPU inference performance

### TFLite model
- **Prepared by**: Lapp, S., and Kitzes, J. (2025)
- **Citation**: "Bioacoustics Model Zoo version 0.12.2"
- **Source**: [Bioacoustics Model Zoo](https://github.com/kitzeslab/bioacoustics-model-zoo)
- **Description**: Compiled from TensorFlow Hub for lightweight deployment

Please cite these if you use this repository in your research.

**Note**: The original Perch v2 model was developed by Google researchers. This project only provides optimized inference implementations.


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


## Contributing

This is a research tool for bioacoustics analysis. For issues or improvements, please open a GitHub issue.

## License

See LICENSE file for details.