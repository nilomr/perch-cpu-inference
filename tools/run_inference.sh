#!/bin/bash
# Run Perch inference for all sites in a directory
# Usage: ./run_inference.sh [input_dir] [output_dir]
#
# Runs in background with nohup - safe to close SSH/VS Code
# Logs to: inference.log (main) and inference.pid (process ID)

set -euo pipefail

INPUT_ROOT="${1:-/mnt/bio-lv-colefs01/2025-wytham-acoustics}"
OUTPUT_ROOT="${2:-output/embeddings/2025}"
LOG_FILE="inference.log"
PID_FILE="inference.pid"

# Check if already running
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "ERROR: Inference already running (PID: $(cat "$PID_FILE"))"
    echo "Use: ./monitor.sh to check status, or ./monitor.sh kill to stop"
    exit 1
fi

# Validate input directory
if [ ! -d "$INPUT_ROOT" ]; then
    echo "ERROR: Input directory not found: $INPUT_ROOT"
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"

# Run the actual processing in background
nohup bash -c '
INPUT_ROOT="'"$INPUT_ROOT"'"
OUTPUT_ROOT="'"$OUTPUT_ROOT"'"
LOG_FILE="'"$LOG_FILE"'"

{
    echo "========================================"
    echo "INFERENCE STARTED: $(date)"
    echo "Input:  $INPUT_ROOT"
    echo "Output: $OUTPUT_ROOT"
    echo "========================================"
    echo ""

    for site_dir in "$INPUT_ROOT"/*/; do
        [ -d "$site_dir" ] || continue
        
        site_name=$(basename "$site_dir")
        site_output="$OUTPUT_ROOT/$site_name"
        
        echo "[$(date +%H:%M:%S)] Starting: $site_name"
        
        if python ../perch-onnx-inference.py \
            --audio-dir "$site_dir" \
            --output-dir "$site_output" \
            --workers 32 \
            --loader-threads 24 \
            --batch-size 4 \
            --max-cpus 120 \
            --max-ram-gb 10 \
            --no-resume 2>&1; then
            echo "[$(date +%H:%M:%S)] DONE: $site_name"
        else
            echo "[$(date +%H:%M:%S)] FAILED: $site_name"
        fi
        echo ""
    done

    echo "========================================"
    echo "ALL SITES COMPLETE: $(date)"
    echo "========================================"
    rm -f "'"$PID_FILE"'"
} >> "$LOG_FILE" 2>&1
' &

echo $! > "$PID_FILE"
echo "Inference started in background (PID: $(cat "$PID_FILE"))"
echo ""
echo "Commands:"
echo "  ./monitor.sh          - View live log"
echo "  ./monitor.sh status   - Check progress"
echo "  ./monitor.sh kill     - Stop inference"
