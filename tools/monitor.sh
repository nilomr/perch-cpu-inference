#!/bin/bash
# Monitor and control the inference process
# Usage: ./monitor.sh [command]
#
# Commands:
#   (none)    - Follow log output live (Ctrl+C to stop watching)
#   status    - Show current progress
#   kill      - Stop the inference process
#   log       - Show last 50 lines of log

PID_FILE="inference.pid"
LOG_FILE="inference.log"

case "${1:-}" in
    status)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "Status: RUNNING (PID: $(cat "$PID_FILE"))"
        else
            echo "Status: NOT RUNNING"
            [ -f "$PID_FILE" ] && rm -f "$PID_FILE"
        fi
        echo ""
        
        if [ -f "$LOG_FILE" ]; then
            echo "=== Recent Progress ==="
            grep -E "^\[|Starting:|DONE:|FAILED:|COMPLETE" "$LOG_FILE" | tail -20
            echo ""
            echo "=== Sites Completed ==="
            grep -c "DONE:" "$LOG_FILE" 2>/dev/null || echo "0"
        fi
        ;;
        
    kill)
        if [ -f "$PID_FILE" ]; then
            pid=$(cat "$PID_FILE")
            if kill -0 "$pid" 2>/dev/null; then
                echo "Stopping inference (PID: $pid)..."
                kill "$pid"
                # Also kill any child python processes
                pkill -P "$pid" 2>/dev/null || true
                sleep 1
                echo "Stopped."
            else
                echo "Process not running."
            fi
            rm -f "$PID_FILE"
        else
            echo "No PID file found. Inference not running."
        fi
        ;;
        
    log)
        if [ -f "$LOG_FILE" ]; then
            tail -50 "$LOG_FILE"
        else
            echo "No log file found."
        fi
        ;;
        
    *)
        if [ -f "$LOG_FILE" ]; then
            echo "Following log (Ctrl+C to stop)..."
            echo ""
            tail -f "$LOG_FILE"
        else
            echo "No log file found. Run ./run_inference.sh first."
        fi
        ;;
esac
