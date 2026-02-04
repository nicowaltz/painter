#!/bin/bash

set -e

# Auto-shutdown expensive instances!
SHUTDOWN=false
GPUS=4

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --shutdown)
            SHUTDOWN=true
            # Check if user can use sudo for shutdown
            if ! sudo -n true 2>/dev/null; then
                echo "ERROR: --shutdown flag requires sudo privileges without password prompt."
                exit 1
            fi
            echo "Server will automatically shutdown after training completes!"
            shift
            ;;
        *)
            # Keep unhandled arguments
            remaining_args+=("$1")
            shift
            ;;
    esac
done

DIR="checkpoints/$(date +"%Y-%m-%d-%H%M%S")"

torchrun --standalone --nproc_per_node="$GPUS" train.py --dir "$DIR"

if [[ "$SHUTDOWN" == true ]]; then
    sudo poweroff
fi

