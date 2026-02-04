#!/bin/bash
# News Intelligence Digest Runner

# CRITICAL: Force Python/PyTorch to CPU only
# Prevents sentence-transformers from grabbing GPU away from Ollama
export CUDA_VISIBLE_DEVICES=""

# Log file
LOG_DIR="/mnt/nvme-fast/news-system/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/digest_$(date +%Y%m%d).log"

# Log start
echo "========================================" >> "$LOG_FILE"
echo "Started: $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Go to project directory
cd /home/will/news-system/news-digest

# Activate virtual environment
source /home/will/news-system/news-digest/venv/bin/activate

# Run the digest pipeline
python advanced_main.py >> "$LOG_FILE" 2>&1

# Log exit code
echo "Exit code: $?" >> "$LOG_FILE"
echo "Finished: $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"
