#!/usr/bin/env bash
# Activate ARIA unified environment
export BIRDNET_REPO="$HOME/BirdNET-Analyzer"
export PYTHONPATH="$BIRDNET_REPO:$PYTHONPATH"
micromamba activate aria_unified
echo "✅ ARIA unified environment activated"
echo "   - TensorFlow 2.20.0 (supports both BirdNET and PERCH)"
echo "   - Python $(python --version | cut -d' ' -f2)"
