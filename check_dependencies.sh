#!/usr/bin/env bash
set -e

echo "========================================="
echo "  ARIA Dependency Check (Unified Env)"
echo "========================================="

# Python
echo ""
echo "Python:"
python3 --version
python3 -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 10) else 1)" && echo "✅ Python 3.10" || echo "⚠️  Not Python 3.10"

# Environment
echo ""
echo "Environment: ${CONDA_DEFAULT_ENV:-none}"
if [[ "${CONDA_DEFAULT_ENV}" == "aria_unified" ]]; then
    echo "✅ Using unified environment"
else
    echo "⚠️  Expected aria_unified environment"
fi

# BirdNET
echo ""
echo "BirdNET:"
[ -d "$HOME/BirdNET-Analyzer" ] && echo "✅ Repo exists" || echo "❌ Repo missing"
[ -n "$BIRDNET_REPO" ] && echo "✅ BIRDNET_REPO set" || echo "⚠️  BIRDNET_REPO not set"
[[ "$PYTHONPATH" == *"BirdNET"* ]] && echo "✅ PYTHONPATH OK" || echo "⚠️  PYTHONPATH missing BirdNET"
python3 -c "import sys; sys.path.insert(0, '$HOME/BirdNET-Analyzer'); import birdnet_analyzer" 2>/dev/null && echo "✅ Importable" || echo "❌ Cannot import"

# TensorFlow version (critical for unified environment)
echo ""
echo "TensorFlow:"
TF_VER=$(python3 -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null)
if [[ -n "$TF_VER" ]]; then
    echo "✅ TensorFlow $TF_VER"
    if [[ "$TF_VER" == "2.20"* ]]; then
        echo "✅ TF 2.20.x (supports both BirdNET & PERCH)"
    else
        echo "⚠️  Expected TF 2.20.x for unified environment"
    fi
else
    echo "❌ TensorFlow not found"
fi

# Core packages
echo ""
echo "Core Packages:"
for pkg in numpy pandas librosa soundfile scipy sklearn openpyxl requests; do
    if python3 -c "import $pkg" 2>/dev/null; then
        ver=$(python3 -c "import $pkg; print(getattr($pkg, '__version__', '?'))" 2>/dev/null)
        echo "✅ $pkg ($ver)"
    else
        echo "❌ $pkg"
    fi
done

# PERCH-specific packages
echo ""
echo "PERCH Dependencies:"
for pkg in kagglehub tensorflow_hub; do
    if python3 -c "import $pkg" 2>/dev/null; then
        ver=$(python3 -c "import $pkg; print(getattr($pkg, '__version__', '?'))" 2>/dev/null)
        echo "✅ $pkg ($ver)"
    else
        echo "❌ $pkg"
    fi
done

# System tools
echo ""
echo "System:"
command -v ffmpeg &>/dev/null && echo "✅ ffmpeg" || echo "❌ ffmpeg"
command -v ffprobe &>/dev/null && echo "✅ ffprobe" || echo "❌ ffprobe"

# Environment variables
echo ""
echo "Environment Variables:"
[ -n "$XC_API_KEY" ] && echo "✅ XC_API_KEY" || echo "⚠️  XC_API_KEY not set"

# Config files
echo ""
echo "Config Files:"
for f in config/species_list.txt config/birds_not_in_birdnet.txt config/zoo_species_config.json; do
    [ -f "$f" ] && echo "✅ $f" || echo "❌ $f"
done

echo ""
echo "========================================="
echo "Check complete!"
echo ""
echo "To activate:"
echo "  micromamba activate aria_unified"
echo ""
echo "This unified environment supports:"
echo "  • BirdNET training & inference"
echo "  • PERCH segmentation"
echo "  • All data processing pipelines"
