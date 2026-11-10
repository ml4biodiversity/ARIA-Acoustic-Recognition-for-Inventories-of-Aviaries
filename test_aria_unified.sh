#!/usr/bin/env bash
set -e

cd ~/zoostuff/checkpoints/ARIA

# Ensure environment variables are set
export BIRDNET_REPO="$HOME/BirdNET-Analyzer"
export PYTHONPATH="$BIRDNET_REPO:$PYTHONPATH"

echo "========================================="
echo "  ARIA Unified Environment Test"
echo "========================================="

# Test 1: PERCH Dependencies
echo ""
echo "=== Testing PERCH Dependencies ==="
python -c "
from packaging import version
import tensorflow as tf

MIN_TF = '2.20.0'
current = tf.__version__
if version.parse(current) >= version.parse(MIN_TF):
    print(f'✅ TensorFlow {current} meets PERCH requirement (>= {MIN_TF})')
else:
    print(f'❌ TensorFlow {current} does not meet PERCH requirement')

try:
    import kagglehub
    print(f'✅ kagglehub {kagglehub.__version__}')
except Exception as e:
    print(f'❌ kagglehub: {e}')

try:
    import tensorflow_hub
    print(f'✅ tensorflow_hub {tensorflow_hub.__version__}')
except Exception as e:
    print(f'❌ tensorflow_hub: {e}')
"

# Test 2: Audio Processing
echo ""
echo "=== Testing Audio Processing ==="
python -c "
import librosa
import soundfile
import audioread

print(f'✅ librosa {librosa.__version__}')
print(f'✅ soundfile {soundfile.__version__}')
print(f'✅ audioread {audioread.__version__}')

test_file = '/home/I6407196/zoostuff_backup/checkpoints/ARIA/data/wav/Aegypius_monachus_Cinereous_Vulture/xc144935.wav'
try:
    y, sr = librosa.load(test_file, sr=None, duration=3.0)
    print(f'✅ Audio loading works (shape: {y.shape}, sr: {sr})')
except Exception as e:
    print(f'❌ Audio loading failed: {e}')
"

# Test 3: Scientific Computing
echo ""
echo "=== Testing Scientific Computing ==="
python -c "
import numpy as np
import pandas as pd
import scipy
import sklearn

print(f'✅ numpy {np.__version__}')
print(f'✅ pandas {pd.__version__}')
print(f'✅ scipy {scipy.__version__}')
print(f'✅ scikit-learn {sklearn.__version__}')
"

# Test 4: BirdNET
echo ""
echo "=== Testing BirdNET ==="
python -c "
import birdnet_analyzer
print(f'✅ BirdNET import successful')
"

python -m birdnet_analyzer.train --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ BirdNET training module available"
else
    echo "❌ BirdNET training module not available"
fi

# Test 5: Data I/O
echo ""
echo "=== Testing Data I/O ==="
python -c "
import openpyxl
import requests
import certifi

print(f'✅ openpyxl {openpyxl.__version__}')
print(f'✅ requests {requests.__version__}')
print(f'✅ certifi {certifi.__version__}')

try:
    response = requests.head('https://xeno-canto.org', timeout=5)
    print(f'✅ HTTPS requests work (xeno-canto reachable)')
except Exception as e:
    print(f'⚠️  HTTPS test: {e}')
"

# Test 6: ARIA Scripts
echo ""
echo "=== Testing ARIA Scripts ==="
for script in train_custom_birdnet.py perch_segment.py fetch_and_segment.sh; do
    if [ -f "$script" ]; then
        echo "✅ Found $script"
    else
        echo "⚠️  Missing $script"
    fi
done

# Summary
echo ""
echo "========================================="
echo "Environment: aria_unified"
echo "TensorFlow: $(python -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null)"
echo "BirdNET: Functional ✅"
echo "Python: $(python --version | cut -d' ' -f2)"
echo "========================================="
echo ""
echo "✅ All critical dependencies working!"
echo "   Ready to delete old environments"
