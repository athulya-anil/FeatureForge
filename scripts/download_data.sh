#!/bin/bash

# FeatureForge - Data Download Script
# Downloads Criteo CTR dataset from Kaggle

set -e  # Exit on error

echo "=================================================="
echo "FeatureForge - Criteo Dataset Download"
echo "=================================================="

# Check if kaggle CLI is installed
if ! command -v kaggle &> /dev/null; then
    echo "❌ Kaggle CLI not found!"
    echo ""
    echo "Please install it:"
    echo "  pip install kaggle"
    echo ""
    echo "Then setup API credentials:"
    echo "  1. Go to https://www.kaggle.com/settings/account"
    echo "  2. Click 'Create New API Token'"
    echo "  3. Move kaggle.json to ~/.kaggle/"
    echo "  4. chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

# Check if kaggle credentials exist
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "❌ Kaggle credentials not found!"
    echo ""
    echo "Please setup API credentials:"
    echo "  1. Go to https://www.kaggle.com/settings/account"
    echo "  2. Click 'Create New API Token'"
    echo "  3. Move kaggle.json to ~/.kaggle/"
    echo "  4. chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

# Create data directory
DATA_DIR="data/raw"
mkdir -p "$DATA_DIR"

echo ""
echo "📥 Downloading Criteo dataset from Kaggle..."
echo "   (This may take a while - dataset is ~10GB)"
echo ""

# Download dataset
kaggle competitions download -c criteo-display-ad-challenge -p "$DATA_DIR"

echo ""
echo "📦 Extracting dataset..."
echo ""

# Extract
cd "$DATA_DIR"
unzip -o criteo-display-ad-challenge.zip

echo ""
echo "✅ Download complete!"
echo ""
echo "Files in $DATA_DIR:"
ls -lh

echo ""
echo "=================================================="
echo "Next steps:"
echo "  1. Run: python scripts/create_sample.py"
echo "  2. Run notebooks/02_eda.ipynb"
echo "  3. Run notebooks/03_baseline_model.ipynb"
echo "=================================================="
