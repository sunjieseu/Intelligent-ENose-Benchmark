#!/bin/bash

# Download CQU E-Nose Drift Dataset
# This script downloads the CQU drift dataset

echo "=========================================="
echo "Downloading CQU E-Nose Drift Dataset"
echo "=========================================="

# Create data directory
mkdir -p data/cqu

# Note: CQU dataset may require manual download from the authors
# or from a specific repository. This script provides a template.

echo "IMPORTANT: The CQU dataset may require:"
echo "  1. Contact with the authors for access"
echo "  2. Or download from a specific repository"
echo ""
echo "Please check the following resources:"
echo "  - Original paper: Zhang et al., IEEE Sensors Journal, 2019"
echo "  - Authors' institutional repository"
echo ""

# Example URL (replace with actual URL if available)
CQU_URL="https://example.com/cqu_enose_dataset.zip"

# Check if URL is placeholder
if [[ "$CQU_URL" == *"example.com"* ]]; then
    echo "⚠️  Placeholder URL detected. Manual download required."
    echo ""
    echo "Manual Download Steps:"
    echo "  1. Visit the dataset repository URL"
    echo "  2. Download the CQU E-Nose Drift Dataset"
    echo "  3. Place files in: data/cqu/"
    echo "  4. Expected files: batch1.csv, batch2.csv, batch3.csv"
    echo ""
    exit 0
fi

# Download the dataset
echo "Downloading dataset..."
if command -v wget &> /dev/null; then
    wget -O data/cqu/cqu_dataset.zip "$CQU_URL"
elif command -v curl &> /dev/null; then
    curl -L -o data/cqu/cqu_dataset.zip "$CQU_URL"
else
    echo "Error: Neither wget nor curl is available."
    exit 1
fi

# Extract
if [ -f data/cqu/cqu_dataset.zip ]; then
    echo "Extracting dataset..."
    unzip -o data/cqu/cqu_dataset.zip -d data/cqu/
    rm -f data/cqu/cqu_dataset.zip
    echo "Extraction completed!"
fi

# Display dataset information
echo ""
echo "=========================================="
echo "CQU Dataset Download Complete!"
echo "=========================================="
echo "Location: data/cqu/"
echo ""
echo "Expected Contents:"
ls -lh data/cqu/ 2>/dev/null || echo "No files found. Manual download may be required."
echo ""
echo "Next Steps:"
echo "  1. Verify data files are present"
echo "  2. Use datasets/dataset_loaders.py to load the data"
echo "=========================================="
