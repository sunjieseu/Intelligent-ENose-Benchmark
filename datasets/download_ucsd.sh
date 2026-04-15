#!/bin/bash

# Download UCSD Gas Sensor Array Drift Dataset
# This script downloads the dataset from UCI Machine Learning Repository

echo "=========================================="
echo "Downloading UCSD Gas Sensor Drift Dataset"
echo "=========================================="

# Create data directory
mkdir -p data/ucsd

# URLs for UCSD dataset
UCSD_URL="https://archive.ics.uci.edu/ml/machine-learning-databases/00368/Drift_Dataset.zip"
UCSD_URL_ALT="https://archive.ics.uci.edu/static/public/368/gas+sensor+array+drift+dataset.zip"

# Download the dataset
echo "Downloading dataset from UCI repository..."
if command -v wget &> /dev/null; then
    wget -O data/ucsd/Drift_Dataset.zip "$UCSD_URL" || \
    wget -O data/ucsd/Drift_Dataset.zip "$UCSD_URL_ALT"
elif command -v curl &> /dev/null; then
    curl -L -o data/ucsd/Drift_Dataset.zip "$UCSD_URL" || \
    curl -L -o data/ucsd/Drift_Dataset.zip "$UCSD_URL_ALT"
else
    echo "Error: Neither wget nor curl is available. Please install one of them."
    exit 1
fi

# Check if download was successful
if [ $? -eq 0 ]; then
    echo "Download completed successfully!"
else
    echo "Error: Download failed!"
    exit 1
fi

# Extract the dataset
echo "Extracting dataset..."
unzip -o data/ucsd/Drift_Dataset.zip -d data/ucsd/

# Verify extraction
if [ $? -eq 0 ]; then
    echo "Extraction completed successfully!"
else
    echo "Error: Extraction failed!"
    exit 1
fi

# Clean up
echo "Cleaning up..."
rm -f data/ucsd/Drift_Dataset.zip

# Display dataset information
echo ""
echo "=========================================="
echo "Dataset Download Complete!"
echo "=========================================="
echo "Location: data/ucsd/"
echo ""
echo "Dataset Contents:"
ls -lh data/ucsd/
echo ""
echo "Files:"
echo "  - Datos3096.txt: Main data file (13,910 samples, 16 sensors + labels)"
echo "  - ReadMe.txt: Dataset documentation"
echo ""
echo "Next Steps:"
echo "  1. Use datasets/dataset_loaders.py to load and preprocess the data"
echo "  2. See datasets/README.md for usage examples"
echo ""
echo "Happy researching! 🚀"
echo "=========================================="
