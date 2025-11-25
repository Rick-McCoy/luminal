#!/bin/bash
# Download MNIST dataset

set -e

DATA_DIR="data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

BASE_URL="https://storage.googleapis.com/cvdf-datasets/mnist"

# Download files if they don't exist
for file in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
    if [ ! -f "${file%.gz}" ]; then
        echo "Downloading $file..."
        curl -O "$BASE_URL/$file"
        gunzip -f "$file"
    else
        echo "$file already exists, skipping..."
    fi
done

echo "MNIST dataset ready in $DATA_DIR/"

