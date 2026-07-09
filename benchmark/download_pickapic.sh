#!/bin/bash

# Base URL for HuggingFace dataset files
BASE_URL="https://huggingface.co/datasets/pickapic-anonymous/pickapic_v1/resolve/main/data"

# Array of files to download
FILES=(
    "test-00000-of-00010-a4b2a15ad966dd4c.parquet"
    "test-00001-of-00010-203a8d11c0186021.parquet"
    "test-00002-of-00010-135c9b4b5e6faa36.parquet"
    "test-00003-of-00010-5c54b6b9c0788603.parquet"
    "test-00004-of-00010-f14d66f4b59a23f2.parquet"
    "test-00005-of-00010-3f4018acaeb256f9.parquet"
    "test-00006-of-00010-46b7ab2bdeeff8ee.parquet"
    "test-00007-of-00010-b0354eeab94f6a4b.parquet"
    "test-00008-of-00010-6fd9d1251a0c366b.parquet"
    "test-00009-of-00010-1ea8cf5f3be9b44e.parquet"
    "test_unique-00000-of-00001-c163d5839d8aed9f.parquet"
)

# Download each file
for file in "${FILES[@]}"; do
    echo "Downloading $file..."
    wget -c "${BASE_URL}/${file}" -O "${file}"
    if [ $? -eq 0 ]; then
        echo "✓ Successfully downloaded $file"
    else
        echo "✗ Failed to download $file"
    fi
done

echo "All downloads completed!"
