#!/bin/bash

# Set the parent directory
PARENT_DIR="/home/ec2-user/Fisher/Data/ID_2db7ee14"

# Loop through each subdirectory
for subdir in "$PARENT_DIR"/*/; do
    if [ -d "$subdir" ]; then
        count=$(find "$subdir" -type f | wc -l)
        echo "$(basename "$subdir"): $count files"
    fi
done
