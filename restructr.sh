#!/bin/bash

# Define the base directory
base_dir="/home/ec2-user/Fisher/s3_zips"  # Replace with your actual path

# Loop through the directories in s3_zips
for dir in "$base_dir"/ID_*; do
  if [ -d "$dir" ]; then
    # Get the inner directory name (same as the outer)
    inner_dir="$dir/$(basename "$dir")"

    if [ -d "$inner_dir" ]; then
      # Move all .dcm files from the inner directory to the outer directory
      mv "$inner_dir"/*.dcm "$dir"

      # Remove the now-empty inner directory
      rmdir "$inner_dir"

      echo "Restructured $dir"
    else
      echo "Inner directory not found in $dir"
    fi
  else
    echo "$dir is not a directory"
  fi
done

echo "Restructuring completed."