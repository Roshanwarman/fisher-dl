#!/bin/bash

# Define the source and destination directories
source_dir="/home/ec2-user/Fisher/s3_zips"
dest_dir="/home/ec2-user/Fisher/Zach-Scored"

# Check if the source directory exists
if [ ! -d "$source_dir" ]; then
  echo "Error: Source directory '$source_dir' not found."
  exit 1
fi

# Create the destination directory if it doesn't exist
mkdir -p "$dest_dir"

# Copy all directories starting with "ID_"
cp -r "$source_dir"/ID_* "$dest_dir"

echo "Copying completed."