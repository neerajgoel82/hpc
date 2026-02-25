#!/bin/bash
# Script to restructure python/samples phases with classwork and homework folders

SAMPLES_DIR="python/samples"
PHASES=(
    "phase1-foundations"
    "phase2-intermediate"
    "phase3-oop"
    "phase4-advanced"
    "phase5-datascience"
    "phase6-pytorch"
)

for phase in "${PHASES[@]}"; do
    echo "Processing $phase..."
    phase_dir="$SAMPLES_DIR/$phase"

    # Skip if phase directory doesn't exist
    if [ ! -d "$phase_dir" ]; then
        echo "  Skipping $phase (not found)"
        continue
    fi

    # Create classwork and homework directories
    mkdir -p "$phase_dir/classwork"
    mkdir -p "$phase_dir/homework"

    # Move all .py files to classwork (but not subdirectories)
    for file in "$phase_dir"/*.py; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            echo "  Moving $filename to classwork/"
            git mv "$file" "$phase_dir/classwork/"
        fi
    done

    # Handle subdirectories (like phase5-datascience has subdirectories)
    for subdir in "$phase_dir"/*/; do
        if [ -d "$subdir" ] && [ "$subdir" != "$phase_dir/classwork/" ] && [ "$subdir" != "$phase_dir/homework/" ]; then
            subdir_name=$(basename "$subdir")
            if [ "$subdir_name" != "classwork" ] && [ "$subdir_name" != "homework" ]; then
                echo "  Moving subdirectory $subdir_name to classwork/"
                git mv "$subdir" "$phase_dir/classwork/"
            fi
        fi
    done

    echo "  Created classwork/ and homework/ directories for $phase"
done

echo ""
echo "Phase restructuring complete!"
echo "Next: Extract exercises to homework files"
