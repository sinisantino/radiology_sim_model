#!/bin/bash

# Simple batch script to run image generation multiple times
# Each run will generate unique images due to different random seeds

# Number of times to run the command
NUM_RUNS=10

echo "Running image generation $NUM_RUNS times..."
echo "Command: python generate_medical_images.py --anatomy prostate --body_region abdomen --num_samples 2"
echo ""

for i in $(seq 1 $NUM_RUNS); do
    echo "Run $i/$NUM_RUNS..."
    python generate_medical_images.py --anatomy prostate --body_region abdomen --num_samples 2
    echo "Run $i completed."
    echo ""
done

echo "All $NUM_RUNS runs completed!"
