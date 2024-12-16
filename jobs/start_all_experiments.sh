#!/bin/bash

# Directory containing the .sh files
JOB_DIR="/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/adv_edv"

# Check if the directory exists
if [ ! -d "$JOB_DIR" ]; then
  echo "Directory $JOB_DIR does not exist."
  exit 1
fi

# Loop through all .sh files in the directory
for job in "$JOB_DIR"/*.sh; 
do
  if [ -f "$job" ]; then
    echo "Starting SLURM job: $job"
    sbatch "$job"
  else
    echo "No .sh files found in the directory."
  fi
done

echo "All SLURM jobs have been submitted."