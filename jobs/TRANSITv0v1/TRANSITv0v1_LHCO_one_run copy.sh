#!/bin/sh
#SBATCH --job-name=v0v1_LHCO
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --partition=shared-gpu,private-dpnc-gpu
#SBATCH --nodes=1
#SBATCH --output=/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/job_output/TRANSITv0v1_LHCO-%A-%x_%a.out
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

# Record the start time
start_time=$(date +%s)
echo "Job started at: $(date)"

module load GCCcore/12.3.0 Python/3.11.3
cd scratch/sing_images/
singularity exec --nv -B /home/users/,/srv,/tmp hyperproject_container.sif \
 bash -c "cd /home/users/o/oleksiyu/WORK/hyperproject/ &&\
 python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run.py\
 --config-name TRANSITv0v1_LHCO general.subfolder=TEST_jobs/\
 verbose_validation=1"

# Record the end time
end_time=$(date +%s)
echo "Job ended at: $(date)"