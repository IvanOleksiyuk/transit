#!/bin/sh
#SBATCH --job-name=v0v2_LHCO
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --partition=private-dpnc-gpu
#SBATCH --nodes=1
#SBATCH --exclude=gpu[017-018,044,047,048,049]
#SBATCH --output=/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/job_output/TRANSITv0v2_LHCO-%A-%x_%a.out
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

# Record the start time
start_time=$(date +%s)
echo "Job started at: $(date)"

module load GCCcore/12.3.0 Python/3.11.3
cd scratch/sing_images/
scontrol show job $SLURM_JOB_ID
# SAVE SCONTROL PLACEHOLDER
singularity exec --nv -B /home/users/,/srv,/tmp hyperproject_container.sif \
 bash -c "cd /home/users/o/oleksiyu/WORK/hyperproject/ &&\
 python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run.py\
 --config-name TRANSITv0v2_LHCO general.subfolder=PAPER/\
 verbose_validation=0"

# Record the end time
end_time=$(date +%s)
echo "Job ended at: $(date)"