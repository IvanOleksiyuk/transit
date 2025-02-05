#!/bin/sh
#SBATCH --job-name=v0v2_LHCO
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
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
cd sing_images/
singularity exec --nv -B /home/users/,/srv,/tmp hyperproject_container.sif \
 bash -c "cd /home/users/o/oleksiyu/WORK/hyperproject/ &&\
 python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run_group.py\
 --config-name full_run_group_0dopings_1seed.yaml\
 full_run_cfg=TRANSITv0v2_LHCO\
 run_dir=workspaces/PAPER/TRANSITv0v2_LHCO_0doping"

# Record the end time
end_time=$(date +%s)
echo "Job ended at: $(date)"