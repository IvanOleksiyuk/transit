#!/bin/sh
#SBATCH --job-name=v25f_SKY
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=0:40:00
#SBATCH --partition=shared-gpu,private-dpnc-gpu
#SBATCH --nodes=1
#SBATCH --output=/home/users/o/oleksiyu/WORK/skycurtains/transit/jobs/job_output/TRANSITv25f_SKY-%A-%x_%a.out
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

# Record the start time
start_time=$(date +%s)
echo "Job started at: $(date)"

module load GCCcore/12.3.0 Python/3.11.3
singularity exec --nv -B /home/users/,/srv,/tmp /srv/beegfs/scratch/groups/rodem/skycurtains/containers/skyCurtains.sif \
 bash -c "cd /home/users/o/oleksiyu/WORK/skycurtains/ &&\
 python /home/users/o/oleksiyu/WORK/skycurtains/transit/scripts/full_run.py\
 --config-name TRANSITv25f_std_SKY general.subfolder=TEST_SKY_0603/"

# Record the end time
end_time=$(date +%s)
echo "Job ended at: $(date)"