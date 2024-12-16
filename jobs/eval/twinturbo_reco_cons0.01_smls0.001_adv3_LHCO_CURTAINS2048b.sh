#!/bin/sh
#SBATCH --job-name=transit_reco_cons0.01_smls0.001_adv3_LHCO_CURTAINS2048b
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --partition=shared-gpu,private-dpnc-gpu
#SBATCH --nodes=1
#SBATCH --output=/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/job_output/transit_reco_cons0.01_smls0.001_adv3_LHCO_CURTAINS2048b-%A-%x_%a.out
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

# Record the start time
start_time=$(date +%s)
echo "Job started at: $(date)"

module load GCCcore/12.3.0 Python/3.11.3
cd scratch/sing_images/
singularity exec --nv -B /home/users/,/srv,/tmp hyperproject_container.sif \
 bash -c "cd /home/users/o/oleksiyu/WORK/hyperproject/ &&\
 python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run.py\
 --config-name transit_reco_cons0.01_smls0.001_adv3_LHCO_CURTAINS2048b general.subfolder=adv_present/ do_train_template=0"

# Record the end time
end_time=$(date +%s)
echo "Job ended at: $(date)"