#!/bin/sh
#SBATCH --job-name=CLASS_TRANSIT_LLV_v2_1cst
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=7:00:008
#SBATCH --partition=shared-gpu
#SBATCH --nodes=1
#SBATCH --output=/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/job_output/CLASS_TRANSIT_LLV_v2_1cst-%A-%x_%a.out
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

# Record the start time
start_time=$(date +%s)
echo "Job started at: $(date)"

module load GCCcore/12.3.0 Python/3.11.3
cd sing_images/
singularity exec --nv -B /home/users/,/srv,/tmp hyperproject_container.sif \
 bash -c "cd /home/users/o/oleksiyu/WORK/hyperproject/ &&\
 HYDRA_FULL_ERROR=1 /opt/conda/bin/python\
 /home/users/o/oleksiyu/WORK/hyperproject/libs_snap/anomdiff/scripts/train_disc.py\
 --config-name=train_disc_0cst.yaml\
 mode=standard\
 n_csts=1\
 output_folder=/home/users/o/oleksiyu/WORK/hyperproject/user/classifiers/low_class_outs_TRANSIT_LLV_v2_1cst\
 network_name=class_llv_0cst_$(date +"%Y%m%d%H%M%S")\
 tem_path=/home/users/o/oleksiyu/WORK/hyperproject/user/llv_hlv_templates/TRANSIT_LLV_v2_0cst.h5"

# Record the end time
end_time=$(date +%s)
echo "Job ended at: $(date)"

