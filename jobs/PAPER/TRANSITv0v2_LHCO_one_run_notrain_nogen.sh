#!/bin/sh
#SBATCH --job-name=v0v2_LHCO
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --partition=private-dpnc-cpu,shared-cpu
#SBATCH --nodes=1
#SBATCH --output=/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/job_output/TRANSITv0v2_LHCO-%A-%x_%a.out
#SBATCH --mem=16GB

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
 step_evaluate.do_2d_hist_instead_of_contour=0\
 verbose_validation=0 \
do_train_template=0 \
do_export_template=0 \
do_export_latent=0 \
do_transport_sideband=0 \
do_evaluation=1 \
do_cwola=1 \
do_evaluate_cwola=1 \
do_plot_compare=1 \
do_update_runtime_file=0"

# Record the end time
end_time=$(date +%s)
echo "Job ended at: $(date)"