#Go to home and run:
sbatch /home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/PAPER/TRANSITv0v1_LHCO_one_run.sh
#WAIT FOR THE JOB TO FINISH
sbatch /home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/PAPER/TRANSITv0v2_LHCO_one_run_eval.sh

# Go in the singularity image and run:
python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run_group.py\
 --config-name full_run_group_dopings_6seeds.yaml\
 full_run_cfg=TRANSITv0v2_LHCO\
 run_dir=workspaces/PAPER/TRANSITv0v2_LHCO_dopings

HYDRA_FULL_ERROR=1 python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run_group.py\
 --config-name full_run_group_stability_30.yaml\
 full_run_cfg=TRANSITv0v2_LHCO\
 run_dir=workspaces/PAPER/TRANSITv0v2_LHCO_group

# Go to home and run:
python submit.py /home/users/o/oleksiyu/WORK/hyperproject/workspaces/PAPER/TRANSITv0v2_LHCO_dopings
python submit.py /home/users/o/oleksiyu/WORK/hyperproject/workspaces/PAPER/TRANSITv0v2_LHCO_group
#WAIT FOR THE JOBS TO FINISH

# Go in the singularity image and run:
python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run_group.py\
 --config-name full_run_group_dopings_6seeds.yaml\
 full_run_cfg=TRANSITv0v2_LHCO\
 run_dir=workspaces/PAPER/TRANSITv0v2_LHCO_dopings

HYDRA_FULL_ERROR=1 python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run_group.py\
 --config-name full_run_group_stability_30.yaml\
 full_run_cfg=TRANSITv0v2_LHCO\
 run_dir=workspaces/PAPER/TRANSITv0v2_LHCO_group



# run /home/users/o/oleksiyu/WORK/hyperproject/transit/notebooks/final_plotsML4J.ipynb and get your plots