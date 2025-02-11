# How to compose a config and a job for run/runs

python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run_group.py \
--config-name full_run_group_0dopings_1seed.yaml \
full_run_cfg=TRANSITv0v2_LHCO \
run_dir=workspaces/PAPER/TRANSITv0v2_LHCO_0doping \
redo=1 \

# How to do a run

HYDRA_FULL_ERROR=1 python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run.py \
--config-path /home/users/o/oleksiyu/WORK/hyperproject/workspaces/PAPER/TRANSITv0v2_LHCO_0doping/run-doping_0/run-TTS_1 --config-name full_config.yaml general.subfolder=PAPER/ \
verbose_validation=0 \
verbose_validation=0 \
do_train_template=0 \
do_export_template=0 \
do_export_latent=0 \
do_transport_sideband=0 \
do_evaluation=0 \
do_cwola=0 \
do_evaluate_cwola=0 \
do_plot_compare=0 \
do_collect_metrics=1 \

# In case you  want to evaluate again without trainng everything

HYDRA_FULL_ERROR=1 python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run_group.py \
--config-name full_run_group_stability_30_notrain.yaml \
full_run_cfg=TRANSITv0v2_LHCO \
run_dir=workspaces/PAPER/TRANSITv0v2_LHCO_group \
redo=1 \

python submit.py /home/users/o/oleksiyu/WORK/hyperproject/workspaces/PAPER/TRANSITv0v2_LHCO_group

HYDRA_FULL_ERROR=1 python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run_group.py \
--config-name full_run_group_dopings_6seeds_notrain.yaml \
full_run_cfg=TRANSITv0v2_LHCO \
run_dir=workspaces/PAPER/TRANSITv0v2_LHCO_dopings \
redo=1 \

python submit.py /home/users/o/oleksiyu/WORK/hyperproject/workspaces/PAPER/TRANSITv0v2_LHCO_dopings

HYDRA_FULL_ERROR=1 python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run_group.py \
--config-name full_run_group_stability_30_notrain.yaml \
full_run_cfg=TRANSITv0v2_LHCO \
run_dir=workspaces/PAPER/TRANSITv0v2_LHCO_group \
redo=1 \

HYDRA_FULL_ERROR=1 python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run_group.py \
--config-name full_run_group_dopings_6seeds_notrain.yaml \
full_run_cfg=TRANSITv0v2_LHCO \
run_dir=workspaces/PAPER/TRANSITv0v2_LHCO_dopings \
redo=1 \


