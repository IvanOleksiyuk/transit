g_loss_gen_weight: [0.05, 0.2, 1.0, 0]
w_cons: [0.02, 0.1, 0.5, 0]
#proposals:
#z_dim: [4, 8, 16]
#n_blocks: [1, 2, 3]
#width: [64, 128, 256]
#epoch_scale: [0.8, 1, 1.2]
#initial_lr: [0.0002, 0.002, 0.02]
#weight_decay: [0.000001, 0.00001, 0.0001]
#disc_steps_per_gen: [1, 2, 3]


# Hyperparameter optimisation

HYDRA_FULL_ERROR=1 python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run_group.py \
--config-name full_run_group_0dopings_10seeds_hyper.yaml \
full_run_cfg=TRANSITv0v2_LHCO_optimisable \
run_dir=workspaces/HYPER/TRANSITv0v2_LHCO_group 

## Ablation: g_loss_gen_weight

HYDRA_FULL_ERROR=1 python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run_group.py \
--config-name full_run_group_0dopings_5seeds_hyper.yaml \
full_run_cfg=TRANSITv0v2_LHCO_optimisable \
run_dir=workspaces/HYPER/TRANSITv0v2_LHCO_group_wt1 \
hyperparameters.g_loss_gen_weight=1

HYDRA_FULL_ERROR=1 python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run_group.py \
--config-name full_run_group_0dopings_5seeds_hyper.yaml \
full_run_cfg=TRANSITv0v2_LHCO_optimisable \
run_dir=workspaces/HYPER/TRANSITv0v2_LHCO_group_wt2 \
hyperparameters.g_loss_gen_weight=0.05

HYDRA_FULL_ERROR=1 python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run_group.py \
--config-name full_run_group_0dopings_5seeds_hyper.yaml \
full_run_cfg=TRANSITv0v2_LHCO_optimisable \
run_dir=workspaces/HYPER/TRANSITv0v2_LHCO_group_wt3 \
hyperparameters.g_loss_gen_weight=0.0

## Ablation: w_cons

HYDRA_FULL_ERROR=1 python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run_group.py \
--config-name full_run_group_0dopings_5seeds_hyper.yaml \
full_run_cfg=TRANSITv0v2_LHCO_optimisable \
run_dir=workspaces/HYPER/TRANSITv0v2_LHCO_group_wc1 \
hyperparameters.w_cons=0.5

HYDRA_FULL_ERROR=1 python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run_group.py \
--config-name full_run_group_0dopings_5seeds_hyper.yaml \
full_run_cfg=TRANSITv0v2_LHCO_optimisable \
run_dir=workspaces/HYPER/TRANSITv0v2_LHCO_group_wc2 \
hyperparameters.w_cons=0.02

HYDRA_FULL_ERROR=1 python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run_group.py \
--config-name full_run_group_0dopings_5seeds_hyper.yaml \
full_run_cfg=TRANSITv0v2_LHCO_optimisable \
run_dir=workspaces/HYPER/TRANSITv0v2_LHCO_group_wc3 \
hyperparameters.w_cons=0.0

## Ablation: latent_dim

HYDRA_FULL_ERROR=1 python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run_group.py \
--config-name full_run_group_0dopings_5seeds_hyper.yaml \
full_run_cfg=TRANSITv0v2_LHCO_optimisable \
run_dir=workspaces/HYPER/TRANSITv0v2_LHCO_group_ld1 \
hyperparameters.latent_dim=6

HYDRA_FULL_ERROR=1 python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run_group.py \
--config-name full_run_group_0dopings_5seeds_hyper.yaml \
full_run_cfg=TRANSITv0v2_LHCO_optimisable \
run_dir=workspaces/HYPER/TRANSITv0v2_LHCO_group_ld2 \
hyperparameters.latent_dim=10

## Ablation: num_blocks

HYDRA_FULL_ERROR=1 python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run_group.py \
--config-name full_run_group_0dopings_5seeds_hyper.yaml \
full_run_cfg=TRANSITv0v2_LHCO_optimisable \
run_dir=workspaces/HYPER/TRANSITv0v2_LHCO_group_nb1 \
hyperparameters.num_blocks=2

HYDRA_FULL_ERROR=1 python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run_group.py \
--config-name full_run_group_0dopings_5seeds_hyper.yaml \
full_run_cfg=TRANSITv0v2_LHCO_optimisable \
run_dir=workspaces/HYPER/TRANSITv0v2_LHCO_group_nb2 \
hyperparameters.num_blocks=4

## Ablation: hddn_dim

HYDRA_FULL_ERROR=1 python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run_group.py \
--config-name full_run_group_0dopings_5seeds_hyper.yaml \
full_run_cfg=TRANSITv0v2_LHCO_optimisable \
run_dir=workspaces/HYPER/TRANSITv0v2_LHCO_group_hd1 \
hyperparameters.hddn_dim=64

# HYDRA_FULL_ERROR=1 python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run_group.py \
# --config-name full_run_group_0dopings_5seeds_hyper.yaml \
# full_run_cfg=TRANSITv0v2_LHCO_optimisable \
# run_dir=workspaces/HYPER/TRANSITv0v2_LHCO_group_hd2 \
# hyperparameters.hddn_dim=256 \
# one_run_sh=/home/users/o/oleksiyu/WORK/hyperproject/transit/jobs/PAPER/TRANSITv0v2_LHCO_one_run_hyper40m.sh

## Ablation: disc_hidd

HYDRA_FULL_ERROR=1 python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run_group.py \
--config-name full_run_group_0dopings_5seeds_hyper.yaml \
full_run_cfg=TRANSITv0v2_LHCO_optimisable \
run_dir=workspaces/HYPER/TRANSITv0v2_LHCO_group_dh1 \
hyperparameters.disc_hidd=32

HYDRA_FULL_ERROR=1 python /home/users/o/oleksiyu/WORK/hyperproject/transit/scripts/full_run_group.py \
--config-name full_run_group_0dopings_5seeds_hyper.yaml \
full_run_cfg=TRANSITv0v2_LHCO_optimisable \
run_dir=workspaces/HYPER/TRANSITv0v2_LHCO_group_dh2 \
hyperparameters.disc_hidd=128
