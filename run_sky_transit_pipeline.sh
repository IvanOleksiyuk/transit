# 1.Comment out step 2
# 2. Run this
snakemake -p --profile \
workflow/profiles/baobab/ \
--config experiment_name="main" template_method=TRANSIT \
experiment_base_dir='/srv/beegfs/scratch/groups/rodem/skycurtains/results/final' \
experiment_group='skyTransit' code_revision='tranit' host='baobab' devstage='transit' config_dir='/home/users/o/oleksiyu/WORK/skycurtains/config' -s \
workflow/main.smk #--dry-run

# 3. Make step 2 uncommented
# 4. run this 
snakemake -p --profile \
workflow/profiles/baobab/ \
--config experiment_name="main" template_method=TRANSIT \
experiment_base_dir='/srv/beegfs/scratch/groups/rodem/skycurtains/results/final' \
experiment_group='skyTransit' code_revision='tranit' host='baobab' devstage='transit' config_dir='/home/users/o/oleksiyu/WORK/skycurtains/config' -s \
workflow/main.smk #--dry-run