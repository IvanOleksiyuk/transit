import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")

import transit.scripts.plot_compare as plot_compare
import yaml
from dotmap import DotMap
import os

# Define the YAML structure as a string
yaml_str = """
stability_analysis_cfg:
  run_dir: "/home/users/o/oleksiyu/WORK/hyperproject/plot/user/compare_v1_v2"
  save_curves: True
  postfix: "comb_seeds_CWOLA"

  main_methods:
    radot: 
      abs_directory: "/home/users/o/oleksiyu/WORK/hyperproject/lit/radot/dope_3000/standard"
      prefix: ""
      postfix: ""
    TRANSITv1: 
      abs_directory: "/home/users/o/oleksiyu/WORK/hyperproject/workspaces/groups5/TRANSITv1/"
      prefix: "transit_"
      postfix: "_comb_CWOLA"
    TRANSITv2: 
      abs_directory : "/home/users/o/oleksiyu/WORK/hyperproject/workspaces/groups5/TRANSITv2/"
      prefix: "transit_"
      postfix: "_comb_CWOLA"

  methods:
    supervised: 
      abs_directory: "/home/users/o/oleksiyu/WORK/hyperproject/lit/radot/dope_3000/supervised"
      prefix: ""
      postfix: ""
    idealised: 
      abs_directory: "/home/users/o/oleksiyu/WORK/hyperproject/lit/radot/dope_3000/idealised"
      prefix: ""
      postfix: ""
"""
"""

"""
# Load the YAML string
data = yaml.safe_load(yaml_str)

# Convert the loaded data into a DotMap
dotmap_data = DotMap(data)

# Print the resulting DotMap object
print(dotmap_data)
os.makedirs(dotmap_data.stability_analysis_cfg.run_dir, exist_ok=True)

plot_compare.main(dotmap_data.stability_analysis_cfg)