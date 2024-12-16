# Standard block for correct import
import unittest

import hydra
import numpy as np


if __name__ == "__main__":
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(version_base=None, config_path="../config")
    cfg = hydra.compose(config_name="test_ProcessorApplyCuts.yaml")
    
    dataset = hydra.utils.instantiate(cfg.data)
    
