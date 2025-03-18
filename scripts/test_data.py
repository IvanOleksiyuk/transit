# Standard block for correct import
import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")
import hydra


if __name__ == "__main__":
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(version_base=None, config_path="../config/data")
    cfg = hydra.compose(config_name="SKY_DEFAULT_test.yaml")
    
    datamodule = hydra.utils.instantiate(cfg.template_training)
    datamodule.setup("fit")
    print(datamodule)
    
