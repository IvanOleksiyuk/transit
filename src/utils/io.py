import pathlib

from omegaconf import OmegaConf


def init_job(cfg):
    # Given a config job run standard set up and return the top directory for an experiment

    if not (
        hasattr(cfg, "output")
        & hasattr(cfg.output, "save_dir")
        & hasattr(cfg.output, "name")
    ):
        raise Exception("Config file must set output.save_dir and output.name")

    print("Configuring job with following options")
    print(OmegaConf.to_yaml(cfg))
    experiment_top_path = ExperimentPath(cfg.output.save_dir + "/" + cfg.output.name)
    with open(experiment_top_path / f"config.yaml", "w") as file:
        OmegaConf.save(config=cfg, f=file)
    return experiment_top_path


class ExperimentPath(pathlib.Path):
    _flavour = type(pathlib.Path())._flavour

    def __init__(self, top_dir):
        super(ExperimentPath, self).__init__()
        self.top_dir = pathlib.Path(top_dir)
        self.top_dir.mkdir(parents=True, exist_ok=True)

    def sub_dir(self, sub_dir):
        """
        Get a path to a subdirectory of the experiment and ensure it exists
        :param sub_dir: String that specifies path to the subdirectory
        :return: pathlib object pointing to sub directory specified by sub_dir
        """
        directory = self / sub_dir
        directory.mkdir(parents=True, exist_ok=True)
        return directory
