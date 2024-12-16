"""A collection of misculaneous functions useful within snakemake workflows."""

import os
from typing import Callable, Optional


def gen_hydra_flags(
    run_dir: str | Callable,
    job_name: str | Callable = None,
    use_orig_configs: Optional[bool] = None,
) -> list[Callable]:
    """Generate specific hydra flags, to manage proper reproducibility.

    Args
    ----
        run_dir (str): path to the run directory
        use_orig_configs (Optional[bool], optional): if true (the default) and there
                                                     already exists a config in the
                                                     hydra run directory, re-use it
                                                     and ignore everything else.

    Note
    ----
        Always use this at the end of your parameter list in snakemake, otherwise
        it is is possible that hydra can't read the command line.
    """
    get_run_dir = run_dir if callable(run_dir) else lambda wc: run_dir
    use_orig_configs = True if use_orig_configs is None else use_orig_configs
    if job_name:
        get_job_name = job_name if callable(job_name) else lambda wc: job_name

    def get_experiment_rerun_flag(wc):
        run_dir = get_run_dir(wc)
        if use_orig_configs and os.path.exists(f"{run_dir}/config.pickle"):
            # this will ignore all other flags
            return f"--experimental-rerun {run_dir}/config.pickle"
        else:
            return ""

    flags = []
    flags.append(lambda wc: f"hydra.run.dir={get_run_dir(wc)}")
    if job_name:
        flags.append(lambda wc: f"hydra.job.name={get_job_name(wc)}")
    # append this at the end, otherwise hydra can't read the command line
    flags.append(lambda wc: get_experiment_rerun_flag(wc))
    return flags
