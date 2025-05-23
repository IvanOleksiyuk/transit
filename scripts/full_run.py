# This is a script to do one full LHCO analysis. 
# It is a combination of the several steps.

# 1 - Create a separate folder for the experiment save all the relavant configs there 
# 2 - train/create a model that will provide a us with a template (e.g. CATHODE, CURTAINS)
# 3 - generate a template dataset using the model
# 4 - evaluate the performance of the template generation model
# 4 - train cwola
# 5 - evaluate the performance and plot the results
# 6 - produce a set of final plots and tables for one run

import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")

import logging
import hydra
from pathlib import Path
import os
from omegaconf import DictConfig, OmegaConf
from transit.src.utils.hydra_utils import instantiate_collection, log_hyperparameters, print_config, reload_original_config, save_config
import transit.scripts.evaluation as evaluation
import transit.scripts.train_ptl as train_ptl
import transit.scripts.generate_teplate as generate_teplate
import transit.scripts.run_cwola as run_cwola
import transit.scripts.cwola_evaluation as cwola_evaluation
import transit.scripts.plot_compare as plot_compare
import transit.scripts.export_latent_space as export_latent_space
import transit.scripts.time_chart as time_chart
import transit.scripts.check_close as check_close
import transit.scripts.collect_metrics as collect_metrics
from datetime import datetime
import subprocess
log = logging.getLogger(__name__)

def get_git_hash(repo_dir=None):
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            raise Exception(f"Error getting git hash: {result.stderr.strip()}")
    except Exception as e:
        return str(e)

def get_uncommitted_changes(repo_dir=None):
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], cwd=repo_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return result.stdout.strip()  # Returns the list of uncommitted changes
        else:
            raise Exception(f"Error checking uncommitted changes: {result.stderr.strip()}")
    except Exception as e:
        return str(e)

def write_git_status_to_file(file_path, repo_dir, mode="w"):
    try:
        git_hash = get_git_hash(repo_dir=repo_dir)
        uncommitted_changes = get_uncommitted_changes(repo_dir=repo_dir)
        with open(file_path, mode) as file:
            file.write(f"Repo Dir: {repo_dir}\n")
            file.write(f"Git Hash: {git_hash}\n")
            if uncommitted_changes:
                file.write("Uncommitted Changes:\n")
                file.write(f"{uncommitted_changes}\n")
            else:
                file.write("Uncommitted Changes: No\n")

        print(f"Git status written to {file_path}")
    except Exception as e:
        print(f"Error writing to file: {str(e)}")

def update_runtime_file_func(file_path, text):
    with open(file_path, 'a') as file:
        file.write(f'{text}\n')

@hydra.main(
    version_base=None, config_path=str('../config'), config_name="TRANSITpad_v25fstd"
)
def main(cfg: DictConfig) -> None:
    log.info("<<<START FULL RUN>>>")
    ## 1 - Create a separate folder for the experiment save all the relavant configs there
    # Create a folder for the experiment
    run_dir = Path(cfg.general.run_dir)
    
    # Delete the folder if it already exists if the flag is set
    if cfg.get("delete_existing_run_dir", False):
        if run_dir.exists():
            import shutil
            shutil.rmtree(run_dir)
    
    # create a summary folder and a file to save the runtime of each step
    
    summary_dir = run_dir / "summary"
    os.makedirs(summary_dir, exist_ok=True)
    rutime_file = summary_dir / "runtime.txt"    

    do_update_runtime_file = cfg.get("do_update_runtime_file", False)
    if do_update_runtime_file:
        with open(rutime_file, 'w') as file:
            file.write('Start time: {}\n'.format(datetime.now()))
        update_runtime_file = lambda text: update_runtime_file_func(rutime_file, text)
    else:
        update_runtime_file = lambda text: None
    
    # Save git hash to a file
    git_hash_file = summary_dir / "git_hash.txt"
    write_git_status_to_file(git_hash_file, repo_dir=root)
    write_git_status_to_file(git_hash_file, repo_dir=root / "transit", mode="a")
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(cfg.step_train_template.paths.full_path, exist_ok=True)
    #print_config(cfg)
    OmegaConf.save(cfg, Path(cfg.general.run_dir, "full_config.yaml"), resolve=True)

    # run all the steps
    if cfg.get("do_train_template", False):
        start_time = datetime.now()
        log.info("===================================")
        log.info(f"Start: Train a model that will provide us with a template")
        train_ptl.main(cfg.step_train_template)
        log.info(f"Finish: Train a model that will provide us with a template. Time taken: {datetime.now() - start_time}")
        log.info(f"===================================")
        update_runtime_file('Train template: {}\n'.format(datetime.now() - start_time))
    
    if cfg.get("do_export_template", False):
        start_time = datetime.now()
        log.info("===================================")
        log.info("Start: Generate a template dataset using the model")
        name = cfg.step_export_template.output_name
        generate_teplate.main(cfg.step_export_template)
        log.info(f"Finish: Generate a template dataset using the model. Time taken: {datetime.now() - start_time}")
        log.info("===================================")
        update_runtime_file('Generate template: {}\n'.format(datetime.now() - start_time))
    
    if hasattr(cfg, "do_transport_sideband") and cfg.do_transport_sideband:
        start_time = datetime.now()
        log.info("===================================")
        log.info("Start: Generate a template dataset using the model")
        name = cfg.step_export_template.output_name
        generate_teplate.main(cfg.step_export_SB1)
        generate_teplate.main(cfg.step_export_SB2)
        if hasattr(cfg, "step_export_SB1toSR"):
            generate_teplate.main(cfg.step_export_SB1toSR)
        if hasattr(cfg, "step_export_SB2toSR"):
            generate_teplate.main(cfg.step_export_SB2toSR)
        log.info(f"Finish: Generate a template dataset using the model. Time taken: {datetime.now() - start_time}")
        log.info("===================================")
        update_runtime_file('Generate sideband: {}\n'.format(datetime.now() - start_time))
    
    if hasattr(cfg, "do_export_latent") and cfg.do_export_latent:
        start_time = datetime.now()
        log.info("===================================")
        log.info("Start:Generate latent representation of events in SR and Sidebands")
        name = cfg.step_export_latent.export_latent_all.output_name
        export_latent_space.main(cfg.step_export_latent.export_latent_all)
        log.info(f"Finish: Generate latent representation of events in SR and Sidebands Time taken: {datetime.now() - start_time}")
        log.info("===================================")
        update_runtime_file('Generate latent: {}\n'.format(datetime.now() - start_time))
        
    if cfg.get("do_evaluation", False):
        start_time = datetime.now()
        log.info("===================================")
        log.info("Start: Evaluate the performance and plot the results")
        evaluation.main(cfg)
        log.info(f"Finish: Evaluate the performance and plot the results. Time taken: {datetime.now() - start_time}")
        log.info("===================================")
        update_runtime_file('Evaluate: {}\n'.format(datetime.now() - start_time))
        
    if cfg.get("do_cwola", False):
        start_time = datetime.now()
        log.info("===================================")
        log.info("Start: Train CWOLA model using the template dataset and the real data")
        if hasattr(cfg.step_cwola, "several_confs"):
            for conf in cfg.step_cwola.several_confs.values():
                run_cwola.main(conf)
        else:
            run_cwola.main(cfg.step_cwola)
        log.info(f"Finish: Train CWOLA model using the template dataset and the real data. Time taken: {datetime.now() - start_time}")
        log.info("===================================")
        update_runtime_file('Train CWOLA: {}\n'.format(datetime.now() - start_time))

    if cfg.get("do_evaluate_cwola", False):
        start_time = datetime.now()
        log.info("===================================")
        log.info("Start: Evaluate the performance of the CWOLA model")
        if hasattr(cfg.step_cwola, "several_confs"):
            for conf in cfg.step_cwola.several_confs.values():
                if hasattr(conf, "do_evaluate"):
                    if conf.do_evaluate:
                        cwola_evaluation.main(conf)
                else:
                    cwola_evaluation.main(conf)
        else:
            cwola_evaluation.main(cfg.step_cwola)
        log.info(f"Finish: Evaluate the performance of the CWOLA model. Time taken: {datetime.now() - start_time}")
        log.info("===================================")
        update_runtime_file('Evaluate CWOLA: {}\n'.format(datetime.now() - start_time))

    if cfg.get("do_plot_compare", False):
        start_time = datetime.now()
        log.info("===================================")
        log.info("Produce a set of final plots and tables for one run")
        plot_compare.main(cfg.step_plot_compare)
        log.info(f"Finish: Produce a set of final plots and tables for one run. Time taken: {datetime.now() - start_time}")
        log.info("===================================")
        update_runtime_file('Plot compare: {}\n'.format(datetime.now() - start_time))
        
    if hasattr(cfg, "do_cleanup") and cfg.do_cleanup:
        start_time = datetime.now()
        log.info("===================================")
        log.info("Clean up the workspace")
        
    
    if hasattr(cfg, "do_summary_plots") and cfg.do_summary_plots:
        start_time = datetime.now()
        log.info("===================================")
        log.info("Produce a set of summary plots")
        time_chart.main(rutime_file, save_path=summary_dir / "time_plot.png")
        log.info(f"Finish: Produce a set of summary plots. Time taken: {datetime.now() - start_time}")
        log.info("===================================")
        update_runtime_file('Summary plots: {}\n'.format(datetime.now() - start_time))

    if hasattr(cfg, "do_collect_metrics") and cfg.do_collect_metrics:
        start_time = datetime.now()
        log.info("===================================")
        log.info("Collect all the most important metrics in one place")
        collect_metrics.main(cfg.step_collect_metrics)
        log.info(f"Finish: Collect all the most important metrics in one place: {datetime.now() - start_time}")
        log.info("===================================")
        update_runtime_file('Summary plots: {}\n'.format(datetime.now() - start_time))

    log.info("<<<END FULL RUN>>>")

    if hasattr(cfg, "check_close"):
        start_time = datetime.now()
        #log.info("===================================")
        #log.info("Check the hash of the output files")
        check_close.main(cfg.check_close)
        #log.info(f"Finish: Check the hash of the output files. Time taken: {datetime.now() - start_time}")
        #log.info("===================================")
        update_runtime_file('Check hash: {}\n'.format(datetime.now() - start_time))

    done_file_path = run_dir/"ALL.DONE"
    with open(done_file_path, "w") as f:
        f.write("All done for this run!")
        f.close()


if __name__ == "__main__":
    main()
    

