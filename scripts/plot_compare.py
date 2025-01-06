# This script  just build SIC curves and simmilar plots.

import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")
import logging
import hydra
from omegaconf import DictConfig
import os
from pathlib import Path 
import pandas as pd
log = logging.getLogger(__name__)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import glob
from scipy.interpolate import interp1d
from transit.src.utils.file_system import find_files_with_name

@hydra.main(
    version_base=None, config_path=str('../config/step_plot_compare'), config_name="transit_LHCO_seeds.yaml"
) 
def main(cfg) -> None:
    out_dir = Path(cfg.run_dir) / "plots/compare"
    out_dir.mkdir(exist_ok=True, parents=True)
    curve_types = ["ROC", "SI_v_rej", "rejection_v_TPR"]
    
    methods=dict(cfg.methods)
    methods.update(dict(cfg.main_methods))
    for curve_type in curve_types:    
        curves = {}
        for key, method in methods.items():
            directory = method["abs_directory"] if "abs_directory" in method else str(cfg.run_dir) + method["rel_directory"]
            curves[key] = get_curve(directory, curve_type, method["prefix"], method["postfix"])
            if cfg.save_curves:
                np.save(str(out_dir)+"/"+key+"_"+curve_type+"_"+cfg.postfix+".npy", curves[key])
        plot_curves(curves, curve_type, out_dir)
        print("curves loaded for ", curve_type)
    
    curve_types = ["ROC_closure"]
    
    methods=dict(cfg.methods)
    methods.update(dict(cfg.main_methods))
    for curve_type in curve_types:    
        curves = {}
        for key, method in methods.items():
            directory = method["abs_directory"] if "abs_directory" in method else str(cfg.run_dir) + method["rel_directory"]
            curves[key] = get_curve(directory, curve_type, method["prefix"], method["postfix"], ignore_missing=True)
            if cfg.save_curves:
                np.save(str(out_dir)+"/"+key+"_"+curve_type+"_"+cfg.postfix+".npy", curves[key])
        plot_curves(curves, curve_type, out_dir)
        print("curves loaded for ", curve_type)
        
def filter_finite_values(x):
    return x[np.isfinite(x)]

def get_common_x(curves, mode="concat"):
    min_x = max([filter_finite_values(curve[0])[0] for curve in curves])
    max_x = min([filter_finite_values(curve[0])[-1] for curve in curves])
    if mode == "concat":
        common_x = np.sort(np.unique(np.concatenate([filter_finite_values(curve[0]) for curve in curves])))
        common_x = common_x[(common_x >= min_x) & (common_x <= max_x)]
    else:
        common_x = np.linspace(min_x, max_x, 1000)
    return common_x

def get_curve(method, curve_type, prefix="", postfix="", ignore_missing=False):
    files = find_files_with_name(method, prefix+curve_type+postfix+".npy")
    if len(files) == 0:
        if ignore_missing:
            return [np.array([]), np.array([]), np.array([])]
        print("No files found for "+prefix+curve_type+postfix+".npy")
        assert False, "No files found for "+prefix+curve_type+postfix+".npy"+" in "+method
    curves = []
    for file in files:
        curves.append(np.load(file))
    # Aggregate the curves that might have different x values (use interpolation)
    if len(curves) == 1:
        return [curves[0][0], curves[0][1], 0]
    # Sort curves so that x is always rising
    for curve in curves:
        if curve[0][0] > curve[0][-1]:
            curve[0] = curve[0][::-1]
            curve[1] = curve[1][::-1]
        else:
            curve[0] = curve[0]
            curve[1] = curve[1]
    
    # Find the largest x range that all curves have
    min_x = max([filter_finite_values(curve[0])[0] for curve in curves])
    max_x = min([filter_finite_values(curve[0])[-1] for curve in curves])
    
    # get a common x range
    common_x = np.sort(np.unique(np.concatenate([filter_finite_values(curve[0]) for curve in curves])))
    common_x = common_x[(common_x >= min_x) & (common_x <= max_x)]
    
    # Interpolate the y values for each curve
    interpolated_y = []
    for xy in curves:
        x = xy[0]
        y = xy[1]
        interp_func = interp1d(x, y, kind='linear', bounds_error=False, fill_value=(y[0], y[-1]))
        interpolated_y.append(interp_func(common_x))
    
    # Convert the list of arrays into a 2D array for easier computation
    interpolated_y = np.array(interpolated_y)

    # Calculate the mean and standard deviation at each x-value
    mean_y = np.mean(interpolated_y, axis=0)
    std_y = np.std(interpolated_y, axis=0)
            
    return [common_x, mean_y, std_y]

def plot_curves(curves, curve_type, out_dir):
    if curve_type == "ROC_closure":
        plt.figure()
        for key, curve in curves.items():
            x, y, std = curve
            plt.fill_between(x, y-x-std, y-x+std, alpha=0.5)
            plt.plot(x, y-x, label=key+f"\nAUC: {np.trapz(y, x):.4f}+/-{np.trapz(std, x):.4f}")
        plt.title("ROC Curve Closure Test")
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR) - FPR')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.plot([0, 1], [0, 0], 'k--')
        plt.grid(which='major', alpha=0.5)
        plt.savefig(str(out_dir)+"/ROC_closure.png", bbox_inches='tight', dpi=300)
        plt.gca().set_aspect('equal')
        print("ROC curve saved to ", str(out_dir)+"/ROC.png")
    if curve_type == "ROC":
        plt.figure()
        for key, curve in curves.items():
            x, y, std = curve
            plt.fill_between(x, y-std, y+std, alpha=0.5)
            plt.plot(x, y, label=key)
        plt.title("ROC Curve")
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.legend()
        plt.grid(which='major', alpha=0.5)
        plt.savefig(str(out_dir)+"/ROC.png", bbox_inches='tight', dpi=300)
        plt.gca().set_aspect('equal')
        print("ROC curve saved to ", str(out_dir)+"/ROC.png")
    if curve_type == "SI_v_rej":
        plt.figure()
        for key, curve in curves.items():
            x, y, std = curve
            plt.fill_between(x, y-std, y+std, alpha=0.5)
            plt.plot(x, y, label=key)
        plt.title("SI vs Rejection")
        plt.xlabel('Rejection')
        plt.ylabel('SI')
        plt.legend()
        plt.xscale('log')
        plt.ylim(0, 20.0)
        plt.grid(which='both', alpha=0.5)
        plt.savefig(str(out_dir)+"/SI_v_rej.png", bbox_inches='tight', dpi=300)
        print("SI vs Rejection curve saved to ", str(out_dir)+"/SI_v_rej.png")
    if curve_type == "rejection_v_TPR":
        plt.figure()
        for key, curve in curves.items():
            x, y, std = curve
            plt.fill_between(x, y-std, y+std, alpha=0.5)
            plt.plot(x, y, label=key)
        plt.title("Rejection vs TPR")
        plt.ylabel('Rejection')
        plt.xlabel('TPR')
        plt.legend()
        plt.yscale('log')
        plt.grid(which='both', alpha=0.5)
        plt.savefig(str(out_dir)+"/rejection_v_TPR.png", bbox_inches='tight', dpi=300)
        print("Rejection vs TPR curve saved to ", str(out_dir)+"/rejection_v_TPR.png")

if __name__ == "__main__":
    main()
