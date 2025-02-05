import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from time import process_time_ns
import yaml

from srccwola.classifier import run_bdt_folds
from srccwola.plotting import plot_closure, plot_svb
from srccwola.utils import get_cwola_datasets


def get_args():
    parser = argparse.ArgumentParser(
        description="Perform anomaly detection using CWOLA"
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        default="/srv/beegfs/scratch/users/l/leighm/linearresults/",
        help="The output folder",
    )
    parser.add_argument(
        "--mode",
        type=lambda x: x.casefold(),
        default="standard",
        help="The mode to run in. Options include: standard and idealised",
    )
    parser.add_argument(
        "--num_signal",
        type=int,
        default=3000,
        help="The number of signal events to use",
    )
    parser.add_argument(
        "--sideband_1",
        type=lambda x: [int(x) for x in x.split("_")],
        default="3100_3300",
        help="Comma seperated sideband 1 range",
    )
    parser.add_argument(
        "--sideband_2",
        type=lambda x: [int(x) for x in x.split("_")],
        default="3700_3900",
        help="Comma seperated sideband 2 range",
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=5,
        help="The number of folds for the BDT",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=250,
        help="The maximum number of iterations for the BDT",
    )
    parser.add_argument(
        "--early_stopping",
        type=lambda x: bool(int(x)),
        default=True,
        help="Whether to use early stopping",
    )
    parser.add_argument(
        "--extra_bkg",
        type=lambda x: bool(int(x)),
        default=True,
        help="Whether to use early stopping",
    )
    parser.add_argument(
        "--extra_signal",
        type=lambda x: bool(int(x)),
        default=True,
        help="Whether to use early stopping",
    )
    parser.add_argument(
        "--validation_fraction",
        type=float,
        default=0.1,
        help="The fraction of data to use for validation",
    )
    parser.add_argument(
        "--class_weight",
        type=str,
        default="balanced",
        help="The class weight for the BDT",
    )
    parser.add_argument(
        "--num_ensemble",
        type=int,
        default=5,
        help="The number of classifiers to train",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The seed for the random number generator",
    )
    parser.add_argument(
        "--title_suffix",
        type=str,
        default="",
        help="Extra suffix to plot title",
    )
    return parser.parse_args()


def setup_run() -> Path:
    """Print the arguments of the run and return the output path."""
    # Load the arguments
    args = get_args()

    # Get the input path
    dir0 = args.input_path
    dir1 = (
        f"window_{args.sideband_1[0]}_{args.sideband_1[1]}"
        f"__{args.sideband_2[0]}_{args.sideband_2[1]}"
    )
    dir2 = f"dope_{args.num_signal}"
    args.input_path = dir0 / dir1 / dir2
    assert args.input_path.exists(), f"Path {args.input_path} does not exist!"

    # Make the output path
    dir3 = f"{args.mode}"
    dir4 = f"seed_{args.seed}"
    args.output_path = args.input_path / dir3 / dir4
    args.output_path.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print("[--] Starting CWoLa")
    print("-" * 80)
    print(f"[--] Input path: {args.input_path}")
    print(f"[--] Output path: {args.output_path}")
    print(f"[--] Mode: {args.mode}")
    print(f"[--] Signal Events: {args.num_signal}")
    print(f"[--] Seed: {args.seed}")
    print("-" * 80)

    return args


def main():
    print("[--] Setting up run")
    args = setup_run()

    print("[--] Loading data")
    data = np.load(args.input_path / "sr.npy")
    template = np.load(args.input_path / "template.npy")
    if args.extra_signal:
        extra_signal = np.load(args.input_path / "extra_signal.npy")
    else:
        extra_signal = None
    if args.extra_bkg:
        extra_bkg = np.load(args.input_path / "extra_bkg.npy")
    else:
        extra_bkg = None
    features = np.loadtxt(args.input_path / "features.txt", dtype=str)

    print(f"[--] Setting datasets for CWoLa using mode {args.mode}")
    signal, background = get_cwola_datasets(data, template, args.mode)

    print(f"[--] Template Shape: {template.shape}")
    print(f"[--] Data Shape: {data.shape}")
    print(f"[--] Signal Shape: {signal.shape}")
    print(f"[--] Background Shape: {background.shape}")

    print(f"[--] Training {args.num_folds} seperate folds")
    time = process_time_ns()
    inputs, labels, outputs, extra_preds_sig, extra_preds_bkg, extra_preds_dict, extra_bkg_preds_dict = run_bdt_folds(
        np.vstack((signal, background)),
        np.concatenate((np.ones(len(signal)), np.zeros(len(background)))),
        num_folds=args.num_folds,
        num_ensemble=args.num_ensemble,
        bdt_args={
            "max_iter": args.max_iter,
            "early_stopping": args.early_stopping,
            "validation_fraction": args.validation_fraction,
            "class_weight": args.class_weight,
        },
        seed=args.seed,
        extra_sig=extra_signal,
        extra_bkg=extra_bkg,
    )
    cwola_time = process_time_ns() - time
    print(f"[--] CWoLa completed in {cwola_time/1e9:.2f} seconds")
    print("[--] Combining and saving outputs")
    pd.DataFrame(
        np.hstack((inputs, labels[:, None], outputs[:, None])),
        columns=[*features, "CWoLa", "preds"],
    ).to_hdf(
        args.output_path / "cwola_outputs.h5",
        key="df",
        mode="w",
    )
    
    # Evaluate predictios for extra signal
    if args.extra_signal:
        extra_sig_df = pd.DataFrame(
            np.hstack((extra_signal, extra_preds_sig[:, None])),
            columns=[*features, "preds"],
        )
        for key, preds in extra_preds_dict.items():
            extra_sig_df[key] = preds
        extra_sig_df.to_hdf(
            args.output_path / "cwola_outputs_extra_sig.h5",
            key="df",
            mode="w",
        )
    
    # Evaluate predictions for extra background
    if args.extra_bkg:
        extra_bkg_df = pd.DataFrame(
            np.hstack((extra_bkg, extra_preds_bkg[:, None])),
            columns=[*features, "preds"],
        )
        for key, preds in extra_bkg_preds_dict.items():
            extra_bkg_df[key] = preds
        extra_bkg_df.to_hdf(
            args.output_path / "cwola_outputs_extra_bkg.h5",
            key="df",
            mode="w",
        )
    with open(args.input_path/"cwola_time.txt", "w") as f:
        f.write(str(cwola_time))
        
    print("[--] Plotting")
    if args.mode == "standard":
        plot_closure(inputs, labels, outputs, args.output_path, title_suffix=args.title_suffix)

    if args.num_signal > 0:
        plot_svb(inputs, outputs, extra_preds_sig, args.output_path, title_suffix=args.title_suffix)


if __name__ == "__main__":
    main()
