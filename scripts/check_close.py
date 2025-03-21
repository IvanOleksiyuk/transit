import pandas as pd
import hashlib
from pathlib import Path
import numpy as np
import os

def check_close_dataframe_from_h5(file_path, compare_csv, tolerance=1e-5):
    # Load the DataFrame from the HDF5 file
    df = pd.read_hdf(file_path)
    if not os.path.exists(compare_csv):
        # Print the expected values from the HDF5 file head
        print(f"Error: The reference CSV file '{compare_csv}' was not found.")
        print("Expected values from the HDF5 file:")
        print(df.head().to_csv(index=False))
        return
    else:
        # Load the reference CSV file
        df_compare = pd.read_csv(compare_csv)
    # complete match
    do_match_complete = np.allclose(df.head(), df_compare, atol=0)
    # Compare the DataFrame head to the reference CSV with a tolerance
    do_match = np.allclose(df.head(), df_compare, atol=tolerance)
    if do_match_complete:
        print("++++++++++++++++ COMPLETE MATCH +++++++++++++")
    elif do_match:
        print("++++++++++++++++ MATCH ++++++++++++++++")
        print("expected:")
        print(df_compare.to_csv(index=False))
        print("actual:")
        print(df.head().to_csv(index=False))
    else:
        print("!!!!!!!!!!!!!!! DIFFERENT !!!!!!!!!!!!!!!")
        print("expected:")
        print(df_compare.to_csv(index=False))
        print("actual:")
        print(df.head().to_csv(index=False))
    return


def main(cfg):
    ref_name="generated_template_approx"
    if cfg.get(ref_name, False):
        hash = check_close_dataframe_from_h5(Path(cfg.get("run_dir"))/"template/outputs/template_sample.h5", cfg.get(ref_name))

