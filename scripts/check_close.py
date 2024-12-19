import pandas as pd
import hashlib
from pathlib import Path
import numpy as np

def hash_dataframe_from_h5(file_path, compare_csv, tolerance=1e-5):
    # Load the DataFrame from the HDF5 file
    df = pd.read_hdf(file_path)
    df_compare = pd.read_csv(compare_csv)
    # complete match
    do_match_complete = np.allclose(df.head(), df_compare, atol=0)
    # Compare the DataFrame head to the reference CSV with a tolerance
    do_match = np.allclose(df.head(), df_compare, atol=tolerance)
    if do_match_complete:
        print("++++++++++++++++ COMPLETE MATCH +++++++++++++")
    elif do_match:
        print("++++++++++++++++ MATCH ++++++++++++++++")
        print(df.head().to_csv(index=False))
    else:
        print("!!!!!!!!!!!!!!! DIFFERENT !!!!!!!!!!!!!!!")
        print(df.head().to_csv(index=False))
    return


def main(cfg):
    ref_name="generated_template_approx"
    if cfg.get(ref_name, False):
        hash = hash_dataframe_from_h5(Path(cfg.get("run_dir"))/"template/outputs/template_sample.h5", cfg.get(ref_name))

