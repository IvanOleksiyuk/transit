import pandas as pd
import hashlib
from pathlib import Path
import numpy as np

def hash_dataframe_from_h5(file_path, approximate=True):
    # Load the DataFrame from the HDF5 file
    df = pd.read_hdf(file_path)
    
    # Serialize the DataFrame to a string for hashing
    if approximate:
        df_string = df.head().to_csv(index=False)  # You can also use to_json, to_string, etc.
        print(df_string)
    else:
        df_string = df.astype(np.float16).to_csv(index=False)
    # Compute the SHA256 hash of the serialized DataFrame
    hash_object = hashlib.sha256(df_string.encode('utf-8'))
    return hash_object.hexdigest()

def hash_png_image(file_path):
    """
    Compute the hash (checksum) of a PNG image file.

    Args:
        file_path (str): The path to the PNG image file.

    Returns:
        str: The hexadecimal hash value of the image.
    """
    # Open the image file in binary mode
    with open(file_path, "rb") as f:
        # Read the entire file content
        image_data = f.read()
    
    # Compute the SHA256 hash of the image data
    hash_object = hashlib.sha256(image_data)
    return hash_object.hexdigest()


def main(cfg):
    hash_name="generated_template_approx"
    if cfg.get(hash_name, False):
        hash = hash_dataframe_from_h5(Path(cfg.get("run_dir"))/"template/outputs/template_sample.h5")
        if hash ==  cfg.get(hash_name, False):
            print(f"{hash_name}: ++++++++++++++++ MATCH ++++++++++++++++")
        else:
            print(f"{hash_name}: !!!!!!!!!!!!!!! DIFFERENT !!!!!!!!!!!!!!!")
            print(f"SHA256 hash of the {hash_name}: {hash}")  

    hash_name="generated_template_full"
    if cfg.get(hash_name, False):
        hash = hash_dataframe_from_h5(Path(cfg.get("run_dir"))/"template/outputs/template_sample.h5", approximate=False)
        if hash ==  cfg.get(hash_name, False):
            print(f"{hash_name}: ++++++++++++++++ MATCH ++++++++++++++++")
        else:
            print(f"{hash_name}: !!!!!!!!!!!!!!! DIFFERENT !!!!!!!!!!!!!!!")
            print(f"SHA256 hash of the {hash_name}: {hash}")

    hash_name="compare_SI_plot"
    if cfg.get(hash_name, False):
        hash = hash_png_image(Path(cfg.get("run_dir"))/"plots/compare/SI_v_rej.png")
        if hash ==  cfg.get(hash_name, False):
            print(f"{hash_name}: ++++++++++++++++ MATCH ++++++++++++++++")
        else:
            print(f"{hash_name}: !!!!!!!!!!!!!!! DIFFERENT !!!!!!!!!!!!!!!")
            print(f"SHA256 hash of the {hash_name}: {hash}")    