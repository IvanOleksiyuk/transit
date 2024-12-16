import numpy as np
import ot
from sklearn.base import BaseEstimator
from tqdm import trange

from mltools.mltools.numpy_utils import chunk_given_size


def build_template(
    sb1: np.ndarray,
    sb2: np.ndarray,
    target_vars: np.ndarray,
    batch_size: int,
    preprocessor: BaseEstimator,
    seed: int = 0,
    interp_preproc: bool = False,
) -> np.ndarray:
    """Build the template batch by batch using OT matching."""
    # Start by shuffling the data
    rng = np.random.default_rng(seed)
    rng.shuffle(sb1)
    rng.shuffle(sb2)

    # Generate the collections of sideband data to iterate through.
    # We could randomly bootstrap each batch but we run into the problem where
    # some sideband data won't be used at all. Instead lets use resize, which
    # tiles the data until its the correct size, then shuffle and chunk.
    sb1_exp = np.resize(sb1, (len(target_vars), sb1.shape[1]))
    sb2_exp = np.resize(sb2, (len(target_vars), sb2.shape[1]))
    rng.shuffle(sb1_exp)
    rng.shuffle(sb2_exp)

    # Chunk the target variables and sidebands into batches
    target_vars_exp = chunk_given_size(target_vars, batch_size)
    sb1_exp = chunk_given_size(sb1_exp, batch_size)
    sb2_exp = chunk_given_size(sb2_exp, batch_size)

    # We drop the last batch to ensure all batches are the same size
    drop_last = len(target_vars_exp[-1]) < batch_size
    if drop_last:
        target_vars_exp = target_vars_exp[:-1]
        sb1_exp = sb1_exp[:-1]
        sb2_exp = sb2_exp[:-1]

    # Loop through the batches
    interp_data = []
    num_batches = len(target_vars_exp)
    for i in trange(num_batches, leave=False):
        # Load the batched data
        sb1_b = sb1_exp[i]
        sb2_b = sb2_exp[i]
        target_b = target_vars_exp[i]

        # Preprocess the data (drop the scanned var and signal flag)
        s1_proc = preprocessor.transform(sb1_b[:, :-2])
        s2_proc = preprocessor.transform(sb2_b[:, :-2])

        # Find the OT pairing between the distributions
        w = np.ones(len(s1_proc)) / len(s1_proc)
        ot_cost = ot.dist(s1_proc, s2_proc)
        ot_matrix = ot.emd(w, w, ot_cost, numItermax=1e8, numThreads="max")
        permute = np.argmax(ot_matrix, axis=1)  # Permutation of sb2 to match sb1

        # Do the interpolation in the preprocessed space
        if interp_preproc:
            sb1_b[:, :-2] = s1_proc
            sb2_b[:, :-2] = s2_proc

        # Use the OT pairing to pick out the appropriate permutation of sb2
        sb2_b = sb2_b[permute]

        # Find the interpolation fraction based on the current and target vars
        frac = (target_b - sb1_b[:, -2]) / (sb2_b[:, -2] - sb1_b[:, -2])

        # Interpolate the data and append to the list
        interp = sb1_b + frac[..., None] * (sb2_b - sb1_b)

        # If we interpolated in the preprocessed space, we need to inverse transform
        if interp_preproc:
            interp[:, :-2] = preprocessor.inverse_transform(interp[:, :-2])

        # Interpolate the data and append to the list
        interp_data.append(interp)

    # Combine all interpolations
    interp_data = np.vstack(interp_data)

    # The final column (label) is -1 for all interpolated data
    interp_data[:, -1] = -1
    return interp_data
