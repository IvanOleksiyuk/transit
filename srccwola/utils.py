"""General utility functions."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import preprocessing

from mltools.plotting import plot_multi_correlations

FEAT_LATEX = {
    "m_j1": r"$m_{j1}$",
    "del_m": r"$\Delta m_j$",
    "tau21_j1": r"$\tau_{21}^{j1}$",
    "tau21_j2": r"$\tau_{21}^{j2}$",
    "del_R": r"$\Delta R$",
}


def get_cwola_datasets(
    data: np.ndarray, template: np.ndarray, mode: str = "standard"
) -> tuple[np.ndarray, np.ndarray]:
    """Get the signal and background for training the CWoLa model."""
    # Standard: Signal = data, Background = template
    if mode == "standard":
        signal = data
        background = template

    # Idealised: Signal = data_sig + half data_bkg, Background = half data_bkg
    elif mode == "idealised":
        is_sig = data[:, -1] > 0
        data_s = data[is_sig]
        data_b = data[~is_sig]
        background, data_b = np.array_split(data_b, 2)
        signal = np.vstack((data_s, data_b))

    # Supervised: Signal = data_sig, Background = data_bkg
    elif mode == "supervised":
        is_sig = data[:, -1] > 0
        signal = data[is_sig]
        background = data[~is_sig]

    # Raise an error if the mode is not recognised
    else:
        raise ValueError(
            f"Mode {mode} not recognised, must be standard, idealised or supervised."
        )

    return signal, background


def get_preprocessor(preproc: str) -> preprocessing.StandardScaler:
    if preproc == "quantile":
        return preprocessing.QuantileTransformer(output_distribution="normal")
    if preproc == "power":
        return preprocessing.PowerTransformer()
    if preproc == "standard":
        return preprocessing.StandardScaler()
    if preproc == "minmax":
        return preprocessing.MinMaxScaler()
    return None


def load_data(
    file_path: str,
    inpt_list: list[str],
    scanned_var: str,
    signal_flag: str,
    sideband_1: tuple[int, int],
    sideband_2: tuple[int, int],
    num_signal: int,
    output_path: Path,
    plot_bands: bool,
    pure_sidebands: bool = False,
) -> tuple[np.ndarray]:
    """Load the data and split it into the sidebands and the signal region."""
    data = pd.read_hdf(file_path)
    data = data[[*inpt_list, scanned_var, signal_flag]]

    # Reduce the number of signal events to the specified amount
    is_sig = data[signal_flag]
    extra_sig = data[is_sig][num_signal:]
    data = pd.concat([data[~is_sig], data[is_sig][:num_signal]])

    # Above step made it bkg -> sig, need to shuffle
    data = data.sample(frac=1).reset_index(drop=True)

    # Split the data into the two sidebands and the signal region
    scanned_col = data[scanned_var]
    sb1 = data[scanned_col.between(*sideband_1)]
    sb2 = data[scanned_col.between(*sideband_2)]
    sr = data[scanned_col.between(sideband_1[1], sideband_2[0])]  # Between
    data = data[scanned_col.between(sideband_1[0], sideband_2[1])]  # Inclusive
    extra_sig = extra_sig[extra_sig[scanned_var].between(sideband_1[1], sideband_2[0])]

    # Make the sidebands pure if required
    if pure_sidebands:
        sb1 = sb1[sb1[signal_flag] == 0]
        sb2 = sb2[sb2[signal_flag] == 0]

    # Convert to numpy (faster)
    data = data.to_numpy().astype(np.float32)
    sb1 = sb1.to_numpy().astype(np.float32)
    sb2 = sb2.to_numpy().astype(np.float32)
    sr = sr.to_numpy().astype(np.float32)
    extra_sig = extra_sig.to_numpy().astype(np.float32)

    # Make some plots
    if plot_bands:
        plot_multi_correlations(
            data_list=[sb1[:, :-2], sr[:, :-2], sb2[:, :-2]],
            data_labels=["SB1", "SR", "SB2"],
            col_labels=[FEAT_LATEX[x] for x in inpt_list],
            n_bins=30,
            fig_scale=1,
            n_kde_points=15,
            path=output_path / f"regions_{num_signal}.pdf",
            hist_kwargs=[{"color": "orange"}, {"color": "red"}, {"color": "green"}],
        )

    return data, sb1, sb2, sr, extra_sig
