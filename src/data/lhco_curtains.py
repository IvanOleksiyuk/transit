import numpy as np
import pandas as pd
INVERTED_KEYS = [
    "pxj2",
    "pyj2",
    "pzj2",
    "mj2",
    "tau1j2",
    "tau2j2",
    "tau3j2",
    "pxj1",
    "pyj1",
    "pzj1",
    "mj1",
    "tau1j1",
    "tau2j1",
    "tau3j1",
    "label",
]

def calculate_mass(four_vector: np.ndarray) -> np.ndarray:
    """Calculate the invariant mass of a four vector."""
    return (
        np.clip(
            four_vector[:, 0] ** 2 - np.sum(four_vector[:, 1:4] ** 2, axis=1),
            0,
            None,
        )
    ) ** 0.5

def convert_lhco_to_curtain_format(df=pd.DataFrame) -> pd.DataFrame:
    """Return a new dataframe with all variables needed by the curtains project."""

    # Peform reordering such that mj1 is the smaller of the two jets
    jet_order_mask = df["mj1"] < df["mj2"]
    proper_order = df.loc[jet_order_mask]
    improper_order = df.loc[~jet_order_mask]
    improper_order.columns = INVERTED_KEYS
    df = pd.concat((proper_order, improper_order))

    data = pd.DataFrame()
    data["is_signal"] = df["label"].astype("bool")

    # Individual jet kinematics
    for jet in ["j1", "j2"]:
        data[f"px_{jet}"] = df[f"px{jet}"]
        data[f"py_{jet}"] = df[f"py{jet}"]
        data[f"pz_{jet}"] = df[f"pz{jet}"]
        data[f"m_{jet}"] = df[f"m{jet}"]

        data[f"pt_{jet}"] = np.sqrt(data[f"px_{jet}"] ** 2 + data[f"py_{jet}"] ** 2)
        data[f"phi_{jet}"] = np.arctan2(data[f"py_{jet}"], data[f"px_{jet}"])
        data[f"eta_{jet}"] = np.arcsinh(data[f"pz_{jet}"] / data[f"pt_{jet}"])
        data[f"p_{jet}"] = np.sqrt(data[f"pz_{jet}"] ** 2 + data[f"pt_{jet}"] ** 2)
        data[f"e_{jet}"] = np.sqrt(data[f"m_{jet}"] ** 2 + data[f"p_{jet}"] ** 2)

    # Combined jet mass
    data["m_jj"] = calculate_mass(
        np.sum(
            [
                data[[f"e_j{i}", f"px_j{i}", f"py_j{i}", f"pz_j{i}"]].to_numpy()
                for i in range(1, 3)
            ],
            0,
        )
    )

    # Subjettiness ratios
    data["del_m"] = data["m_j2"] - data["m_j1"]
    data["tau21_j1"] = df["tau2j1"] / df["tau1j1"]
    data["tau32_j1"] = df["tau3j1"] / df["tau2j1"]
    data["tau21_j2"] = df["tau2j2"] / df["tau1j2"]
    data["tau32_j2"] = df["tau3j2"] / df["tau2j2"]
    data["m_n"] = data["m_jj"] - data["m_j1"] - data["m_j2"] # Fake variable delete as soon as possible
    
    # Other variables
    phi_1 = data["phi_j1"]
    phi_2 = data["phi_j2"]
    delPhi = np.arctan2(np.sin(phi_1 - phi_2), np.cos(phi_1 - phi_2))
    data["del_R"] = ((data["eta_j1"] - data["eta_j2"]) ** 2 + delPhi**2) ** (0.5)
    data["del_phi"] = abs(delPhi)
    data["del_eta"] = abs(data["eta_j1"] - data["eta_j2"])

    return data.dropna()