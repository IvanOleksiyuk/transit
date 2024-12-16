"""Functions and utilities for training the BDT ensembles."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from tqdm import trange


def apply_aggregate(x: np.ndarray, aggregate: str, dim: int = -1):
    if aggregate == "mean":
        x = np.mean(x, axis=dim)
    elif aggregate == "median":
        x = np.median(x, axis=dim)
    elif aggregate == "max":
        x = np.max(x, axis=dim)
    elif aggregate == "min":
        x = np.min(x, axis=dim)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregate}")
    return x


def k_fold_split(
    dataset: np.ndarray | list[np.ndarray], fold_idx: int, num_folds: int
) -> tuple[list, list, list]:
    """Perform a k-fold splitting of a numpy array."""
    # Check the settings
    assert num_folds > 0
    assert fold_idx < num_folds
    if isinstance(dataset, list):
        return tuple(k_fold_split(d) for d in dataset)

    # Get the fold index of each element in the set
    in_k = np.arange(len(dataset)) % num_folds

    # Get the indicies of the test val and train sets
    test_fold = fold_idx
    train_folds = [i for i in range(num_folds) if i != test_fold]

    # Get a mask to filter the dataset into train val and test
    in_test = in_k == test_fold
    in_train = np.isin(in_k, train_folds)

    # Use the masks to split the datasets
    test = dataset[in_test]
    train = dataset[in_train]
    return train, test


def fit_ensemble(
    train_x: np.ndarray,
    train_y: np.ndarray,
    num_ensemble: int,
    bdt_args: dict | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Train an ensemble of classifiers and return all predictions on the test set."""
    clf = VotingClassifier(
        [
            (
                str(j),
                HistGradientBoostingClassifier(
                    random_state=seed * 1000 + j, **(bdt_args or {})
                ),
            )
            for j in range(num_ensemble)
        ],
        n_jobs=-1,
        voting="soft",
    )
    clf.fit(train_x, train_y)
    return clf


def run_bdt_folds(
    x: np.ndarray,
    y: np.ndarray,
    num_folds: int,
    num_ensemble: int,
    extra_sig: np.ndarray | None = None,
    extra_bkg: np.ndarray | None = None,
    bdt_args: dict | None = None,
    seed: int = 0,
) -> tuple:
    """Run a BDT ensemble over each of the k-fold splits of the some data."""
    # The lists to hold the test set outputs
    all_x = []
    all_y = []
    all_preds = []
    extra_preds_dict = {}
    extra_bkg_preds_dict = {}
    # Loop over the folds
    for fold_idx in trange(num_folds, leave=False):
        # Split the data from the signal region using the fold_idx
        train_x, test_x = k_fold_split(x, fold_idx, num_folds)
        train_y, test_y = k_fold_split(y, fold_idx, num_folds)

        # Train an ensemble of decision trees
        clf = fit_ensemble(
            train_x[:, :-2],  # Dont include the actual label (-1) and mass (-2) 
            train_y,
            num_ensemble,
            bdt_args,
            seed,
        )

        # Get the predictions using the ensemble
        preds = clf.predict_proba(test_x[:, :-2])[:, 1]

        # Save the predictions and the fold outputs
        all_x.append(test_x)
        all_y.append(test_y)
        all_preds.append(preds)
        
        # Also get the outputs for the extra signal file
        extra_preds_dict[f"fold_{fold_idx}"] = (
            clf.predict_proba(extra_sig[:, :-2])[:, 1] if extra_sig is not None else None
        )
        extra_bkg_preds_dict[f"fold_{fold_idx}"] = (
            clf.predict_proba(extra_bkg[:, :-2])[:, 1] if extra_bkg is not None else None
        )


    # Combine the results
    all_x = np.vstack(all_x)
    all_y = np.hstack(all_y)
    all_preds = np.hstack(all_preds)

    # Also get the outputs for the extra signal file
    extra_preds = (
        clf.predict_proba(extra_sig[:, :-2])[:, 1] if extra_sig is not None else None
    )
    extra_bkg_preds = (
        clf.predict_proba(extra_bkg[:, :-2])[:, 1] if extra_bkg is not None else None
    )

    return all_x, all_y, all_preds, extra_preds, extra_bkg_preds, extra_preds_dict, extra_bkg_preds_dict
