from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier as hgbc
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Union, Sequence


def _to_numpy(x: Union[np.ndarray, pd.DataFrame, torch.Tensor]) -> np.ndarray:
	if isinstance(x, torch.Tensor):
		return x.detach().cpu().numpy()
	if isinstance(x, pd.DataFrame):
		return x.values
	return np.asarray(x)


def bdt_two_sample_test(
	x1: Union[np.ndarray, pd.DataFrame, torch.Tensor],
	x2: Union[np.ndarray, pd.DataFrame, torch.Tensor],
	n_splits: int = 5,
	random_state: int | None = None,
	use_scaler: bool = True,
	return_folds: bool = False,
	verbose: bool = False,
	**hgbc_kwargs,
) -> Union[float, Tuple[float, Sequence[float]]]:
	"""Run a two-sample classifier test using histogram-gradient-boosting BDTs.

	Contract
	- Inputs: x1, x2  : arrays-like with shape (n_samples, n_features).
	- Output: mean AUC across CV folds (and optionally per-fold AUCs).

	Procedure
	- Concatenate x1 and x2, create labels (0 for x1, 1 for x2).
	- Use StratifiedKFold(n_splits) to split indices. For each split train on
	  n_splits-1 folds and evaluate on the remaining fold. Compute ROC AUC.
	- Return the mean AUC across folds (and optionally the list of fold AUCs).

	Notes
	- Data are converted to numpy; torch tensors and pandas DataFrames are supported.
	- By default a StandardScaler is fit on the train set for each fold.
	- Additional keyword arguments are forwarded to sklearn's
	  HistGradientBoostingClassifier constructor.
	"""

	x1_np = _to_numpy(x1)
	x2_np = _to_numpy(x2)

	if x1_np.ndim != 2 or x2_np.ndim != 2:
		raise ValueError("x1 and x2 must be 2D arrays of shape (n_samples, n_features)")
	if x1_np.shape[1] != x2_np.shape[1]:
		raise ValueError("x1 and x2 must have the same number of features")

	X = np.vstack([x1_np, x2_np])
	y = np.concatenate([np.zeros(len(x1_np), dtype=int), np.ones(len(x2_np), dtype=int)])

	skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
	fold_aucs = []

	for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
		X_train, X_test = X[train_idx], X[test_idx]
		y_train, y_test = y[train_idx], y[test_idx]

		if use_scaler:
			scaler = StandardScaler()
			X_train = scaler.fit_transform(X_train)
			X_test = scaler.transform(X_test)

		clf = hgbc(**hgbc_kwargs)
		clf.fit(X_train, y_train)

		if hasattr(clf, "predict_proba"):
			probs = clf.predict_proba(X_test)[:, 1]
		else:
			# fallback to decision_function if predict_proba not present
			probs = clf.decision_function(X_test)

		fpr, tpr, _ = roc_curve(y_test, probs)
		fold_auc = auc(fpr, tpr)
		fold_aucs.append(float(fold_auc))
		if verbose:
			print(f"Fold {fold_idx}: AUC={fold_auc:.6f}")

	mean_auc = float(np.mean(fold_aucs))
	if return_folds:
		return mean_auc, fold_aucs
	return mean_auc


### Some example usage functions below ###

def _generate_normals(mean: float, std: float, n_samples: int, n_features: int, seed: int | None = None) -> np.ndarray:
	rng = np.random.default_rng(seed)
	return rng.normal(loc=mean, scale=std, size=(n_samples, n_features))


def _run_example_same_and_diff():
	"""Generate two example tests and print AUCs.

	- Case A: x1 vs x2 where both are N(0,1) -> expect AUC approx 0.5
	- Case B: x1 vs x2 where x2 ~ N(1.5,1) -> expect AUC > 0.5
	"""
	n_samples = 2000
	n_features = 7
	seed = 42

	x1 = _generate_normals(0.0, 1.0, n_samples, n_features, seed=seed)
	x2_same = _generate_normals(0.0, 1.0, n_samples, n_features, seed=seed + 1)
	x2_diff = _generate_normals(0.5, 1.0, n_samples, n_features, seed=seed + 2)

	print("Running BDT two-sample test: same distribution (expect AUC ~0.5)")
	auc_same = bdt_two_sample_test(x1, x2_same, n_splits=5, random_state=0, verbose=True)
	print(f"Mean AUC (same): {auc_same:.6f}\n")

	print("Running BDT two-sample test: different mean (expect AUC >0.5)")
	auc_diff = bdt_two_sample_test(x1, x2_diff, n_splits=5, random_state=0, verbose=True)
	print(f"Mean AUC (different): {auc_diff:.6f}\n")


if __name__ == "__main__":
	_run_example_same_and_diff()

