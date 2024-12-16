import os

os.environ[
    "ZFIT_DISABLE_TF_WARNINGS"
] = "1"  # Disable the TF warning you get when you import zfit.

import pandas as pd
from zfit import z

import torch

import matplotlib.pyplot as plt
import zfit

zfit.settings.changed_warnings.hesse_name = False
import numpy as np


class dijet_ATLAS_fit(zfit.pdf.ZPDF):
    """
    This is a zfit class for the three parameter dijet fit function.


    public methods
    ----------
    sample: returns a sample of mass in a given band from the fit pdf.
    """

    _PARAMS = ["p1", "p2", "p3"]

    def __init__(self, root_s=None, *args, **kwargs):
        """
        The init method for the dijet_ATLAS_fit class.

        Parameters:
        -----------
        root_s: float:
            The CM energy value for the fit.
        """
        super().__init__(*args, **kwargs)
        self.root_s = root_s

    def _unnormalized_pdf(self, x):
        data = z.unstack_x(x)
        p1 = self.params["p1"]
        p2 = self.params["p2"]
        p3 = self.params["p3"]

        return p1 * z.pow(1 - data, p2) * z.pow(data, p3)

    def sample(self, *args, limits=None, **kwargs):
        if limits is None:
            raise ValueError(
                "No limits specified for sampling from fit pdf. Limits must be specified"
            )
        else:
            limits = [limit / self.root_s for limit in limits]

        x = super().sample(*args, limits=limits, **kwargs)
        x = x.to_pandas() * self.root_s

        return x.to_numpy().reshape(-1, 1)


def fit_plot_summary(model, data, edges, sv_dir, **kwargs):
    bins = kwargs["bins"] if "bins" in kwargs else 80
    samples = kwargs["samples"] if "samples" in kwargs else 10000

    root_s = model.root_s
    lower, higher = edges
    data_np = zfit.run(data.value()[:, 0])

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    x = np.linspace(lower, higher, samples) / root_s
    y = zfit.run(model.pdf(x))

    ax.hist(data_np, bins=bins, density=True, label="Data", color="green")
    ax.plot(x, y, label="Fit", lw=2, color="red")
    ax.set_xlabel("Mass [GeV]")
    ax.set_ylabel("A.U.")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"{sv_dir}/fit_summary.png")
    plt.clf()


def pd_torch_to_numpy(mass):
    if torch.is_tensor(mass):
        mass = mass.cpu().numpy()
    elif isinstance(mass, pd.Series):
        mass = mass.to_numpy()
    return mass


def get_fit_pdf(sb1, sb2, sv_dir):
    """
    This is a function to fit the dijet_ATLAS_fit pdf to the data.

    Parameters:
    -----------
    sb1: masses from sideband 1
    sb2: masses from sideband 2
    sv_dir: str:
        The directory to save the fit pdf results/plots to.

    Returns:
    --------
    zfit.pdf.ZPDF: fit model
    results: fit results
    """

    # Get the data

    sb1 = pd_torch_to_numpy(sb1)
    sb2 = pd_torch_to_numpy(sb2)

    edge1, edge2 = sb1.min(), sb2.max()
    data = [sb1, sb2]

    root_s = 13000.0  # The ATLAS di jet function expects values in mass/ sqrt(s)
    data = [item / root_s for item in data]

    sideband1 = zfit.Space("massRange", limits=(data[0].min(), data[0].max()))
    sideband2 = zfit.Space("massRange", limits=(data[1].min(), data[1].max()))
    mass_range = sideband1 + sideband2

    data = np.concatenate(data)

    # parameters for the dijet_ATLAS_fit function
    p1 = zfit.Parameter("p1", 100.0, lower_limit=0.0)
    p2 = zfit.Parameter("p2", 10.0, lower_limit=0.0)
    p3 = zfit.Parameter("p3", -0.1, upper_limit=0.0)

    data = zfit.Data.from_numpy(obs=mass_range, array=data)
    model = dijet_ATLAS_fit(obs=mass_range, p1=p1, p2=p2, p3=p3, root_s=root_s)

    nll = zfit.loss.UnbinnedNLL(model=model, data=data)
    minimizer = zfit.minimize.Minuit()
    results = minimizer.minimize(nll)

    # saving fit results/plots:
    fit_plot_summary(model, data, (edge1, edge2), sv_dir, bins=80, samples=10000)

    with open(sv_dir / "fit_results.txt", "w") as f:
        f.write(f"Minima: {results.fmin}\n")
        f.write(f"Converge Status: {results.converged}\n")
        f.write(f"Valid Status: {results.valid}\n")
        f.write(f"Hessian: {results.hesse()}\n")

    return model


def get_target_masses(
    top_dir,
    sb1_masses,
    sb2_masses,
    sr_masses,
    sr_bands,
    n_oversample=1,
    use_sampler=True,
):
    if use_sampler:
        fit_directory = top_dir.sub_dir("mass_sampler")
        # Define a mass sampling object
        mass_sampler = get_fit_pdf(sb1_masses, sb2_masses, fit_directory)
        # Sample the necessary number of masses
        n_sb_data = len(sb1_masses) + len(sb2_masses)
        target_masses = mass_sampler.sample(n_sb_data * n_oversample, limits=sr_bands)
    else:
        target_masses = sr_masses.to_numpy().reshape(-1, 1)
        target_masses = np.tile(target_masses, (1, n_oversample)).reshape(-1, 1)
    return target_masses
