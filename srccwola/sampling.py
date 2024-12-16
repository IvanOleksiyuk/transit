"""Functions and utilities for parameter fitting."""

import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

os.environ["ZFIT_DISABLE_TF_WARNINGS"] = "1"
import zfit
from zfit import z


def fit_and_sample_scanned(
    data: np.ndarray,
    sb1: np.ndarray,
    sb2: np.ndarray,
    output_path: Path,
    gen_region: tuple[int, int],
    sampler_class: str = "KDE",
    pdf_fit_on_all: bool = True,
    template_size: int = 500_000,
) -> np.ndarray:
    """Fit the pdf to the scanned variable and sample in the signal region."""
    # Get the appropriate sampler
    pdf = globals()[sampler_class](output_path)

    # Fit using all data or just the sidebands
    pdf.fit(data[:, -2]) if pdf_fit_on_all else pdf.fit(sb1[:, -2], sb2[:, -2])

    # Do this iteratively as some generators sample outside SR
    target_vars = []
    while len(target_vars) < template_size:
        samples = pdf.sample(template_size)
        samples = samples[(gen_region[0] <= samples) & (samples < gen_region[1])]
        target_vars.extend(samples)
    return np.array(target_vars)[:template_size]


class KDE:
    def __init__(self, directory: Path | None = None) -> None:
        self.pdf = None
        self.directory = directory

    def fit(self, data: np.ndarray) -> None:
        self.pdf = gaussian_kde(data)

    def sample(self, n: int) -> np.ndarray:
        return self.pdf.resample(n).T


class ATLASDIJETFit:
    def __init__(self, directory: Path | None = None) -> None:
        self.pdf = None
        self.directory = directory

    def fit(self, data1, data2) -> None:
        self.pdf = get_fit_pdf(data1, data2, self.directory)

    def sample(self, n: int) -> np.ndarray:
        return self.pdf.sample(n).T


###################################################################################


class DijetZFit(zfit.pdf.ZPDF):
    """A zfit class for the three/four parameter dijet fit function."""

    _PARAMS = ["p1", "p2", "p3", "p4"]  # noqa: RUF012

    def __init__(
        self, root_s=None, *args, default_window: np.ndarray | None = None, **kwargs
    ):
        """Parameters
        ----------
        root_s: float:
            The CM energy value for the fit.
        """
        super().__init__(*args, **kwargs)
        self.root_s = root_s
        self.default_window = default_window

    def _unnormalized_pdf(self, x):
        data = z.unstack_x(x)
        p1 = self.params["p1"]
        p2 = self.params["p2"]
        p3 = self.params["p3"]
        p4 = self.params["p4"]

        return p1 * z.pow(1 - data, p2) * z.pow(data, p3 + p4 * z.math.log(data))

    def sample(self, *args, limits: np.ndarray | None = None, **kwargs) -> np.ndarray:
        if limits is None:
            limits = self.default_window
        limits = limits / self.root_s
        x = super().sample(*args, limits=limits, **kwargs)
        return x.numpy().squeeze() * self.root_s


def fit_plot_summary(model, data, edges, sv_dir, bins=80, samples=10000):
    """Save a plot to verfify a zfit object."""
    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    x = np.linspace(*edges, samples)
    y = zfit.run(model.pdf(x / model.root_s)) / model.root_s

    ax.hist(data, bins=bins, density=True, label="Data", color="green")
    ax.plot(x, y, label="Fit", lw=2, color="red")
    ax.set_xlabel("Mass [GeV]")
    ax.set_ylabel("A.U.")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"{sv_dir}/fit_summary.png")
    plt.close()


def get_fit_pdf(
    sb1_masses: np.ndarray, sb2_masses: np.ndarray, sv_dir: str = "."
) -> DijetZFit:
    """Fit the dijet_ATLAS_fit pdf to the data.

    Parameters
    ----------
    sb1_masses: np.ndarray:
        The masses of the first sideband.
    sb2_masses: np.ndarray:
        The masses of the second sideband.
    sv_dir: str:
        The directory to save the fit pdf results/plots to.

    Returns
    -------
    zfit.pdf.ZPDF: fit model
    results: fit results
    """
    # Get the data, ATLAS di jet function expects values in mass/ sqrt(s)
    root_s = 13000.0
    data = [sb1_masses, sb2_masses]
    data = [item / root_s for item in data]

    # We need the mass ranges for each sideband for the fit
    sideband1 = zfit.Space("massRange", limits=(data[0].min(), data[0].max()))
    sideband2 = zfit.Space("massRange", limits=(data[1].min(), data[1].max()))
    mass_range = sideband1 + sideband2

    # parameters for the dijet_ATLAS_fit function
    p1 = zfit.Parameter("p1", 100.0, lower=0.0)
    p2 = zfit.Parameter("p2", 10.0, lower=0.0)
    p3 = zfit.Parameter("p3", -0.1, upper=0.0)
    p4 = zfit.Parameter("p4", -1.0)

    # Combine the sidebands and initialise the fit object
    data = np.concatenate(data)
    zdata = zfit.Data.from_numpy(obs=mass_range, array=data)
    model = DijetZFit(
        obs=mass_range,
        p1=p1,
        p2=p2,
        p3=p3,
        p4=p4,
        root_s=root_s,
        default_window=np.array([sb1_masses.max(), sb2_masses.min()]),
    )

    # Run the fit using zfit and the unbinned data
    nll = zfit.loss.UnbinnedNLL(model=model, data=zdata)
    minimizer = zfit.minimize.Minuit()
    results = minimizer.minimize(nll)

    # Save plot and text file to verify that the fit was a success
    fit_plot_summary(
        model,
        np.concatenate([sb1_masses, sb2_masses]),
        (sb1_masses.min(), sb2_masses.max()),
        sv_dir,
        bins=80,
        samples=10000,
    )

    with open(Path(sv_dir, "fit_results.txt"), "w") as f:
        f.write(f"Minima: {results.fmin}\n")
        f.write(f"Converge Status: {results.converged}\n")
        f.write(f"Valid Status: {results.valid}\n")
        f.write(f"Hessian: {results.hesse(name='hesse')}\n")

    return model
