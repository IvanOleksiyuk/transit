# Some plotting functions
import os

import numpy as np
import torch 
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, auc
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
from scipy import stats
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap
colors = [(1, 1, 1, 0), (1, 0, 0, 1)]  # (R, G, B, Alpha) - from transparent to red
custom_reds = LinearSegmentedColormap.from_list("custom_reds", colors)

def shuffle_tensor(data):
    mx = torch.randperm(len(data), device=torch.device('cpu'))
    return data[mx]

def tensor2numpy(x):
    return x.detach().cpu().numpy()

font = {
    "family": "sans serif",
    "color": "black",
    "weight": "normal",
    "va": "center",
    "ha": "center",
    "size": 18,
}
# Font dict for the axis label "Normalised Entries" - Had to be handled separately
sep_font = {
    "family": "sans serif",
    "color": "black",
    "weight": "normal",
    "size": 15,
}

legend_dict = {
    "loc": "upper left",
    "bbox_to_anchor": (0.43, 0.92),
    "fontsize": 27,
    "frameon": False,
}

text_dict = {"color": "black", "fontsize": 25}


def get_weights(data):
    return np.ones_like(data) / len(data)


def get_bins(data, nbins=20, sd=None):
    max_ent = data.max().item()
    min_ent = data.min().item()
    if sd is not None:
        max_ent = max(max_ent, sd.max().item())
        min_ent = min(min_ent, sd.min().item())
    return np.linspace(min_ent, max_ent, num=nbins)

def get_bins_std(data, nbins=20, sd=None):
    std_ent = data.std().item()
    mean_ent = data.mean().item()
    return np.linspace(mean_ent - 4 * std_ent, mean_ent + 4 * std_ent, num=nbins)


def plot_marginals(
    originals, sample, labels=None, legend=True, axs_nms=None, limits=None, nbins=20
):
    data_dim = originals.shape[1]
    fig, axs = plt.subplots(1, data_dim, figsize=(5 * data_dim, 4))
    if labels is None:
        labels = ["original", "samples"]
    for i in range(data_dim):
        if sample is not None:
            bins = get_bins(originals[:, i], sd=sample[:, i], nbins=nbins)
        else:
            bins = get_bins(originals[:, i], nbins=nbins)
        axs[i].hist(
            tensor2numpy(originals[:, i]),
            label=labels[0],
            alpha=0.5,
            density=True,
            bins=bins,
            histtype="step",
        )
        # Plot samples drawn from the model
        if sample is not None:
            axs[i].hist(
                tensor2numpy(sample[:, i]),
                label=labels[1],
                alpha=0.5,
                density=True,
                bins=bins,
                histtype="step",
            )
        if axs_nms:
            axs[i].set_title(axs_nms[i])
        else:
            axs[i].set_title("Feature {}".format(i))
        if legend:
            axs[i].legend()
        if limits is not None:
            axs[i].set_xlim(limits)
    return fig


def add_hist(ax, data, bin, color, label):
    _, bins, _ = ax.hist(
        data,
        bins=bin,
        density=False,
        histtype="step",
        color=color,
        label=label,
        weights=get_weights(data),
    )
    return bins


def add_error_hist(
    ax, data, bins, color, error_bars=False, normalised=True, label="", norm=None
):
    y, binEdges = np.histogram(data, bins=bins)
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    width = 0.05
    norm_passed = norm is None
    n_fact = np.sum(y) if norm_passed else norm
    menStd = np.sqrt(y)
    if normalised or norm_passed:
        y = y / n_fact
        menStd = menStd / n_fact
    if error_bars:
        ax.errorbar(bincenters, y, yerr=menStd, color=color, fmt=".", label=label)
    else:
        ax.bar(
            bincenters,
            menStd,
            width=width,
            edgecolor=color,
            lw=0,
            fc=(0, 0, 0, 0),
            bottom=y,
            hatch="\\\\\\\\\\",
            label=label,
        )
        ax.bar(
            bincenters,
            -menStd,
            width=width,
            edgecolor=color,
            lw=0,
            fc=(0, 0, 0, 0),
            bottom=y,
            hatch="\\\\\\\\\\",
            label=label,
        )


def add_off_diagonal(axes, i, j, data, color):
    bini = get_bins(data[:, i])
    binj = get_bins(data[:, j])
    f1 = tensor2numpy(data[:, i])
    f2 = tensor2numpy(data[:, j])
    axes[i, j].hist2d(f1, f2, bins=[bini, binj], density=True, cmap=color)
    # axes[i, j].set_xlim([-1, 1.1])
    axes[i, j].set_ylim([-1, 1.2])
    # Pearson correlation
    # coef = np.corrcoef(f1, f2)[0, 1]
    # Spearman correlation between features
    coef, pval = stats.spearmanr(f1, f2)
    axes[i, j].annotate(
        f"SPR {coef:.2f}",
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        size=6,
    )


def add_contour(axes, i, j, data, sampled, x_bounds=None):
    if x_bounds is None:
        x_bounds = [-3, 3]
    sns.kdeplot(
        x=data[:, j],
        y=data[:, i],
        ax=axes[i, j],
        alpha=0.4,
        levels=3,
        color="blue",
        fill=True,
    )
    sns.kdeplot(
        x=sampled[:, j],
        y=sampled[:, i],
        ax=axes[i, j],
        alpha=0.4,
        levels=3,
        color="red",
        fill=True,
    )
    axes[i, j].set_xlim(x_bounds)
    axes[i, j].set_ylim(x_bounds)
    
def add_2d_hist(axes, i, j, data, sampled, x_bounds=None):
    if x_bounds is None:
        x_bounds = [-3, 3]
    else:
        if x_bounds[0] is None and x_bounds[1] is None:
            x_bounds = None
    
    bins = 30
    range_x = x_bounds
    range_y = x_bounds
    thresholds = [0, 0.1, 0.25, 0.5, 1.0]
    
    hist1, xedges, yedges = np.histogram2d(x=data[:, j], y=data[:, i], bins=bins, range=[range_x, range_y])
    hist2, _, _ = np.histogram2d(x=sampled[:, j], y=sampled[:, i], bins=[xedges, yedges])
    
    hist1_normalized = hist1 / np.quantile(hist1, 0.99)
    hist2_normalized = hist2 / np.quantile(hist2, 0.99)
    
    cmap1 = cm.Blues
    cmap2 = custom_reds
    norm = BoundaryNorm(thresholds, ncolors=cmap1.N, clip=True)

    # Plot the first histogram with thresholds
    ax=axes[i, j]
    ax.pcolormesh(xedges, yedges, hist1_normalized.T, cmap=cmap1, norm=norm, alpha=0.5)

    # Plot the second histogram with thresholds
    ax.pcolormesh(xedges, yedges, hist2_normalized.T, cmap=cmap2, norm=norm, alpha=0.5)

    axes[i, j].set_xlim(x_bounds)
    axes[i, j].set_ylim(x_bounds)


def plot_feature_spread(
    target_data,
    sampled,
    original_data = None,
    feature_nms=None,
    tag=None,
    save_dir=None,
    plot_mode="diagnose",
    combined=False,
    x_bounds=None,
    shuffle=True,
    do_2d_hist_instead_of_contour=False,
    save_name=None,
):
    nbins = 60
    n_features = sampled.shape[1] #- 1
    if x_bounds is None:
        x_bounds = [-3, 3]
    if n_features > 1:
        if shuffle:
            target_data = shuffle_tensor(target_data)
            sampled = shuffle_tensor(sampled)
            original_data = shuffle_tensor(original_data)
        if plot_mode == "diagnose":
            n_sample = 1000
        else:
            n_sample = 40_000
        if feature_nms is None:
            feature_nms = [f"feature_{i}" for i in range(n_features)]
        if tag is None:
            tag = "generic_jump"
        assert n_features == len(
            feature_nms
        ), "Number of feature names must match number of features"

        fig, axes = plt.subplots(
            n_features,
            n_features,
            figsize=(2 * n_features + 2, 2 * n_features + 1),
            gridspec_kw={"wspace": 0.03, "hspace": 0.03},
        )

        for i in range(n_features):
            if i != 0:
                axes[i, 0].set_ylabel(feature_nms[i])
            else:
                axes[0, 0].set_ylabel(
                    "Normalised Entries", horizontalalignment="right", y=1.0
                )
            for j in range(n_features):
                if j != 0:
                    axes[i, j].set_yticklabels([])
                axes[-1, j].set_xlabel(feature_nms[j])
                if i != n_features - 1:
                    axes[i, j].tick_params(
                        axis="x", which="both", direction="in", labelbottom=False
                    )
                    axes[i, j].set_yticks([-2, 0, 2])
                    if i == j == 0:
                        axes[i, j].tick_params(axis="y", colors="w")
                    elif j > 0:
                        axes[i, j].tick_params(
                            axis="y", which="both", direction="in", labelbottom=False
                        )
                if i == j:
                    og = target_data[:, i]
                    if x_bounds[0] is None and x_bounds[1] is None:
                        bins = get_bins_std(og, nbins=nbins)
                    else:
                        bins = np.linspace(-4, 4, nbins)
                    bins = add_hist(
                        axes[i, j],
                        sampled[:, i],
                        bins,
                        "red",
                        "Transformed",
                    )
                    add_hist(
                        axes[i, j],
                        target_data[:, i],
                        bins,
                        "blue",
                        "Target",
                    )
                    if original_data is not None:
                        add_hist(
                            axes[i, j],
                            original_data[:, i],
                            bins,
                            "green",
                            "Original",
                        )
                    add_error_hist(
                        axes[i, j], sampled[:, i], bins=bins, color="red"
                    )
                    add_error_hist(
                        axes[i, j],
                        target_data[:, i],
                        bins=bins,
                        color="blue",
                    )
                    axes[i, j].set_yticklabels([])
                    axes[i, j].set_xlim(x_bounds)

                if i > j:
                    # TODO fix the singular matrix issue
                    # try:
                    if do_2d_hist_instead_of_contour:
                        add_2d_hist(
                            axes,
                            i,
                            j,
                            target_data[:n_sample],
                            sampled[:n_sample],
                            x_bounds=x_bounds,
                        )
                    else:
                        add_contour(
                            axes,
                            i,
                            j,
                            target_data[:n_sample],
                            sampled[:n_sample],
                            x_bounds=x_bounds,
                        )
                    # except:
                    #     pass


                elif i < j:
                    axes[i, j].set_visible(False)

        if combined:
            transformation = r"$\mathrm{SB1} \rightarrow \mathrm{SR} \cup \mathrm{SB2} \rightarrow \mathrm{SR}$"
        else:
            transformation = rf"{tag[0]} $\rightarrow$ {tag[1]}"

        labels = ["Original", "Transformed", "Target"]
        ip_handles = [Line2D([], [], color="black", linestyle="--")]
        other_handles = [Line2D([], [], color=colors) for colors in ["red", "blue"]]
        handles = ip_handles + other_handles
        fig.text(
            2.15,
            -0.1,
            transformation,
            fontdict=text_dict,
            transform=axes[0, 0].transAxes,
        )
        fig.align_xlabels(axes)
        fig.align_ylabels(axes)
        fig.legend(handles, labels, **legend_dict)
        if save_name is not None:
            fig.savefig(save_dir / save_name, bbox_inches="tight")
        elif combined:
            fig.savefig(
                save_dir / f"feature_spread_SB12_to_SR.png", bbox_inches="tight"
            )
        else:
            fig.savefig(
                save_dir
                / "featurespread_{}_to_{}.png".format(tag[0].upper(), tag[1].upper()),
                bbox_inches="tight",
            dpi=400)


def plot_samples(samples, results_dir, nm, labels=None):
    if not isinstance(samples, list):
        samples = [samples, None]
    plots_dir = results_dir / "marginals"
    plots_dir.mkdir(exist_ok=True)
    fig = plot_marginals(*samples, labels=labels)
    fig.savefig(plots_dir / f"marginals_{nm}.png")
    plt.close(fig)


def auc_roc(labels, truth, name):
    fpr, tpr, _ = roc_curve(labels, truth)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"{roc_auc:.4f}", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend()
    fig.savefig(name)
    plt.close(fig)
    return roc_auc


def plot_classification(
    scores, results_directory, nm, label="sot_label", extra_stats=False
):
    """
    Plot the classification scores.
    :param scores: A dataframe produced by evaluate samples.
    :param results_directory: The directory used to save the results, a pathlib object.
    :param nm: The name to assign to the plot
    :param label: The label column to use in the scores dataframe, defaults to sample or truth.
    :return:
    """
    if label == "is_signal":
        scores = scores.loc[scores["is_signal"] != -1]
    if not extra_stats:
        scores = scores.loc[scores["sot_label"] != -2]
    os.makedirs(results_directory, exist_ok=True)
    roc_auc = auc_roc(
        scores[label], scores["predictions"], results_directory / f"roc_{nm}.png"
    )
    plot_sic(results_directory, nm, scores[label], scores["predictions"])
    return roc_auc


def add_sic(ax, fpr, tpr, nm, line="-", color=None):
    mx = fpr != 0.0
    fpr_nz = fpr[mx]
    tpr_nz = tpr[mx]
    ax.plot(
        tpr_nz,
        tpr_nz / (fpr_nz**0.5),
        linewidth=2,
        label=nm,
        linestyle=line,
        color=color,
    )


def plot_sic(sv_dir, title, labels, scores):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    rates_dict = {"direct": roc_curve(labels, scores)[:2]}
    rates_dict["random"] = [np.linspace(0, 1, 50), np.linspace(0, 1, 50)]
    for nm, rates in rates_dict.items():
        fpr, tpr = rates
        if nm == "random":
            line = "--"
            color = "k"
        else:
            line = "-"
            color = None

        add_sic(ax, fpr, tpr, nm, line=line, color=color)
    ax.set_ylabel("Significance improvement")
    ax.set_xlabel("Signal efficiency (true positive rate)")
    ax.set_title(title)
    y_max = min(20, ax.get_ylim()[1])
    ax.set_ylim([0, y_max])
    fig.savefig(sv_dir / f"{title}_sic.png")


def plot_sample_spreads(samples_dict, item, data_object, plots_dir, args):
    plots_dir = plots_dir / "spreads"
    plots_dir.mkdir(exist_ok=True)
    target, sampled = samples_dict[item][0], samples_dict[item][1]
    target, sampled = data_object.apply_preprocessing(
        target
    ), data_object.apply_preprocessing(sampled)

    if "sb1_2" in item:
        name = "SB12"
        combined = True
    else:
        split_names = item.split("_")
        base_region, target_region = split_names[0], split_names[-1]
        name = [base_region, target_region]
        combined = False
    plot_feature_spread(
        target,
        sampled,
        data_object.feature_names,
        name,
        plots_dir,
        plot_mode=args.plotmode,
        combined=combined,
    )


def plot_bump_hunt(evt_hist_dict, mjj_bins, num_signal, q_cuts=None, ax=None):
    flat_mjj = np.unique(mjj_bins)
    xbounds = flat_mjj[0], flat_mjj[-1]
    xdown, xup = xbounds

    lines = [
        Line2D([0], [0], color="y", lw=8, alpha=0.5),
        Line2D([0], [0], color="k", lw=2, ls="--"),
        Line2D([0], [0], color="r", marker="o", ls=""),
    ]

    for counter, keys in enumerate(evt_hist_dict.keys()):
        ax.hlines(
            evt_hist_dict[str(keys)][1],
            eval(keys)[0],
            eval(keys)[1],
            ls="--",
            color="k",
        )

        for idx, items in enumerate(evt_hist_dict[str(keys)][1]):
            if counter == 0:
                ax.text(
                    xdown - 80,
                    items,
                    f"{(1 - q_cuts[idx]) * 100:3.1f} %",
                    horizontalalignment="center",
                    verticalalignment="center",
                )
            ax.fill_between(
                [eval(keys)[0], eval(keys)[1]],
                [items - np.sqrt(items)],
                [items + np.sqrt(items)],
                color="blue",
                alpha=0.12,
            )

        yerr = np.sqrt(evt_hist_dict[str(keys)][1])
        ax.errorbar(
            np.ones_like(evt_hist_dict[str(keys)][1])
            * (eval(keys)[0] + eval(keys)[1])
            * 0.5,
            evt_hist_dict[str(keys)][0],
            yerr=yerr,
            ls="",
            marker="o",
            ms=4,
            elinewidth=2,
            color="r",
        )

    ax.set_xlabel(r"$m_{jj}$ [GeV]", fontsize=16)
    ax.set_xticks(flat_mjj)
    ax.set_xticklabels([str(flat) for flat in flat_mjj], rotation=30)
    ax.set_xlim(xdown - 170, xup + 50)

    ax.set_ylabel("Events / bin", fontsize=16, loc="top")
    ax.set_yscale("log")

    ax.legend(
        handles=lines,
        labels=[f"{num_signal} signal", "Expected", "Observed"],
        frameon=False,
        fontsize=14,
    )


def plot_mass_cuts(quantiles, cuts, data, plots_dir):
    fig, ax = plt.subplots()
    bins = None
    # Iterate through the different cuts
    for quant, threshold in zip(quantiles, cuts):
        mass = data["mass"][data["predictions"] > threshold]
        if bins is None:
            bins = get_bins(mass)
        # TODO this visualisation sucks a bit
        ax.hist(mass, bins=bins, histtype="step", label=quant)
    ax.set_yscale("log")
    fig.legend()
    fig.savefig(plots_dir / "cuts_to_sidebands.png")
    plt.close(fig)
