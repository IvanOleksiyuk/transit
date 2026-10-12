"""Memorisation test for TRANSIT: held-out vs training error, stratified by local density.

The question: does the transport fit individual training events where the
data are sparse? Aggregate train/valid losses cannot answer it -- memorisation
lives in the tails, and the two loaders also differ in how they weight the
sidebands (BinMix draws lower- and upper-SB bins equally; validation sees the
natural ~95/5 mixture). So this test

  1. draws fixed probe events from the training split and from the held-out
     split, separately for each sideband ("group") and UNIFORMLY IN MASS within
     it (equal draws per equal-width mass bin). Random draws would be dominated
     by the low-mass end of the lower sideband, where statistics are huge and
     nothing gets memorised; the sparse high-mass tails are what we want to
     see. (BinMix's own bins are equal-count, so equal draws from those would
     just reproduce the natural distribution within a sideband.)
  2. estimates the local density of *training* data at each probe event as the
     distance to its k-th nearest training neighbour in the standardised
     (features + mass) space, and cuts it into quantile bins ("strata") using
     the training probes' quantiles per group;
  3. computes per-event errors that do not depend on batching:
        reco          |D(E(x,m),m) - x|
        cycle         |E(D(E(x,m),m'),m') - E(x,m)|        (m' fixed per event)
        parallelity   |dx_rec - dx| / eps  along a fixed random direction
        rotationality |u.dv_rec - v.du_rec| / eps on a fixed random 2-plane
     -- the latter two are the per-event versions of the model's
     parralelity_loss / rotationality_loss;
  4. reports, per error, group and stratum,
        gap = median(err | held-out) / median(err | train)
     plus the medians themselves. gap ~ 1 everywhere means the model treats
     unseen events like seen ones; gap >> 1 only in the sparse strata is the
     fingerprint of memorising isolated points. Ratios are invariant to the
     sideband reweighting and to the overall error scale, so they compare
     across models.

Two entry points share the computation:
  * MemorisationMonitor -- a Lightning callback that evaluates the metric on
    fixed probe sets at the end of every validation epoch (while the EMA
    weights are swapped in) and logs it as memo/... . Disable with
    `enabled: false` in callbacks/default.yaml (or the hydra override
    step_train_template.callbacks.memorisation_monitor.enabled=false) to save
    the compute; the cost is one forward pass over the probe sets per
    validation.
  * run_memorisation_test -- the offline version for evaluation.py on the final
    checkpoint with more probes; result goes into evaluation_results.json.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset

log = logging.getLogger(__name__)

ERRORS = ("reco", "cycle", "parallelity", "rotationality")


# --- probe sets -------------------------------------------------------------------
def _split_members(dataset) -> tuple[Any, np.ndarray]:
    """(base ConcatDataset, global index of every member) for train_data / val_data."""
    if isinstance(dataset, Subset):
        return dataset.dataset, np.asarray(dataset.indices)
    return dataset, np.arange(len(dataset))


def _group_of(base, global_idx: np.ndarray) -> np.ndarray:
    """0 for members of the first concatenated dataset (lower SB), 1 for the second."""
    datasets = getattr(base, "datasets", None)
    if datasets is None or len(datasets) < 2:
        return np.zeros(len(global_idx), dtype=int)
    return (global_idx >= len(datasets[0])).astype(int)


def _masses_of(base, global_idx: np.ndarray, batch_size: int) -> np.ndarray:
    """Source mass of the given members, read from the datasets' `cond` frame."""
    datasets = getattr(base, "datasets", None) or [base]
    offsets = np.cumsum([0] + [len(d) for d in datasets])
    masses = np.empty(len(global_idx), dtype=np.float32)
    try:
        for i, d in enumerate(datasets):
            sel = (global_idx >= offsets[i]) & (global_idx < offsets[i + 1])
            local = global_idx[sel] - offsets[i]
            frame = d.data["cond"]
            if hasattr(frame, "to_numpy"):  # DataFrame
                arr = frame["m_jj"].to_numpy()
            else:  # numpy / torch with stored column names
                col = list(d.data_columns["cond"]).index("m_jj")
                arr = frame[:, col]
                if isinstance(arr, torch.Tensor):
                    arr = arr.cpu().numpy()
            masses[sel] = np.asarray(arr)[local]
        return masses
    except (AttributeError, KeyError, TypeError):
        log.info("memorisation: reading masses through the DataLoader (slow path)")
        return _load_xm(base, global_idx, batch_size)[1][:, 0].numpy()


def _load_xm(base, global_idx: np.ndarray, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Features and source mass of the given members, in the datamodule's own format."""
    xs, ms = [], []
    # a dataset that assembles whole batches itself (src.data.data.BatchFetchConcatDataset)
    # needs its own pass-through collate; Subset forwards its __getitems__
    loader_kwargs = {}
    collate = getattr(base, "prefetch_collate_fn", None)
    if collate is not None:
        loader_kwargs["collate_fn"] = collate
    for batch in DataLoader(Subset(base, global_idx.tolist()), batch_size=batch_size, shuffle=False, **loader_kwargs):
        xs.append(torch.as_tensor(batch[0]).float())
        ms.append(torch.as_tensor(batch[1]).float().reshape(len(batch[0]), -1))
    return torch.cat(xs), torch.cat(ms)


def build_probe_set(
    datamodule,
    n_probe_per_group: int = 10_000,
    n_reference: int = 200_000,
    k: int = 50,
    n_quantiles: int = 5,
    n_mass_bins: int = 10,
    batch_size: int = 4096,
    seed: int = 0,
) -> dict:
    """Fixed probe events, their density strata and the fixed random draws.

    n_mass_bins > 0: within each group, probes are drawn in equal numbers from
    equal-width mass bins (uniform in mass); 0 = plain random within the group.
    """
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    train_base, train_idx = _split_members(datamodule.train_data)
    val_base, val_idx = _split_members(datamodule.val_data)
    train_group = _group_of(train_base, train_idx)
    val_group = _group_of(val_base, val_idx)
    n_groups = int(max(train_group.max(), val_group.max())) + 1

    def pick(base, idx, group, n):
        chosen = []
        masses = _masses_of(base, idx, batch_size) if n_mass_bins > 0 else None
        for g in range(n_groups):
            members = idx[group == g]
            if n_mass_bins <= 0:
                chosen.append(rng.choice(members, size=min(n, len(members)), replace=False))
                continue
            m_g = masses[group == g]
            edges = np.linspace(m_g.min(), m_g.max(), n_mass_bins + 1)
            which = np.clip(np.searchsorted(edges, m_g, side="right") - 1, 0, n_mass_bins - 1)
            per_bin = max(1, n // n_mass_bins)
            for b in range(n_mass_bins):
                pool = members[which == b]
                if len(pool):
                    chosen.append(rng.choice(pool, size=min(per_bin, len(pool)), replace=False))
        return np.concatenate(chosen)

    probe_train = pick(train_base, train_idx, train_group, n_probe_per_group)
    probe_val = pick(val_base, val_idx, val_group, n_probe_per_group)
    x_tr, m_tr = _load_xm(train_base, probe_train, batch_size)
    x_va, m_va = _load_xm(val_base, probe_val, batch_size)
    g_tr = _group_of(train_base, probe_train)
    g_va = _group_of(val_base, probe_val)

    # Density reference: a subsample of ALL training events (both groups), in the
    # standardised feature space with the mass scaled to unit variance too.
    ref_idx = rng.choice(train_idx, size=min(n_reference, len(train_idx)), replace=False)
    x_ref, m_ref = _load_xm(train_base, ref_idx, batch_size)
    m_scale = float(m_ref.std()) + 1e-8
    space = lambda x, m: np.concatenate([x.numpy(), (m / m_scale).numpy()], axis=1)  # noqa: E731
    nn_index = NearestNeighbors(n_neighbors=k + 1).fit(space(x_ref, m_ref))
    # +1 and drop the first column: a training probe may be its own nearest neighbour.
    d_tr = nn_index.kneighbors(space(x_tr, m_tr))[0][:, -1]
    d_va = nn_index.kneighbors(space(x_va, m_va))[0][:, -1]

    # Strata: quantile edges of the TRAINING probes' k-NN distance, per group;
    # held-out probes are binned with the same edges.
    strata_tr = np.zeros(len(d_tr), dtype=int)
    strata_va = np.zeros(len(d_va), dtype=int)
    edges = {}
    for g in range(n_groups):
        q = np.quantile(d_tr[g_tr == g], np.linspace(0, 1, n_quantiles + 1)[1:-1])
        edges[g] = q
        strata_tr[g_tr == g] = np.searchsorted(q, d_tr[g_tr == g])
        strata_va[g_va == g] = np.searchsorted(q, d_va[g_va == g])

    # Fixed random draws: target mass (a random probe mass from the other rows,
    # i.e. the pooled training mass distribution) and the perturbation directions.
    gen = torch.Generator().manual_seed(seed)
    n_tr, n_va = len(x_tr), len(x_va)
    pool = m_tr[torch.randperm(n_tr, generator=gen)]
    m_target_tr = pool[:n_tr]
    m_target_va = pool[torch.randint(n_tr, (n_va,), generator=gen)]

    def directions(n, d):
        u = torch.randn(n, d, generator=gen)
        u = u / u.norm(dim=1, keepdim=True)
        v = torch.randn(n, d, generator=gen)
        v = v - (u * v).sum(1, keepdim=True) * u
        v = v / v.norm(dim=1, keepdim=True)
        return u, v

    d = x_tr.shape[1]
    u_tr, v_tr = directions(n_tr, d)
    u_va, v_va = directions(n_va, d)
    log.info(
        f"memorisation probe set: {n_tr} train + {n_va} held-out events, "
        f"{n_groups} group(s), {n_quantiles} density strata, k={k} on {len(ref_idx)} reference events"
    )
    return {
        "train": {"x": x_tr, "m": m_tr, "m_target": m_target_tr, "u": u_tr, "v": v_tr, "group": g_tr, "stratum": strata_tr, "knn_dist": d_tr},
        "val": {"x": x_va, "m": m_va, "m_target": m_target_va, "u": u_va, "v": v_va, "group": g_va, "stratum": strata_va, "knn_dist": d_va},
        "n_groups": n_groups,
        "n_quantiles": n_quantiles,
        "edges": edges,
        "k": k,
    }


# --- per-event errors -----------------------------------------------------------------
@torch.no_grad()
def per_event_errors(model, probes: dict, eps_scale: float = 1e-2, batch_size: int = 4096) -> dict[str, np.ndarray]:
    """The four per-event errors for one probe split, on the model's current weights."""
    device = next(model.parameters()).device
    out = {name: [] for name in ERRORS}
    n = len(probes["x"])
    for start in range(0, n, batch_size):
        sl = slice(start, start + batch_size)
        x = probes["x"][sl].to(device)
        m = probes["m"][sl].to(device)
        mt = probes["m_target"][sl].to(device)
        u = probes["u"][sl].to(device)
        v = probes["v"][sl].to(device)

        if getattr(model, "do_dequantization", False):
            x = model.dequantization_layer(x)
        if model.add_standardizing_layer:
            x = model.std_layer_x(x)
            m = model.std_layer_ctxt(m)
            mt = model.std_layer_ctxt(mt)

        z = model.encode_content(x, m)
        style = model.encode_style(m)
        style_t = model.encode_style(mt)
        rec = model.decode(z, style)
        transported = model.decode(z, style_t)
        z_n = model.encode_content(transported, mt)
        out["reco"].append((rec - x).norm(dim=1))
        out["cycle"].append((z_n - z).norm(dim=1))

        # parallelity and rotationality, exactly as the losses but per event and
        # with fixed directions; decoded at the (fixed) target mass.
        z_u = model.encode_content(x + eps_scale * u, m)
        z_v = model.encode_content(x + eps_scale * v, m)
        x_rec = model.decode(z, style_t)
        du = model.decode(z_u, style_t) - x_rec
        dv = model.decode(z_v, style_t) - x_rec
        out["parallelity"].append((du - eps_scale * u).norm(dim=1) / eps_scale)
        out["rotationality"].append(((u * dv).sum(1) - (v * du).sum(1)).abs() / eps_scale)
    return {name: torch.cat(vals).cpu().numpy() for name, vals in out.items()}


# --- aggregation -------------------------------------------------------------------------
GROUP_NAMES = {0: "lowSB", 1: "highSB"}


def stratified_gaps(err_train: dict, err_val: dict, probe_set: dict) -> dict[str, float]:
    """memo/<error>/<group>/{gap,train_median,val_median}_q<i>, plus summaries."""
    tr, va = probe_set["train"], probe_set["val"]
    n_q = probe_set["n_quantiles"]
    metrics: dict[str, float] = {}
    for name in ERRORS:
        for g in range(probe_set["n_groups"]):
            gname = GROUP_NAMES.get(g, f"group{g}")
            gaps = []
            for q in range(n_q):
                sel_tr = (tr["group"] == g) & (tr["stratum"] == q)
                sel_va = (va["group"] == g) & (va["stratum"] == q)
                if sel_tr.sum() < 10 or sel_va.sum() < 10:
                    continue
                med_tr = float(np.median(err_train[name][sel_tr]))
                med_va = float(np.median(err_val[name][sel_va]))
                gap = med_va / max(med_tr, 1e-12)
                gaps.append(gap)
                key = f"memo/{name}/{gname}"
                metrics[f"{key}/gap_q{q + 1}"] = gap
                metrics[f"{key}/train_median_q{q + 1}"] = med_tr
                metrics[f"{key}/val_median_q{q + 1}"] = med_va
            if len(gaps) >= 2:
                # one number per error and group: how much worse the sparsest
                # stratum's gap is than the densest one's
                metrics[f"memo/{name}/{gname}/sparse_over_dense"] = gaps[-1] / max(gaps[0], 1e-12)
    return metrics


def run_memorisation_test(model, datamodule, eps_scale: float = 1e-2, batch_size: int = 4096, probe_set: dict | None = None, **probe_kwargs) -> dict[str, float]:
    """Build (or reuse) the probe set and return the metric dict for the current weights."""
    if probe_set is None:
        probe_set = build_probe_set(datamodule, batch_size=batch_size, **probe_kwargs)
    was_training = model.training
    model.eval()
    try:
        err_tr = per_event_errors(model, probe_set["train"], eps_scale, batch_size)
        err_va = per_event_errors(model, probe_set["val"], eps_scale, batch_size)
    finally:
        model.train(was_training)
    return stratified_gaps(err_tr, err_va, probe_set)


# --- Lightning callback -----------------------------------------------------------------
class MemorisationMonitor(pl.Callback):
    """Log the memorisation metrics at the end of every validation epoch.

    Runs inside on_validation_epoch_end, i.e. before the LightningModule's own
    hook restores the raw weights, so with EMA enabled it evaluates the EMA
    weights -- the same ones validation and template generation use.
    `enabled: false` turns it into a no-op.
    """

    def __init__(
        self,
        enabled: bool = True,
        n_probe_per_group: int = 10_000,
        n_reference: int = 200_000,
        k: int = 50,
        n_quantiles: int = 5,
        n_mass_bins: int = 10,
        eps_scale: float = 1e-2,
        batch_size: int = 4096,
        seed: int = 0,
        every_n_validations: int = 1,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.probe_kwargs = dict(
            n_probe_per_group=n_probe_per_group,
            n_reference=n_reference,
            k=k,
            n_quantiles=n_quantiles,
            n_mass_bins=n_mass_bins,
            seed=seed,
        )
        self.eps_scale = eps_scale
        self.batch_size = batch_size
        self.every_n_validations = max(1, int(every_n_validations))
        self.probe_set: dict | None = None
        self._n_validations = 0

    def _ensure_probes(self, trainer) -> None:
        if self.probe_set is None:
            self.probe_set = build_probe_set(trainer.datamodule, batch_size=self.batch_size, **self.probe_kwargs)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self.enabled or trainer.sanity_checking:
            return
        self._n_validations += 1
        if (self._n_validations - 1) % self.every_n_validations:
            return
        self._ensure_probes(trainer)
        metrics = run_memorisation_test(
            pl_module, trainer.datamodule, eps_scale=self.eps_scale, batch_size=self.batch_size, probe_set=self.probe_set
        )
        pl_module.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True)
