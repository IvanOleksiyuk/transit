"""Mix of utility functions specifically for pytorch."""
import os
from functools import partial
from typing import Any, Iterable, Mapping, Tuple, Union

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim.lr_scheduler as schd
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, Subset, random_split

from .loss import ChampferLoss, MyBCEWithLogit, VAELoss
from .schedulers import CyclicWithWarmup, LinearWarmupRootDecay, WarmupToConstant


def sum_except_batch(x: T.Tensor, num_batch_dims: int = 1) -> T.Tensor:
    """Sum all elements of x except for the first num_batch_dims dimensions."""
    return T.sum(x, dim=list(range(num_batch_dims, x.ndim)))


def append_dims(x: T.Tensor, target_dims: int, add_to_front: bool = False) -> T.Tensor:
    """Append dimensions of size 1 to the end or front of a tensor.

    Parameters
    ----------
    x : T.Tensor
        The input tensor to be reshaped.
    target_dims : int
        The target number of dimensions for the output tensor.
    add_to_front : bool, optional
        If True, dimensions are added to the front of the tensor.
        If False, dimensions are added to the end of the tensor.
        Defaults to False.

    Returns
    -------
    T.Tensor
        The reshaped tensor with target_dims dimensions.

    Raises
    ------
    ValueError
        If the input tensor already has more dimensions than target_dims.

    Examples
    --------
    >>> x = T.tensor([1, 2, 3])
    >>> x.shape
    torch.Size([3])

    >>> append_dims(x, 3)
    tensor([[[1]], [[2]], [[3]]])
    >>> append_dims(x, 3).shape
    torch.Size([3, 1, 1])

    >>> append_dims(x, 3, add_to_front=True)
    tensor([[[[1, 2, 3]]]])
    >>> append_dims(x, 3, add_to_front=True).shape
    torch.Size([1, 1, 3])
    """
    dim_diff = target_dims - x.dim()
    if dim_diff < 0:
        raise ValueError(f"x has more dims ({x.ndim}) than target ({target_dims})")
    if add_to_front:
        return x[(None,) * dim_diff + (...,)]  # x.view(*dim_diff * (1,), *x.shape)
    return x[(...,) + (None,) * dim_diff]  # x.view(*x.shape, *dim_diff * (1,))


def dtype_lookup(dtype: Any) -> T.dtype:
    """Return a torch dtype based on a string."""
    return {
        "double": T.float64,
        "float": T.float32,
        "half": T.float16,
        "int": T.int32,
        "long": T.int64,
    }[dtype]


class GradsOff:
    """Context manager for passing through a model without it tracking gradients."""

    def __init__(self, model) -> None:
        self.model = model

    def __enter__(self) -> None:
        self.model.requires_grad_(False)

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.model.requires_grad_(True)


def rms(tens: T.Tensor, dim: int = 0) -> T.Tensor:
    """Return RMS of a tensor along a dimension."""
    return tens.pow(2).mean(dim=dim).sqrt()


def rmse(tens_a: T.Tensor, tens_b: T.Tensor, dim: int = 0) -> T.Tensor:
    """Return RMSE without using torch's warning filled mseloss method."""
    return (tens_a - tens_b).pow(2).mean(dim=dim).sqrt()


def get_act(name: str) -> nn.Module:
    """Return a pytorch activation function given a name."""
    if isinstance(name, partial):
        return name()
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name == "lrlu":
        return nn.LeakyReLU(0.1)
    if name == "silu" or name == "swish":
        return nn.SiLU()
    if name == "selu":
        return nn.SELU()
    if name == "softmax":
        return nn.Softmax()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "softmax":
        return nn.Softmax()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "identity" or name == "none":
        return nn.Identity()
    if name == "prelu":
        return nn.PReLU()
    raise ValueError("No activation function with name: ", name)


def base_modules(module: nn.Module) -> list:
    """Return a list of all of the base modules in a network."""
    total = []
    children = list(module.children())
    if not children:
        total += [module]
    else:
        for c in children:
            total += base_modules(c)
    return total


def empty_0dim_like(inpt: T.Tensor | np.ndarray) -> T.Tensor | np.ndarray:
    """Return an empty tensor with same size as input but with final dim = 0."""

    # Get all but the final dimension
    all_but_last = inpt.shape[:-1]

    # Ensure that this is a tuple/list so it can agree with return syntax
    if isinstance(all_but_last, int):
        all_but_last = [all_but_last]

    if isinstance(inpt, T.Tensor):
        return T.empty((*all_but_last, 0), dtype=inpt.dtype, device=inpt.device)
    return np.empty((*all_but_last, 0))


def get_nrm(name: str, outp_dim: int) -> nn.Module:
    """Return a 1D pytorch normalisation layer given a name and a output size."""
    if name == "batch":
        return nn.BatchNorm1d(outp_dim)
    if name == "batch_nontr":
        return nn.BatchNorm1d(outp_dim, affine=False)
    if name == "batch_nontr_0":
        return nn.BatchNorm1d(outp_dim, affine=False, momentum=1)
    if name == "batch_nontr_99":
        return nn.BatchNorm1d(outp_dim, affine=False, momentum=0.01)
    if name == "batch_nontr_999":
        return nn.BatchNorm1d(outp_dim, affine=False, momentum=0.001)
    if name in ["lyr", "layer"]:
        return nn.LayerNorm(outp_dim)
    if name == "none":
        return None
    else:
        raise ValueError("No normalistation with name: ", name)


def get_loss_fn(name: Union[partial, str], **kwargs) -> nn.Module:
    """Return a pytorch loss function given a name."""

    # Supports using partial methods instad of having to do support each string
    if isinstance(name, partial):
        return name()

    if name == "none":
        return None

    # Classification losses
    if name == "crossentropy":
        return nn.CrossEntropyLoss(reduction="none")
    if name == "bcewithlogit":
        return MyBCEWithLogit(reduction="none")

    # Regression losses
    if name == "huber":
        return nn.HuberLoss(reduction="none")
    if name == "mse":
        return nn.MSELoss(reduction="none")
    if name == "mae":
        return nn.L1Loss(reduction="none")

    # Distribution matching losses
    if name == "champfer":
        return ChampferLoss()

    # Encoding losses
    if name == "vaeloss":
        return VAELoss()

    else:
        raise ValueError(f"No standard loss function with name: {name}")


def get_sched(
    sched_dict: Mapping,
    opt: Optimizer,
    steps_per_epoch: int = 0,
    max_lr: float | None = None,
    max_epochs: float | None = None,
) -> schd._LRScheduler:
    """Return a pytorch learning rate schedular given a dict containing a name and other
    kwargs.

    I still prefer this method as opposed to the hydra implementation as
    it allows us to specify the cyclical scheduler periods as a function of epochs
    rather than steps.

    Parameters
    ----------
    sched_dict : dict
        A dictionary of kwargs used to select and configure the scheduler.
    opt : Optimizer
        The optimizer to apply the learning rate to.
    steps_per_epoch : int
        The number of minibatches in a single training epoch.
    max_lr : float, optional
        The maximum learning rate for the one shot. Only for OneCycle learning.
    max_epochs : int, optional
        The maximum number of epochs to train for. Only for OneCycle learning.
    """

    # Pop off the name and learning rate for the optimizer
    dict_copy = sched_dict.copy()
    name = dict_copy.pop("name")

    # Get the max_lr from the optimizer if not specified
    max_lr = max_lr or opt.defaults["lr"]

    # Exit if the name indicates no scheduler
    if name in ["", "none", "None"]:
        return None

    # If the steps per epoch is 0, try and get it from the sched_dict
    if steps_per_epoch == 0:
        try:
            steps_per_epoch = dict_copy.pop("steps_per_epoch")
        except KeyError:
            raise ValueError(
                "steps_per_epoch was not passed to get_sched and was ",
                "not in the scheduler dictionary!",
            )

    # Pop off the number of epochs per cycle (needed as arg)
    if "epochs_per_cycle" in dict_copy:
        epochs_per_cycle = dict_copy.pop("epochs_per_cycle")
    else:
        epochs_per_cycle = 1

    # Use the same div_factor for cyclic with warmup
    if name == "cyclicwithwarmup":
        if "div_factor" not in dict_copy:
            dict_copy["div_factor"] = 1e4

    if name == "cosann":
        return schd.CosineAnnealingLR(
            opt, steps_per_epoch * epochs_per_cycle, **dict_copy
        )
    elif name == "cosannwr":
        return schd.CosineAnnealingWarmRestarts(
            opt, steps_per_epoch * epochs_per_cycle, **dict_copy
        )
    elif name == "onecycle":
        return schd.OneCycleLR(
            opt, max_lr, total_steps=steps_per_epoch * max_epochs, **dict_copy
        )
    elif name == "cyclicwithwarmup":
        return CyclicWithWarmup(
            opt, max_lr, total_steps=steps_per_epoch * epochs_per_cycle, **dict_copy
        )
    elif name == "linearwarmuprootdecay":
        return LinearWarmupRootDecay(opt, **dict_copy)
    elif name == "warmup":
        return WarmupToConstant(opt, **dict_copy)
    elif name == "lr_sheduler.ExponentialLR":
        return schd.ExponentialLR(opt, **dict_copy)
    elif name == "lr_scheduler.ConstantLR":
        return schd.ConstantLR(opt, **dict_copy)
    else:
        raise ValueError(f"No scheduler with name: {name}")


def train_valid_split(
    dataset: Dataset, v_frac: float, split_type="interweave"
) -> Tuple[Subset, Subset]:
    """Split a pytorch dataset into a training and validation pytorch Subsets.

    Parameters
    ----------
    dataset:
        The dataset to split
    v_frac:
        The validation fraction, reciprocals of whole numbers are best
    split_type: The type of splitting for the dataset. Default is interweave.
        basic: Take the first x event for the validation
        interweave: The every x events for the validation
        rand: Use a random splitting method (seed 42)
    """

    if split_type == "rand":
        v_size = int(v_frac * len(dataset))
        t_size = len(dataset) - v_size
        return random_split(
            dataset, [t_size, v_size], generator=T.Generator().manual_seed(42)
        )
    elif split_type == "basic":
        v_size = int(v_frac * len(dataset))
        valid_indxs = np.arange(0, v_size)
        train_indxs = np.arange(v_size, len(dataset))
        return Subset(dataset, train_indxs), Subset(dataset, valid_indxs)
    elif split_type == "interweave":
        v_every = int(1 / v_frac)
        valid_indxs = np.arange(0, len(dataset), v_every)
        train_indxs = np.delete(np.arange(len(dataset)), np.s_[::v_every])
        return Subset(dataset, train_indxs), Subset(dataset, valid_indxs)


def k_fold_split(
    dataset: Dataset, num_folds: int, fold_idx: int
) -> tuple[Subset, Subset, Subset]:
    """Perform a k-fold cross."""
    assert num_folds > 2
    assert fold_idx < num_folds

    test_fold = fold_idx
    val_fold = (fold_idx + 1) % num_folds
    train_folds = [i for i in range(num_folds) if i not in [fold_idx, val_fold]]

    data_idxes = np.arange(len(dataset))
    in_k = data_idxes % num_folds

    test = Subset(dataset, data_idxes[in_k == test_fold])
    valid = Subset(dataset, data_idxes[in_k == val_fold])
    train = Subset(dataset, data_idxes[np.isin(in_k, train_folds)])

    return train, valid, test


def masked_pool(
    pool_type: str, tensor: T.Tensor, mask: T.BoolTensor, axis: int | None = None
) -> T.Tensor:
    """Apply a pooling operation to masked elements of a tensor
    args:
        pool_type: Which pooling operation to use, currently supports max, sum and mean
        tensor: The input tensor to pool over
        mask: The mask to show which values should be included in the pool

    kwargs:
        axis: The axis to pool over, gets automatically from shape of mask

    """

    # Automatically get the pooling dimension from the shape of the mask
    if axis is None:
        axis = len(mask.shape) - 1

    # Or at least ensure that the axis is a positive number
    elif axis < 0:
        axis = len(tensor.shape) - axis

    # Apply the pooling method
    if pool_type == "max":
        tensor[~mask] = -T.inf
        return tensor.max(dim=axis)
    if pool_type == "sum":
        tensor[~mask] = 0
        return tensor.sum(dim=axis)
    if pool_type == "mean":
        tensor[~mask] = 0
        return tensor.sum(dim=axis) / (mask.sum(dim=axis, keepdim=True) + 1e-8)

    raise ValueError(f"Unknown pooling type: {pool_type}")


def smart_cat(inputs: Iterable, dim=-1) -> T.Tensor:
    """Concatenate without memory copy if tensors are are empty or None."""

    # Check number of non-empty tensors in the dimension for pooling
    n_nonempt = [0 if i is None else bool(i.size(dim=dim)) for i in inputs]

    # If there is only one non-empty tensor then we just return it directly
    if sum(n_nonempt) == 1:
        return inputs[np.argmax(n_nonempt)]

    # Otherwise concatenate the not None variables
    return T.cat([i for i in inputs if i is not None], dim=dim)


def sel_device(dev: str | T.device) -> T.device:
    """Return a pytorch device given a string (or a device)

    Passing cuda or gpu will run a hardware check first
    """
    # Not from config, but when device is specified already
    if isinstance(dev, T.device):
        return dev

    # Tries to get gpu if available
    if dev in ["cuda", "gpu"]:
        print("Trying to select cuda based on available hardware")
        dev = "cuda" if T.cuda.is_available() else "cpu"

    # Tries to get specific gpu
    elif "cuda" in dev:
        print(f"Trying to select {dev} based on available hardware")
        dev = dev if T.cuda.is_available() else "cpu"

    print(f"Running on hardware: {dev}")
    return T.device(dev)


def move_dev(
    tensor: T.Tensor | tuple | list | dict, dev: str | T.device
) -> T.Tensor | tuple | list | dict:
    """Return a copy of a tensor on the targetted device.

    This function calls pytorch's .to() but allows for values to be:
    - list of tensors
    - tuple of tensors
    - dict of tensors
    """

    # Select the pytorch device object if dev was a string
    if isinstance(dev, str):
        dev = sel_device(dev)

    if isinstance(tensor, tuple):
        return tuple(t.to(dev) for t in tensor)
    elif isinstance(tensor, list):
        return [t.to(dev) for t in tensor]
    elif isinstance(tensor, dict):
        return {t: tensor[t].to(dev) for t in tensor}
    else:
        return tensor.to(dev)


def to_np(inpt: Union[T.Tensor, tuple]) -> np.ndarray:
    """More consicse way of doing all the necc steps to convert a pytorch tensor to
    numpy array.

    - Includes gradient deletion, and device migration
    """
    if inpt is None:
        return None
    if isinstance(inpt, dict):
        return {k: to_np(inpt[k]) for k in inpt}
    if isinstance(inpt, (tuple, list)):
        return type(inpt)(to_np(x) for x in inpt)
    if inpt.dtype == T.bfloat16:  # Numpy conversions don't support bfloat16s
        inpt = inpt.half()
    return inpt.detach().cpu().numpy()


def print_gpu_info(dev=0):
    """Print the current gpu usage."""
    total = T.cuda.get_device_properties(dev).total_memory / 1024**3
    reser = T.cuda.memory_reserved(dev) / 1024**3
    alloc = T.cuda.memory_allocated(dev) / 1024**3
    print(f"\nTotal = {total:.2f}\nReser = {reser:.2f}\nAlloc = {alloc:.2f}")


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a pytorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_grad_norm(model: nn.Module, norm_type: float = 2.0):
    """Return the norm of the gradients of a given model."""
    return to_np(
        T.norm(
            T.stack([T.norm(p.grad.detach(), norm_type) for p in model.parameters()]),
            norm_type,
        )
    )


def reparam_trick(tensor: T.Tensor) -> Tuple[T.Tensor, T.Tensor, T.Tensor]:
    """Apply the reparameterisation trick to split a tensor into means and devs.

    - Returns a sample, the means and devs as a tuple
    - Splitting is along the final dimension
    - Used primarily in variational autoencoders
    """
    means, lstds = T.chunk(tensor, 2, dim=-1)
    latents = means + T.randn_like(means) * lstds.exp()
    return latents, means, lstds


def apply_residual(rsdl_type: str, res: T.Tensor, outp: T.Tensor) -> T.Tensor:
    """Apply a residual connection by either adding or concatenating."""
    if rsdl_type == "cat":
        return smart_cat([res, outp], dim=-1)
    if rsdl_type == "add":
        return outp + res
    raise ValueError(f"Unknown residual type: {rsdl_type}")


def aggr_via_sparse(compressed: T.Tensor, mask: T.BoolTensor, reduction: str, dim: int):
    """Aggregate a compressed tensor by first blowing up to a sparse representation.

    The tensor is blown up to full size such that: full[mask] = compressed
    This is suprisingly quick and the fastest method I have tested to do sparse
    aggregation in pytorch.
    I am sure that there might be a way to get this to work with gather though!

    Supports sum, mean, and softmax
    - mean is not supported by torch.sparse, so we use sum and the mask
    - softmax does not reduce the size of the tensor, but applies softmax along dim

    Parameters
    ----------
    cmprsed:
        The nonzero elements of the compressed tensor
    mask:
        A mask showing where the nonzero elements should go
    reduction:
        A string indicating the type of reduction
    dim:
        Which dimension to apply the reduction
    """

    # Create a sparse representation of the tensor
    sparse_rep = sparse_from_mask(compressed, mask, is_compressed=True)

    # Apply the reduction
    if reduction == "sum":
        return T.sparse.sum(sparse_rep, dim).values()
    if reduction == "mean":
        reduced = T.sparse.sum(sparse_rep, dim)
        mask_sum = mask.sum(dim)
        mask_sum = mask_sum.unsqueeze(-1).expand(reduced.shape)[mask_sum > 0]
        return reduced.values() / mask_sum
    if reduction == "softmax":
        return T.sparse.softmax(sparse_rep, dim).coalesce().values()
    raise ValueError(f"Unknown sparse reduction method: {reduction}")


def sparse_from_mask(inpt: T.Tensor, mask: T.BoolTensor, is_compressed: bool = False):
    """Create a pytorch sparse matrix given a tensor and a mask.

    - Shape is infered from the mask, meaning the final dim will be dense
    """
    return T.sparse_coo_tensor(
        T.nonzero(mask).t(),
        inpt if is_compressed else inpt[mask],
        size=(*mask.shape, inpt.shape[-1]),
        device=inpt.device,
        dtype=inpt.dtype,
        requires_grad=inpt.requires_grad,
    ).coalesce()


def decompress(compressed: T.Tensor, mask: T.BoolTensor) -> T.Tensor:
    """Take a compressed input and use the mask to blow it to its original shape.

    Ensures that: full[mask] = cmprsed.
    """
    # We first create the zero padded tensor of the right size then replace
    full = T.zeros(
        (*mask.shape, compressed.shape[-1]),
        dtype=compressed.dtype,
        device=compressed.device,
    )

    # Place the nonpadded samples into the full shape
    full[mask] = compressed

    return full


def get_max_cpu_suggest():
    """Try to compute a suggested max number of worker based on system's resource."""
    max_num_worker_suggest = None
    if hasattr(os, "sched_getaffinity"):
        try:
            max_num_worker_suggest = len(os.sched_getaffinity(0))
        except Exception:
            pass
    if max_num_worker_suggest is None:
        max_num_worker_suggest = os.cpu_count()
    return max_num_worker_suggest


def log_squash(data: T.Tensor) -> T.Tensor:
    """Apply a log squashing function for distributions with high tails."""
    return T.sign(data) * T.log(T.abs(data) + 1)


def torch_undo_log_squash(data: np.ndarray) -> np.ndarray:
    """Undo the log squash function above."""
    return T.sign(data) * (T.exp(T.abs(data)) - 1)


@T.no_grad()
def ema_param_sync(source: nn.Module, target: nn.Module, ema_decay: float) -> None:
    """Synchronize the parameters of two modules using exponential moving average (EMA).

    Parameters
    ----------
    source : nn.Module
        The source module whose parameters are used to update the target module.
    target : nn.Module
        The target module whose parameters are updated.
    ema_decay : float
        The decay rate for the EMA update.
    """
    for s_params, t_params in zip(source.parameters(), target.parameters()):
        t_params.data.copy_(
            ema_decay * t_params.data + (1.0 - ema_decay) * s_params.data
        )
