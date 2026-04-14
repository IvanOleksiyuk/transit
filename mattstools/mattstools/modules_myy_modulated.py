import torch as T
import torch.nn as nn

from .bayesian import BayesianLinear
from .torch_utils import get_act, get_nrm, masked_pool, smart_cat
from .modules_myy import MLPBlock, HalfActivatedMLPBlock


class _ShakeShakeMix(T.autograd.Function):
    """Shake-shake mix with independent forward and backward coefficients."""

    @staticmethod
    def forward(ctx, a: T.Tensor, b: T.Tensor, alpha: T.Tensor, beta: T.Tensor) -> T.Tensor:
        ctx.save_for_backward(beta)
        return alpha * a + (1.0 - alpha) * b

    @staticmethod
    def backward(ctx, grad_output: T.Tensor):
        (beta,) = ctx.saved_tensors
        grad_a = grad_output * beta
        grad_b = grad_output * (1.0 - beta)
        return grad_a, grad_b, None, None


class JustPropagateInput(nn.Module):
    """
    A simple module that just projects the input (ignoring context).
    Useful as a sanity check or ablation for context modules.
    """
    def __init__(self, 
                 inpt_dim: int,
                 ctxt_dim: int,
                 outp_dim: int):
        super().__init__()
        
        # Trainable parameter that does absolutely nothing
        self.dummy = nn.Parameter(T.zeros(1))

    def forward(self, x_inp, ctxt=None, mask=None):
        # Include dummy in computation graph without affecting output
        return x_inp + 0.0 * self.dummy

class DecoupledContextModule(nn.Module):
    """
    y = base
        + sum_{k=1..K} w_k(x_proj) * f_k(ctxt_proj)
        + w(x_proj) * ctxt_proj
        + b(x_proj)

    - f_k: ctxt-only MLPs, each outputs (..., outp_dim)
    - w_k: x-only gating weights, outputs (..., K) (one scalar per function)
           (you can switch to (..., K, outp_dim) if you want per-dim gating)
    - No mixing: x-only nets never take ctxt, ctxt-only nets never take x.

    base =
        x_proj                                   if preserve_x=False
        concat(x, x_proj[..., inpt_dim:])         if preserve_x=True AND outp_dim >= inpt_dim
        x_inp[..., :outp_dim]                     if preserve_x=True AND outp_dim < inpt_dim (truncate)

    mask is accepted but ignored (defaults to None).
    """

    def __init__(
        self,
        inpt_dim: int,
        ctxt_dim: int,
        outp_dim: int,
        hidden_dim: int = 128,
        num_funcs: int = 8,
        preserve_x: bool = False,
        per_dim_gating: bool = False,  # False: w_k is scalar per function; True: w_k is vector per function
    ):
        super().__init__()

        self.inpt_dim = inpt_dim
        self.ctxt_dim = ctxt_dim
        self.outp_dim = outp_dim
        self.num_funcs = num_funcs
        self.preserve_x = preserve_x
        self.per_dim_gating = per_dim_gating

        # Projections into common output space
        self.inp_proj = (
            nn.Identity()
            if inpt_dim == outp_dim
            else nn.Linear(inpt_dim, outp_dim, bias=False)
        )
        self.ctxt_proj = (
            nn.Identity()
            if ctxt_dim == outp_dim
            else nn.Linear(ctxt_dim, outp_dim, bias=False)
        )

        # ctxt-only experts f_k(ctxt_proj): K separate MLPs, each -> outp_dim
        self.f_funcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(outp_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, outp_dim),
            )
            for _ in range(num_funcs)
        ])

        # x-only gating weights w_k(x_proj)
        gate_out_dim = (num_funcs * outp_dim) if per_dim_gating else num_funcs
        self.w_gates = nn.Sequential(
            nn.Linear(outp_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, gate_out_dim),
        )

        # extra x-only terms
        self.w = nn.Sequential(
            nn.Linear(outp_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, outp_dim),
        )
        self.b = nn.Sequential(
            nn.Linear(outp_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, outp_dim),
        )

    def forward(
        self,
        x_inp: T.Tensor,
        ctxt: T.Tensor,
        mask: T.Tensor | None = None,
    ) -> T.Tensor:

        x_proj = self.inp_proj(x_inp)   # (..., outp_dim)
        c_proj = self.ctxt_proj(ctxt)   # (..., outp_dim)

        # Base term (preserve_x logic)
        if self.preserve_x:
            if self.outp_dim >= self.inpt_dim:
                base = T.cat([x_inp, x_proj[..., self.inpt_dim:]], dim=-1)
            else:
                base = x_inp[..., : self.outp_dim]
        else:
            base = x_proj

        # Compute f_k(ctxt_proj) and stack -> (..., K, outp_dim)
        f_stack = T.stack([f(c_proj) for f in self.f_funcs], dim=-2)

        # Compute gates from x only
        gates = self.w_gates(x_proj)
        if self.per_dim_gating:
            # (..., K*outp_dim) -> (..., K, outp_dim)
            gates = gates.view(*x_proj.shape[:-1], self.num_funcs, self.outp_dim)
        else:
            # (..., K) -> (..., K, 1) for broadcasting over outp_dim
            gates = gates.view(*x_proj.shape[:-1], self.num_funcs, 1)

        # Mixture term: sum_k w_k(x) * f_k(ctxt)
        mix = (gates * f_stack).sum(dim=-2)  # (..., outp_dim)

        w = self.w(x_proj)                   # (..., outp_dim)
        b = self.b(x_proj)                   # (..., outp_dim)

        return base + mix + w * c_proj + b

class ContextLineHyper(nn.Module):
    """
    y = base + w(x,ctxt) * ctxt_proj + b(x,ctxt)

    base =
        x_proj                                   if preserve_x=False
        concat(x, x_proj[..., inpt_dim:])         if preserve_x=True AND outp_dim >= inpt_dim
        x_inp[..., :outp_dim]                     if preserve_x=True AND outp_dim < inpt_dim (truncate)

    mask is accepted but ignored (defaults to None).
    """

    def __init__(
        self,
        inpt_dim: int,
        ctxt_dim: int,
        outp_dim: int,
        hidden_dim: int = 128,
        preserve_x: bool = False,
    ):
        super().__init__()

        self.inpt_dim = inpt_dim
        self.ctxt_dim = ctxt_dim
        self.outp_dim = outp_dim
        self.preserve_x = preserve_x

        # Input projection
        self.inp_proj = (
            nn.Identity()
            if inpt_dim == outp_dim
            else nn.Linear(inpt_dim, outp_dim, bias=False)
        )

        # Context projection
        self.ctxt_proj = nn.Linear(ctxt_dim, outp_dim, bias=False)

        # Smooth hypernetwork
        self.hyper = nn.Sequential(
            nn.Linear(outp_dim + ctxt_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * outp_dim),
        )

    def forward(
        self,
        x_inp: T.Tensor,
        ctxt: T.Tensor,
        mask: T.Tensor | None = None,
    ) -> T.Tensor:

        x_proj = self.inp_proj(x_inp)   # (..., outp_dim)
        c_proj = self.ctxt_proj(ctxt)   # (..., outp_dim)

        # Base term
        if self.preserve_x:
            if self.outp_dim >= self.inpt_dim:
                # preserve full x_inp, fill remaining dims from x_proj
                base = T.cat(
                    [x_inp, x_proj[..., self.inpt_dim:]],
                    dim=-1,
                )
            else:
                # out smaller than input: preserve by truncating x_inp
                base = x_inp[..., : self.outp_dim]
        else:
            base = x_proj

        # Hypernetwork input (always uses projected x so dims are stable)
        h = T.cat([x_proj, ctxt], dim=-1)
        wb = self.hyper(h)
        w, b = wb.chunk(2, dim=-1)

        return base + w * c_proj + b

class SimpleContextModule(nn.Module):
    def __init__(self, inpt_dim, ctxt_dim, outp_dim):
        super().__init__()
        
        # Linear projection for context (trainable weight)
        self.ctxt_weight = nn.Parameter(
            T.randn(ctxt_dim, outp_dim)/10
        )
        
        # Trainable bias
        self.bias = nn.Parameter(
            T.randn(outp_dim)/10
        )
        
        # Optional: project input if dimensions differ
        if inpt_dim != outp_dim:
            self.inp_proj = nn.Linear(inpt_dim, outp_dim, bias=False)
        else:
            self.inp_proj = None

    def forward(self, x_inp, ctxt, mask=None):
        # Ignore mask for now
        
        if self.inp_proj is not None:
            x_inp = self.inp_proj(x_inp)
        
        # ctxt @ weight
        ctxt_term = ctxt @ self.ctxt_weight
        
        # Add everything
        out = x_inp + ctxt_term + self.bias
        
        return out

class SelfModulatedFeatureLayer(nn.Module):
    """
    Dimension-preserving modulation:
        y = x + v(x,ctxt) * gate(x,ctxt) + beta(x,ctxt)

    v, gate, beta are small MLPs mapping R^(d+ctxt_dim)->R^d
    """
    def __init__(
        self,
        d: int,
        ctxt_dim: int = 0,
        hidden: int = 32,
        bias_hidden: int = 16,
        n_layers: int = 2,
        act: str = "silu",
        nrm: str = "none",
        drp: float = 0.0,
        gate: str = "sigmoid",
        init_zeros: bool = False,
        use_bias: bool = True,
    ):
        super().__init__()
        self.d = d
        self.ctxt_dim = ctxt_dim
        self.gate = gate

        # Use your HalfActivatedMLPBlock so activations/norm/dropout match the rest of the codebase
        self.v_net = HalfActivatedMLPBlock(
            inpt_dim=d,
            outp_dim=d,
            hidden_dim=hidden,
            ctxt_dim=ctxt_dim,
            n_layers=n_layers,
            act=act,
            nrm=nrm,
            drp=drp,
            do_res=False,
            init_zeros=init_zeros,
            use_bias=use_bias,
        )

        self.q_net = HalfActivatedMLPBlock(
            inpt_dim=d,
            outp_dim=d,
            hidden_dim=hidden,
            ctxt_dim=ctxt_dim,
            n_layers=n_layers,
            act=act,
            nrm=nrm,
            drp=drp,
            do_res=False,
            init_zeros=init_zeros,
            use_bias=use_bias,
        )

        # Small bias MLP (intentionally smaller capacity)
        self.beta_net = HalfActivatedMLPBlock(
            inpt_dim=d,
            outp_dim=d,
            hidden_dim=bias_hidden,
            ctxt_dim=ctxt_dim,
            n_layers=n_layers,
            act=act,
            nrm=nrm,
            drp=drp,
            do_res=False,
            init_zeros=init_zeros,
            use_bias=use_bias,
            # If you want to enforce "small", pass smaller hidden via DenseNetwork instead;
            # HalfActivatedMLPBlock width is fixed to outp_dim, so we keep it shallow here.
        )

        # If you really want a smaller hidden for beta, replace beta_net with a DenseNetwork
        # configured with a smaller hddn_dim. Kept simple/consistent here.

    def _apply_gate(self, logits: T.Tensor) -> T.Tensor:
        if self.gate == "sigmoid":
            return T.sigmoid(logits)
        if self.gate == "tanh":
            return T.tanh(logits)
        if self.gate == "softplus":
            return T.nn.functional.softplus(logits)
        raise ValueError(f"Unknown gate type: {self.gate}")

    def forward(self, x: T.Tensor, ctxt: T.Tensor | None = None) -> T.Tensor:
        v = self.v_net(x, ctxt)
        g = self._apply_gate(self.q_net(x, ctxt))
        beta = self.beta_net(x, ctxt)
        return x + v * g + beta


class MultiHeadContextOnlyVBetaModulatedFeatureLayer(nn.Module):
    """
    Multi-head context-only modulation:
                y = x + sum_h gate_h(x[,ctxt]) * value_h(ctxt) + beta(ctxt)

        - All gate heads are produced by one MLP that always uses x and can
            optionally also use ctxt.
        - All value heads are produced by another context-only MLP.
    - The head aggregation uses a dot product over the head axis.
    """

    def __init__(
        self,
        d: int,
        ctxt_dim: int,
        num_heads: int = 4,
        hidden: int = None,
        bias_hidden: int = 16,
        n_layers: int = 2,
        act: str = "silu",
        nrm: str = "none",
        drp: float = 0.0,
        gate: str = "tanh",
        gate_uses_ctxt: bool = True,
        init_zeros: bool = False,
        use_bias: bool = True,
    ):
        super().__init__()

        if ctxt_dim <= 0:
            raise ValueError("MultiHeadContextOnlyVBetaModulatedFeatureLayer requires ctxt_dim > 0")
        if num_heads < 0:
            raise ValueError("num_heads must be >= 0")

        self.d = d
        self.ctxt_dim = ctxt_dim
        self.num_heads = num_heads
        self.gate = gate
        self.gate_uses_ctxt = gate_uses_ctxt
        self.hidden = hidden
        self.bias_hidden = bias_hidden

        # Context-only branches for gate/value heads. When num_heads == 0,
        # the head modulation path is disabled and only beta(ctxt) is used.
        if self.num_heads > 0:
            self.gate_net = HalfActivatedMLPBlock(
                inpt_dim=d,
                outp_dim=num_heads * d,
                hidden_dim=hidden,
                ctxt_dim=ctxt_dim if self.gate_uses_ctxt else 0,
                n_layers=n_layers,
                act=act,
                nrm=nrm,
                drp=drp,
                do_res=False,
                init_zeros=init_zeros,
                use_bias=use_bias,
            )

            self.value_net = HalfActivatedMLPBlock(
                inpt_dim=ctxt_dim,
                outp_dim=num_heads * d,
                hidden_dim=hidden,
                ctxt_dim=0,
                n_layers=n_layers,
                act=act,
                nrm=nrm,
                drp=drp,
                do_res=False,
                init_zeros=init_zeros,
                use_bias=use_bias,
            )
        else:
            self.gate_net = None
            self.value_net = None

        # Context-only bias branch beta(ctxt).
        self.beta_net = HalfActivatedMLPBlock(
            inpt_dim=ctxt_dim,
            outp_dim=d,
            hidden_dim=bias_hidden,
            ctxt_dim=0,
            n_layers=n_layers,
            act=act,
            nrm=nrm,
            drp=drp,
            do_res=False,
            init_zeros=init_zeros,
            use_bias=use_bias,
        )

    def _apply_gate(self, logits: T.Tensor) -> T.Tensor:
        if self.gate == "sigmoid":
            return T.sigmoid(logits)
        if self.gate == "tanh":
            return T.tanh(logits)
        if self.gate == "softplus":
            return T.nn.functional.softplus(logits)
        if self.gate == "softmax":
            return T.softmax(logits, dim=-2)
        raise ValueError(f"Unknown gate type: {self.gate}")

    def forward(self, x: T.Tensor, ctxt: T.Tensor | None = None) -> T.Tensor:
        if ctxt is None:
            raise ValueError("ctxt must be provided for MultiHeadContextOnlyVBetaModulatedFeatureLayer")

        if self.num_heads > 0:
            gate_ctxt = ctxt if self.gate_uses_ctxt else None
            gate_logits = self.gate_net(x, gate_ctxt).view(*x.shape[:-1], self.num_heads, self.d)
            values = self.value_net(ctxt).view(*ctxt.shape[:-1], self.num_heads, self.d)

            gates = self._apply_gate(gate_logits)
            delta = T.einsum("...hd,...hd->...d", gates, values)
        else:
            delta = T.zeros_like(x)

        beta = self.beta_net(ctxt)
        return x + delta + beta


class SingleLayerMultiHeadContextOnlyVBetaNetwork(nn.Module):
    """A minimal network made of exactly one multi-head context-only layer.

    Input and output feature dimensions are identical.
    """

    def __init__(
        self,
        inpt_dim: int,
        ctxt_dim: int,
        outp_dim: int = 0,
        num_heads: int = 4,
        hidden: int = None,
        bias_hidden: int = 32,
        n_layers: int = 3,
        act: str = "silu",
        nrm: str = "none",
        drp: float = 0.0,
        gate: str = "tanh",
        gate_uses_ctxt: bool = True,
        init_zeros: bool = False,
        use_bias: bool = True,
    ):
        super().__init__()

        if outp_dim not in (0, inpt_dim):
            raise ValueError(
                "SingleLayerMultiHeadContextOnlyVBetaNetwork enforces outp_dim == inpt_dim. "
                f"Got inpt_dim={inpt_dim}, outp_dim={outp_dim}."
            )

        self.inpt_dim = inpt_dim
        self.outp_dim = inpt_dim
        self.ctxt_dim = ctxt_dim

        self.layer = MultiHeadContextOnlyVBetaModulatedFeatureLayer(
            d=inpt_dim,
            ctxt_dim=ctxt_dim,
            num_heads=num_heads,
            hidden=hidden,
            bias_hidden=bias_hidden,
            n_layers=n_layers,
            act=act,
            nrm=nrm,
            drp=drp,
            gate=gate,
            gate_uses_ctxt=gate_uses_ctxt,
            init_zeros=init_zeros,
            use_bias=use_bias,
        )

    def forward(self, inputs: T.Tensor, ctxt: T.Tensor | None = None) -> T.Tensor:
        if ctxt is None:
            raise ValueError("ctxt must be provided for SingleLayerMultiHeadContextOnlyVBetaNetwork")

        # Broadcast ctxt like the other network wrappers when rank differs.
        dim_diff = inputs.dim() - ctxt.dim()
        if dim_diff > 0:
            ctxt = ctxt.view(ctxt.shape[0], *dim_diff * (1,), *ctxt.shape[1:])
            ctxt = ctxt.expand(*inputs.shape[:-1], -1)

        return self.layer(inputs, ctxt)

    def __repr__(self):
        return f"SingleLayerMultiHeadContextOnlyVBetaNetwork({self.inpt_dim}->{self.outp_dim})"


class ShakeShakeSingleLayerMultiHeadContextOnlyVBetaNetwork(nn.Module):
    """A single-layer multi-head network with N-way shake-shake regularization.

    Each branch is one `MultiHeadContextOnlyVBetaModulatedFeatureLayer`.
    During training, branch outputs are mixed with independent forward/backward
    random convex coefficients. During evaluation, outputs are averaged.
    """

    def __init__(
        self,
        inpt_dim: int,
        ctxt_dim: int,
        outp_dim: int = 0,
        num_heads: int = 4,
        num_branches: int = 5,
        hidden: int = None,
        bias_hidden: int = 32,
        n_layers: int = 3,
        act: str = "silu",
        nrm: str = "none",
        drp: float = 0.0,
        gate: str = "tanh",
        gate_uses_ctxt: bool = True,
        init_zeros: bool = False,
        use_bias: bool = True,
    ):
        super().__init__()

        if outp_dim not in (0, inpt_dim):
            raise ValueError(
                "ShakeShakeSingleLayerMultiHeadContextOnlyVBetaNetwork enforces outp_dim == inpt_dim. "
                f"Got inpt_dim={inpt_dim}, outp_dim={outp_dim}."
            )
        if num_branches < 1:
            raise ValueError("num_branches must be >= 1")

        self.inpt_dim = inpt_dim
        self.outp_dim = inpt_dim
        self.ctxt_dim = ctxt_dim
        self.num_branches = num_branches

        self.branches = nn.ModuleList(
            [
                MultiHeadContextOnlyVBetaModulatedFeatureLayer(
                    d=inpt_dim,
                    ctxt_dim=ctxt_dim,
                    num_heads=num_heads,
                    hidden=hidden,
                    bias_hidden=bias_hidden,
                    n_layers=n_layers,
                    act=act,
                    nrm=nrm,
                    drp=drp,
                    gate=gate,
                    gate_uses_ctxt=gate_uses_ctxt,
                    init_zeros=init_zeros,
                    use_bias=use_bias,
                )
                for _ in range(num_branches)
            ]
        )

    def _shake_shake(self, branch_outputs: list[T.Tensor]) -> T.Tensor:
        if len(branch_outputs) == 1:
            return branch_outputs[0]

        if not self.training:
            return T.stack(branch_outputs, dim=0).mean(dim=0)

        # Per-sample random convex coefficients for forward and backward paths.
        ref = branch_outputs[0]
        coeff_shape = [ref.shape[0]] + [1] * (ref.dim() - 1) if ref.dim() > 1 else [1]

        alpha = T.rand((len(branch_outputs), *coeff_shape), device=ref.device, dtype=ref.dtype)
        alpha = alpha / alpha.sum(dim=0, keepdim=True).clamp_min(1e-12)

        beta = T.rand((len(branch_outputs), *coeff_shape), device=ref.device, dtype=ref.dtype)
        beta = beta / beta.sum(dim=0, keepdim=True).clamp_min(1e-12)

        forward_mix = sum(alpha[i] * branch_outputs[i] for i in range(len(branch_outputs)))
        backward_mix = sum(beta[i] * branch_outputs[i] for i in range(len(branch_outputs)))

        # Forward uses alpha-mix, backward uses beta-mix.
        return backward_mix + (forward_mix - backward_mix).detach()

    def forward(self, inputs: T.Tensor, ctxt: T.Tensor | None = None) -> T.Tensor:
        if ctxt is None:
            raise ValueError("ctxt must be provided for ShakeShakeSingleLayerMultiHeadContextOnlyVBetaNetwork")

        # Broadcast ctxt like the other network wrappers when rank differs.
        dim_diff = inputs.dim() - ctxt.dim()
        if dim_diff > 0:
            ctxt = ctxt.view(ctxt.shape[0], *dim_diff * (1,), *ctxt.shape[1:])
            ctxt = ctxt.expand(*inputs.shape[:-1], -1)

        branch_outputs = [branch(inputs, ctxt) for branch in self.branches]
        return self._shake_shake(branch_outputs)

    def __repr__(self):
        return (
            "ShakeShakeSingleLayerMultiHeadContextOnlyVBetaNetwork("
            f"{self.inpt_dim}->{self.outp_dim}, branches={self.num_branches})"
        )


class SharedWeightShakeShakeSingleLayerMultiHeadContextOnlyVBetaNetwork(
    ShakeShakeSingleLayerMultiHeadContextOnlyVBetaNetwork
):
    """Same as shake-shake single-layer multi-head net, but with shared fwd/bwd weights.

    During training, one random convex coefficient vector is sampled per batch
    item and used identically in forward and backward. During eval, outputs are
    averaged over branches.
    """

    def _shake_shake(self, branch_outputs: list[T.Tensor]) -> T.Tensor:
        if len(branch_outputs) == 1:
            return branch_outputs[0]

        if not self.training:
            return T.stack(branch_outputs, dim=0).mean(dim=0)

        ref = branch_outputs[0]
        coeff_shape = [ref.shape[0]] + [1] * (ref.dim() - 1) if ref.dim() > 1 else [1]

        coeff = T.rand((len(branch_outputs), *coeff_shape), device=ref.device, dtype=ref.dtype)
        coeff = coeff / coeff.sum(dim=0, keepdim=True).clamp_min(1e-12)

        # Same coefficients are used for both forward and backward passes.
        return sum(coeff[i] * branch_outputs[i] for i in range(len(branch_outputs)))


class _HalfActivatedMLPBranch(nn.Module):
    """One branch used inside shake-shake half-activated MLP networks."""

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int,
        ctxt_dim: int,
        hddn_dim: list[int],
        n_lyr_pbk: int,
        act_h: str,
        act_o: str,
        do_out: bool,
        nrm: str,
        drp: float,
        nrm_inside: bool,
        drp_on_output: bool,
        nrm_on_output: bool,
        do_res: bool,
        ctxt_in_inpt: bool,
        ctxt_in_hddn: bool,
        ctxt_in_out: bool,
        init_zeros: bool,
        do_bayesian: bool,
        use_bias: bool,
        apply_act_o_on_last_layer: bool = False,
    ) -> None:
        super().__init__()

        self.do_out = do_out

        self.input_block = HalfActivatedMLPBlock(
            inpt_dim=inpt_dim,
            outp_dim=hddn_dim[0],
            ctxt_dim=ctxt_dim if ctxt_in_inpt else 0,
            n_layers=1,
            act=act_h,
            nrm=nrm if nrm_inside else "none",
            drp=drp,
            do_bayesian=do_bayesian,
            use_bias=use_bias,
        )

        self.hidden_blocks = nn.ModuleList()
        if len(hddn_dim) > 1:
            for h_1, h_2 in zip(hddn_dim[:-1], hddn_dim[1:]):
                self.hidden_blocks.append(
                    HalfActivatedMLPBlock(
                        inpt_dim=h_1,
                        outp_dim=h_2,
                        ctxt_dim=ctxt_dim if ctxt_in_hddn else 0,
                        n_layers=n_lyr_pbk,
                        act=act_h,
                        nrm=nrm if nrm_inside else "none",
                        drp=drp,
                        do_res=do_res,
                        init_zeros=init_zeros,
                        do_bayesian=do_bayesian,
                        use_bias=use_bias,
                    )
                )

        if self.do_out:
            self.output_block = HalfActivatedMLPBlock(
                inpt_dim=hddn_dim[-1],
                outp_dim=outp_dim,
                ctxt_dim=ctxt_dim if ctxt_in_out else 0,
                n_layers=1,
                act=act_o,
                do_bayesian=do_bayesian,
                init_zeros=init_zeros,
                nrm=nrm if nrm_on_output else "none",
                drp=drp if drp_on_output else 0,
                use_bias=use_bias,
                apply_act_on_last_layer=apply_act_o_on_last_layer,
            )

    def forward(self, x: T.Tensor, ctxt: T.Tensor | None = None) -> T.Tensor:
        x = self.input_block(x, ctxt)
        for h_block in self.hidden_blocks:
            x = h_block(x, ctxt)
        if self.do_out:
            x = self.output_block(x, ctxt)
        return x


class ShakeShakeHalfActivatedMLP(nn.Module):
    """Half-activated MLP with shake-shake regularization across branches.

    This class is designed as a drop-in discriminator replacement for
    `transit.mltools.mlp.MLP` configs while using `HalfActivatedMLPBlock`
    internals and N-way shake-shake branch mixing.
    """

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int = 0,
        ctxt_dim: int = 0,
        hddn_dim: int | list = 32,
        num_blocks: int = 1,
        n_lyr_pbk: int = 1,
        act_h: str = "silu",
        act_o: str = "sigmoid",
        do_out: bool = True,
        nrm: str = "none",
        drp: float = 0,
        nrm_inside: bool = True,
        drp_on_output: bool = False,
        nrm_on_output: bool = False,
        do_res: bool = False,
        ctxt_in_inpt: bool = True,
        ctxt_in_hddn: bool = False,
        ctxt_in_out: bool = False,
        do_bayesian: bool = False,
        init_zeros: bool = False,
        use_bias: bool = True,
        num_branches: int = 5,
        shared_forward_backward_weights: bool = True,
        apply_act_o_on_last_layer: bool = True,
    ) -> None:
        super().__init__()

        if ctxt_dim and not (ctxt_in_inpt or ctxt_in_hddn or ctxt_in_out):
            raise ValueError("Network has context inputs but nowhere to use them!")
        if num_branches < 1:
            raise ValueError("num_branches must be >= 1")

        self.inpt_dim = inpt_dim
        if not isinstance(hddn_dim, int):
            self.hddn_dim = hddn_dim
        else:
            self.hddn_dim = num_blocks * [hddn_dim]

        self.outp_dim = outp_dim or inpt_dim if do_out else self.hddn_dim[-1]
        self.num_blocks = len(self.hddn_dim)
        self.ctxt_dim = ctxt_dim
        self.do_out = do_out
        self.num_branches = num_branches
        self.shared_forward_backward_weights = shared_forward_backward_weights

        # nflows compatibility like other MLPs.
        self.hidden_features = self.hddn_dim[-1]

        self.branches = nn.ModuleList(
            [
                _HalfActivatedMLPBranch(
                    inpt_dim=self.inpt_dim,
                    outp_dim=self.outp_dim,
                    ctxt_dim=self.ctxt_dim,
                    hddn_dim=self.hddn_dim,
                    n_lyr_pbk=n_lyr_pbk,
                    act_h=act_h,
                    act_o=act_o,
                    do_out=self.do_out,
                    nrm=nrm,
                    drp=drp,
                    nrm_inside=nrm_inside,
                    drp_on_output=drp_on_output,
                    nrm_on_output=nrm_on_output,
                    do_res=do_res,
                    ctxt_in_inpt=ctxt_in_inpt,
                    ctxt_in_hddn=ctxt_in_hddn,
                    ctxt_in_out=ctxt_in_out,
                    init_zeros=init_zeros,
                    do_bayesian=do_bayesian,
                    use_bias=use_bias,
                    apply_act_o_on_last_layer=apply_act_o_on_last_layer,
                )
                for _ in range(self.num_branches)
            ]
        )

    def _shake_shake(self, branch_outputs: list[T.Tensor]) -> T.Tensor:
        if len(branch_outputs) == 1:
            return branch_outputs[0]

        if not self.training:
            return T.stack(branch_outputs, dim=0).mean(dim=0)

        ref = branch_outputs[0]
        coeff_shape = [ref.shape[0]] + [1] * (ref.dim() - 1) if ref.dim() > 1 else [1]

        alpha = T.rand((len(branch_outputs), *coeff_shape), device=ref.device, dtype=ref.dtype)
        alpha = alpha / alpha.sum(dim=0, keepdim=True).clamp_min(1e-12)

        if self.shared_forward_backward_weights:
            return sum(alpha[i] * branch_outputs[i] for i in range(len(branch_outputs)))

        beta = T.rand((len(branch_outputs), *coeff_shape), device=ref.device, dtype=ref.dtype)
        beta = beta / beta.sum(dim=0, keepdim=True).clamp_min(1e-12)

        forward_mix = sum(alpha[i] * branch_outputs[i] for i in range(len(branch_outputs)))
        backward_mix = sum(beta[i] * branch_outputs[i] for i in range(len(branch_outputs)))
        return backward_mix + (forward_mix - backward_mix).detach()

    def forward(
        self,
        inputs: T.Tensor,
        ctxt: T.Tensor | None = None,
        context: T.Tensor | None = None,
    ) -> T.Tensor:
        # Use context as a synonym for ctxt (normflow compatibility).
        if context is not None:
            ctxt = context

        if ctxt is not None:
            dim_diff = inputs.dim() - ctxt.dim()
            if dim_diff > 0:
                ctxt = ctxt.view(ctxt.shape[0], *dim_diff * (1,), *ctxt.shape[1:])
                ctxt = ctxt.expand(*inputs.shape[:-1], -1)

        outputs = [branch(inputs, ctxt) for branch in self.branches]
        return self._shake_shake(outputs)

    def __repr__(self):
        return (
            "ShakeShakeHalfActivatedMLP("
            f"{self.inpt_dim}->{self.outp_dim}, blocks={self.num_blocks}, "
            f"branches={self.num_branches})"
        )


class ShakeShakeContextOnlyVBetaModulatedFeatureLayer(nn.Module):
    """
    Context-only v/beta modulation with shake-shake regularization:
        y = x + shake(delta_1, delta_2)

    where
        delta_i = v_i(ctxt) * gate_i(x,ctxt) + beta_i(ctxt)

    - v_i and beta_i depend only on ctxt
    - gate_i can optionally use ctxt via gate_uses_ctxt
    - During eval, shake reduces to the average of both branches
    """

    def __init__(
        self,
        d: int,
        ctxt_dim: int,
        hidden: int = 32,
        bias_hidden: int = 16,
        n_layers: int = 2,
        act: str = "silu",
        nrm: str = "none",
        drp: float = 0.0,
        gate: str = "sigmoid",
        gate_uses_ctxt: bool = True,
        init_zeros: bool = False,
        use_bias: bool = True,
    ):
        super().__init__()

        if ctxt_dim <= 0:
            raise ValueError("ShakeShakeContextOnlyVBetaModulatedFeatureLayer requires ctxt_dim > 0")

        self.d = d
        self.ctxt_dim = ctxt_dim
        self.gate = gate
        self.gate_uses_ctxt = gate_uses_ctxt

        # Branch 1
        self.v_net_1 = HalfActivatedMLPBlock(
            inpt_dim=ctxt_dim,
            outp_dim=d,
            hidden_dim=hidden,
            ctxt_dim=0,
            n_layers=n_layers,
            act=act,
            nrm=nrm,
            drp=drp,
            do_res=False,
            init_zeros=init_zeros,
            use_bias=use_bias,
        )
        self.q_net_1 = HalfActivatedMLPBlock(
            inpt_dim=d,
            outp_dim=d,
            hidden_dim=hidden,
            ctxt_dim=ctxt_dim if gate_uses_ctxt else 0,
            n_layers=n_layers,
            act=act,
            nrm=nrm,
            drp=drp,
            do_res=False,
            init_zeros=init_zeros,
            use_bias=use_bias,
        )
        self.beta_net_1 = HalfActivatedMLPBlock(
            inpt_dim=ctxt_dim,
            outp_dim=d,
            hidden_dim=bias_hidden,
            ctxt_dim=0,
            n_layers=n_layers,
            act=act,
            nrm=nrm,
            drp=drp,
            do_res=False,
            init_zeros=init_zeros,
            use_bias=use_bias,
        )

        # Branch 2
        self.v_net_2 = HalfActivatedMLPBlock(
            inpt_dim=ctxt_dim,
            outp_dim=d,
            hidden_dim=hidden,
            ctxt_dim=0,
            n_layers=n_layers,
            act=act,
            nrm=nrm,
            drp=drp,
            do_res=False,
            init_zeros=init_zeros,
            use_bias=use_bias,
        )
        self.q_net_2 = HalfActivatedMLPBlock(
            inpt_dim=d,
            outp_dim=d,
            hidden_dim=hidden,
            ctxt_dim=ctxt_dim if gate_uses_ctxt else 0,
            n_layers=n_layers,
            act=act,
            nrm=nrm,
            drp=drp,
            do_res=False,
            init_zeros=init_zeros,
            use_bias=use_bias,
        )
        self.beta_net_2 = HalfActivatedMLPBlock(
            inpt_dim=ctxt_dim,
            outp_dim=d,
            hidden_dim=bias_hidden,
            ctxt_dim=0,
            n_layers=n_layers,
            act=act,
            nrm=nrm,
            drp=drp,
            do_res=False,
            init_zeros=init_zeros,
            use_bias=use_bias,
        )

    def _apply_gate(self, logits: T.Tensor) -> T.Tensor:
        if self.gate == "sigmoid":
            return T.sigmoid(logits)
        if self.gate == "tanh":
            return T.tanh(logits)
        if self.gate == "softplus":
            return T.nn.functional.softplus(logits)
        raise ValueError(f"Unknown gate type: {self.gate}")

    def _shake_shake(self, a: T.Tensor, b: T.Tensor) -> T.Tensor:
        if not self.training:
            return 0.5 * (a + b)
        coeff_shape = [a.shape[0]] + [1] * (a.dim() - 1) if a.dim() > 1 else [1]
        alpha = T.rand(coeff_shape, device=a.device, dtype=a.dtype)
        beta = T.rand(coeff_shape, device=a.device, dtype=a.dtype)
        return _ShakeShakeMix.apply(a, b, alpha, beta)

    def forward(self, x: T.Tensor, ctxt: T.Tensor | None = None) -> T.Tensor:
        if ctxt is None:
            raise ValueError("ctxt must be provided for ShakeShakeContextOnlyVBetaModulatedFeatureLayer")

        gate_ctxt = ctxt if self.gate_uses_ctxt else None

        v_1 = self.v_net_1(ctxt)
        g_1 = self._apply_gate(self.q_net_1(x, gate_ctxt))
        beta_1 = self.beta_net_1(ctxt)
        delta_1 = v_1 * g_1 + beta_1

        v_2 = self.v_net_2(ctxt)
        g_2 = self._apply_gate(self.q_net_2(x, gate_ctxt))
        beta_2 = self.beta_net_2(ctxt)
        delta_2 = v_2 * g_2 + beta_2

        return x + self._shake_shake(delta_1, delta_2)


class ModulatedBlock(nn.Module):
    """
    A residual block of the form:
        pre_dense:  x -> h
        mod:        h -> h   (SelfModulatedFeatureLayer)
        post_dense: h -> y
        residual:   y += x (if dims match, or "adjust" like your HalfActivatedMLPBlock)
    """
    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int,
        ctxt_dim: int = 0,
        pre_layers: int = 1,
        post_layers: int = 1,
        act: str = "silu",
        nrm: str = "none",
        drp: float = 0.0,
        do_res: bool | str = False,   # False | True | "adjust"
        use_bias: bool = True,
        mod_hidden: int = 32,
        mod_bias_hidden: int = 16,
        mod_n_layers: int = 2,
        gate: str = "sigmoid",
        context_only_vbeta: bool = False,
        use_multi_head_context_only_vbeta: bool = False,
        multi_head_num_heads: int = 4,
        use_shake_shake: bool = False,
        gate_uses_ctxt: bool = True,
        init_zeros_post: bool = False,
    ):
        super().__init__()
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim

        # Residual behavior: mimic HalfActivatedMLPBlock semantics
        if do_res == "adjust":
            self.do_res = "adjust"
        else:
            self.do_res = bool(do_res) and (inpt_dim == outp_dim)

        self.pre = HalfActivatedMLPBlock(
            inpt_dim=inpt_dim,
            outp_dim=outp_dim,          # keep it simple: pre brings to block width
            ctxt_dim=ctxt_dim,
            n_layers=pre_layers,
            act=act,
            nrm=nrm,
            drp=drp,
            do_res=False,
            use_bias=use_bias,
        )

        if context_only_vbeta:
            if ctxt_dim <= 0:
                raise ValueError("context_only_vbeta=True requires ctxt_dim > 0 in ModulatedBlock")

            if use_multi_head_context_only_vbeta and use_shake_shake:
                raise ValueError(
                    "use_multi_head_context_only_vbeta=True is incompatible with use_shake_shake=True"
                )

            if use_multi_head_context_only_vbeta:
                self.mod = MultiHeadContextOnlyVBetaModulatedFeatureLayer(
                    d=outp_dim,
                    ctxt_dim=ctxt_dim,
                    num_heads=multi_head_num_heads,
                    hidden=mod_hidden,
                    bias_hidden=mod_bias_hidden,
                    n_layers=mod_n_layers,
                    act=act,
                    nrm=nrm,
                    drp=drp,
                    gate=gate,
                    gate_uses_ctxt=gate_uses_ctxt,
                    init_zeros=False,
                    use_bias=use_bias,
                )
            elif use_shake_shake:
                self.mod = ShakeShakeContextOnlyVBetaModulatedFeatureLayer(
                    d=outp_dim,
                    ctxt_dim=ctxt_dim,
                    hidden=mod_hidden,
                    bias_hidden=mod_bias_hidden,
                    n_layers=mod_n_layers,
                    act=act,
                    nrm=nrm,
                    drp=drp,
                    gate=gate,
                    gate_uses_ctxt=gate_uses_ctxt,
                    init_zeros=False,
                    use_bias=use_bias,
                )
            else:
                self.mod = MultiHeadContextOnlyVBetaModulatedFeatureLayer(
                    d=outp_dim,
                    ctxt_dim=ctxt_dim,
                    num_heads=1,
                    hidden=mod_hidden,
                    bias_hidden=mod_bias_hidden,
                    n_layers=mod_n_layers,
                    act=act,
                    nrm=nrm,
                    drp=drp,
                    gate=gate,
                    gate_uses_ctxt=gate_uses_ctxt,
                    init_zeros=False,
                    use_bias=use_bias,
                )
        else:
            self.mod = SelfModulatedFeatureLayer(
                d=outp_dim,
                ctxt_dim=ctxt_dim,
                hidden=mod_hidden,
                bias_hidden=mod_bias_hidden,
                n_layers=mod_n_layers,
                act=act,
                nrm=nrm,
                drp=drp,
                gate=gate,
                init_zeros=False,
                use_bias=use_bias,
            )

        self.post = HalfActivatedMLPBlock(
            inpt_dim=outp_dim,
            outp_dim=outp_dim,
            ctxt_dim=ctxt_dim,
            n_layers=post_layers,
            act=act,
            nrm=nrm,
            drp=drp,
            do_res=False,
            init_zeros=init_zeros_post,   # useful for resnet-ish behavior
            use_bias=use_bias,
        )

    def forward(self, x: T.Tensor, ctxt: T.Tensor | None = None) -> T.Tensor:
        h = self.pre(x, ctxt)
        h = self.mod(h, ctxt)
        y = self.post(h, ctxt)

        if self.do_res == "adjust":
            if self.inpt_dim < self.outp_dim:
                y[..., : self.inpt_dim] = y[..., : self.inpt_dim] + x
            else:
                y = y + x[..., : self.outp_dim]
        elif self.do_res:
            y = y + x

        return y


class ModulatedNetwork(nn.Module):
    """
    Mimics DenseNetwork (input block -> N blocks -> optional output block),
    but hidden blocks are ModulatedBlock (Dense -> SelfModulated -> Dense) with residuals.

    Notes:
    - Context handling matches DenseNetwork (broadcast ctxt to input shape).
    - Input/output MLP blocks are optional and disabled by default.
    """
    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int = 0,
        ctxt_dim: int = 0,
        hddn_dim: int | list = 32,
        num_blocks: int = 1,
        # pre/post depths inside each modulated block
        do_inpt: bool = False,
        pre_layers: int = 1,
        post_layers: int = 1,
        act_h: str = "silu",
        act_o: str = "none",
        act_i: str | None = None,
        do_out: bool = False,
        nrm: str = "none",
        drp: float = 0.0,
        drp_on_output: bool = False,
        nrm_on_output: bool = False,
        do_res: bool | str = False,
        ctxt_in_inpt: bool = True,
        ctxt_in_hddn: bool = False,
        ctxt_in_out: bool = False,
        use_bias: bool = True,
        # modulation specifics
        gate: str = "sigmoid",
        mod_hidden: int = 32,
        mod_bias_hidden: int = 16,
        mod_n_layers: int = 2,
        context_only_vbeta: bool = False,
        use_multi_head_context_only_vbeta: bool = False,
        multi_head_num_heads: int = 4,
        use_shake_shake: bool = False,
        gate_uses_ctxt: bool = True,
        # init options
        hddn_init_zeros: bool = False,
        output_init_zeros: bool = False,
        unit_in_out_res: bool = False,
    ):
        super().__init__()

        if ctxt_dim:
            if not ctxt_in_inpt and not ctxt_in_hddn and not ctxt_in_out:
                raise ValueError("Network has context inputs but nowhere to use them!")

        self.inpt_dim = inpt_dim
        if not isinstance(hddn_dim, int):
            self.hddn_dim = hddn_dim
        else:
            self.hddn_dim = num_blocks * [hddn_dim]

        self.num_blocks = len(self.hddn_dim)
        self.ctxt_dim = ctxt_dim
        self.do_inpt = do_inpt
        self.do_out = do_out
        self.unit_in_out_res = unit_in_out_res

        if self.do_inpt and self.num_blocks == 0:
            raise ValueError("do_inpt=True requires at least one hidden block")

        # Input block (optional plain HalfActivatedMLPBlock, like DenseNetwork)
        if self.do_inpt:
            self.input_block = HalfActivatedMLPBlock(
                inpt_dim=self.inpt_dim,
                outp_dim=self.hddn_dim[0],
                ctxt_dim=self.ctxt_dim if ctxt_in_inpt else 0,
                n_layers=1,
                act=act_i or act_h,
                nrm=nrm,
                drp=drp,
                do_res=do_res,
                use_bias=use_bias,
            )
        else:
            self.input_block = None

        # Hidden modulated blocks
        self.hidden_blocks = nn.ModuleList()
        block_dims = []
        if self.num_blocks > 0:
            if self.do_inpt:
                block_dims.extend(zip(self.hddn_dim[:-1], self.hddn_dim[1:]))
            else:
                block_dims.append((self.inpt_dim, self.hddn_dim[0]))
                block_dims.extend(zip(self.hddn_dim[:-1], self.hddn_dim[1:]))

        for h_1, h_2 in block_dims:
            self.hidden_blocks.append(
                ModulatedBlock(
                    inpt_dim=h_1,
                    outp_dim=h_2,
                    ctxt_dim=self.ctxt_dim if ctxt_in_hddn else 0,
                    pre_layers=pre_layers,
                    post_layers=post_layers,
                    act=act_h,
                    nrm=nrm,
                    drp=drp,
                    do_res=do_res,
                    use_bias=use_bias,
                    gate=gate,
                    mod_hidden=mod_hidden,
                    mod_bias_hidden=mod_bias_hidden,
                    mod_n_layers=mod_n_layers,
                    context_only_vbeta=context_only_vbeta,
                    use_multi_head_context_only_vbeta=use_multi_head_context_only_vbeta,
                    multi_head_num_heads=multi_head_num_heads,
                    use_shake_shake=use_shake_shake,
                    gate_uses_ctxt=gate_uses_ctxt,
                    init_zeros_post=hddn_init_zeros,
                )
            )

        if self.num_blocks > 0:
            if self.do_inpt and len(self.hidden_blocks) == 0:
                core_out_dim = self.hddn_dim[0]
            else:
                core_out_dim = self.hddn_dim[-1]
        else:
            core_out_dim = self.inpt_dim

        if (not do_out) and outp_dim and (outp_dim != core_out_dim):
            raise ValueError(
                "ModulatedNetwork was given outp_dim != core output dim, but do_out=False. "
                "Set do_out=True to project to outp_dim, or leave outp_dim unset/matching core width."
            )

        self.hidden_features = core_out_dim  # nflows compat, like DenseNetwork
        self.outp_dim = (outp_dim or inpt_dim) if do_out else core_out_dim

        # Output block (plain HalfActivatedMLPBlock, like DenseNetwork)
        if do_out:
            self.output_block = HalfActivatedMLPBlock(
                inpt_dim=core_out_dim,
                outp_dim=self.outp_dim,
                ctxt_dim=self.ctxt_dim if ctxt_in_out else 0,
                n_layers=1,
                act=act_o,
                init_zeros=output_init_zeros,
                nrm=nrm if nrm_on_output else "none",
                drp=drp if drp_on_output else 0.0,
                do_res=do_res,
                use_bias=use_bias,
            )
        else:
            self.output_block = None

    def forward(self, inputs: T.Tensor, ctxt: T.Tensor | None = None) -> T.Tensor:
        # Broadcast ctxt like DenseNetwork
        if ctxt is not None:
            dim_diff = inputs.dim() - ctxt.dim()
            if dim_diff > 0:
                ctxt = ctxt.view(ctxt.shape[0], *dim_diff * (1,), *ctxt.shape[1:])
                ctxt = ctxt.expand(*inputs.shape[:-1], -1)

        x = self.input_block(inputs, ctxt) if self.input_block is not None else inputs

        for blk in self.hidden_blocks:
            x = blk(x, ctxt)

        if self.do_out and self.output_block is not None:
            x = self.output_block(x, ctxt)

        if self.unit_in_out_res and self.do_out:
            if self.inpt_dim <= self.outp_dim:
                x = x + nn.functional.pad(inputs, (0, self.outp_dim - self.inpt_dim), "constant", 0)
            else:
                x = x + inputs[..., : self.outp_dim]

        return x

    def __repr__(self):
        string = ""
        if self.input_block is not None:
            string += "\n  (inp): " + repr(self.input_block) + "\n"
        for i, blk in enumerate(self.hidden_blocks):
            string += f"  (m-{i+1}): {blk.__class__.__name__}({blk.inpt_dim}->{blk.outp_dim})\n"
        if self.do_out and self.output_block is not None:
            string += "  (out): " + repr(self.output_block)
        return string