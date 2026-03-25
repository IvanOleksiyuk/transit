import torch as T
import torch.nn as nn

from .bayesian import BayesianLinear
from .torch_utils import get_act, get_nrm, masked_pool, smart_cat
from .modules_myy import MLPBlock


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
        act: str = "lrlu",
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

        # Use your MLPBlock so activations/norm/dropout match the rest of the codebase
        self.v_net = MLPBlock(
            inpt_dim=d,
            outp_dim=d,
            ctxt_dim=ctxt_dim,
            n_layers=2,
            act=act,
            nrm=nrm,
            drp=drp,
            do_res=False,
            init_zeros=init_zeros,
            use_bias=use_bias,
        )

        self.q_net = MLPBlock(
            inpt_dim=d,
            outp_dim=d,
            ctxt_dim=ctxt_dim,
            n_layers=2,
            act=act,
            nrm=nrm,
            drp=drp,
            do_res=False,
            init_zeros=init_zeros,
            use_bias=use_bias,
        )

        # Small bias MLP (intentionally smaller capacity)
        self.beta_net = MLPBlock(
            inpt_dim=d,
            outp_dim=d,
            ctxt_dim=ctxt_dim,
            n_layers=2,
            act=act,
            nrm=nrm,
            drp=drp,
            do_res=False,
            init_zeros=init_zeros,
            use_bias=use_bias,
            # If you want to enforce "small", pass smaller hidden via DenseNetwork instead;
            # MLPBlock width is fixed to outp_dim, so we keep it shallow here.
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

class ContextOnlyVBetaModulatedFeatureLayer(nn.Module):
    """
    Dimension-preserving modulation with context-only value and bias:
        y = x + v(ctxt) * gate(x,ctxt) + beta(ctxt)

    - v and beta depend only on ctxt
    - gate can still depend on both x and ctxt
    """

    def __init__(
        self,
        d: int,
        ctxt_dim: int,
        hidden: int = 32,
        bias_hidden: int = 16,
        act: str = "lrlu",
        nrm: str = "none",
        drp: float = 0.0,
        gate: str = "sigmoid",
        gate_uses_ctxt: bool = True,
        init_zeros: bool = False,
        use_bias: bool = True,
    ):
        super().__init__()

        if ctxt_dim <= 0:
            raise ValueError("ContextOnlyVBetaModulatedFeatureLayer requires ctxt_dim > 0")

        self.d = d
        self.ctxt_dim = ctxt_dim
        self.gate = gate
        self.gate_uses_ctxt = gate_uses_ctxt

        # Context-only value branch v(ctxt)
        self.v_net = MLPBlock(
            inpt_dim=ctxt_dim,
            outp_dim=d,
            ctxt_dim=0,
            n_layers=2,
            act=act,
            nrm=nrm,
            drp=drp,
            do_res=False,
            init_zeros=init_zeros,
            use_bias=use_bias,
        )

        # Gate branch gate(x, ctxt)
        self.q_net = MLPBlock(
            inpt_dim=d,
            outp_dim=d,
            ctxt_dim=ctxt_dim if gate_uses_ctxt else 0,
            n_layers=2,
            act=act,
            nrm=nrm,
            drp=drp,
            do_res=False,
            init_zeros=init_zeros,
            use_bias=use_bias,
        )

        # Context-only bias branch beta(ctxt)
        self.beta_net = MLPBlock(
            inpt_dim=ctxt_dim,
            outp_dim=d,
            ctxt_dim=0,
            n_layers=2,
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

    def forward(self, x: T.Tensor, ctxt: T.Tensor | None = None) -> T.Tensor:
        if ctxt is None:
            raise ValueError("ctxt must be provided for ContextOnlyVBetaModulatedFeatureLayer")

        v = self.v_net(ctxt)
        gate_ctxt = ctxt if self.gate_uses_ctxt else None
        g = self._apply_gate(self.q_net(x, gate_ctxt))
        beta = self.beta_net(ctxt)
        return x + v * g + beta


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
        act: str = "lrlu",
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
        self.v_net_1 = MLPBlock(
            inpt_dim=ctxt_dim,
            outp_dim=d,
            ctxt_dim=0,
            n_layers=2,
            act=act,
            nrm=nrm,
            drp=drp,
            do_res=False,
            init_zeros=init_zeros,
            use_bias=use_bias,
        )
        self.q_net_1 = MLPBlock(
            inpt_dim=d,
            outp_dim=d,
            ctxt_dim=ctxt_dim if gate_uses_ctxt else 0,
            n_layers=2,
            act=act,
            nrm=nrm,
            drp=drp,
            do_res=False,
            init_zeros=init_zeros,
            use_bias=use_bias,
        )
        self.beta_net_1 = MLPBlock(
            inpt_dim=ctxt_dim,
            outp_dim=d,
            ctxt_dim=0,
            n_layers=2,
            act=act,
            nrm=nrm,
            drp=drp,
            do_res=False,
            init_zeros=init_zeros,
            use_bias=use_bias,
        )

        # Branch 2
        self.v_net_2 = MLPBlock(
            inpt_dim=ctxt_dim,
            outp_dim=d,
            ctxt_dim=0,
            n_layers=2,
            act=act,
            nrm=nrm,
            drp=drp,
            do_res=False,
            init_zeros=init_zeros,
            use_bias=use_bias,
        )
        self.q_net_2 = MLPBlock(
            inpt_dim=d,
            outp_dim=d,
            ctxt_dim=ctxt_dim if gate_uses_ctxt else 0,
            n_layers=2,
            act=act,
            nrm=nrm,
            drp=drp,
            do_res=False,
            init_zeros=init_zeros,
            use_bias=use_bias,
        )
        self.beta_net_2 = MLPBlock(
            inpt_dim=ctxt_dim,
            outp_dim=d,
            ctxt_dim=0,
            n_layers=2,
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
        residual:   y += x (if dims match, or "adjust" like your MLPBlock)
    """
    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int,
        ctxt_dim: int = 0,
        pre_layers: int = 1,
        post_layers: int = 1,
        act: str = "lrlu",
        nrm: str = "none",
        drp: float = 0.0,
        do_res: bool | str = False,   # False | True | "adjust"
        use_bias: bool = True,
        mod_hidden: int = 32,
        mod_bias_hidden: int = 16,
        gate: str = "sigmoid",
        context_only_vbeta: bool = False,
        use_shake_shake: bool = False,
        gate_uses_ctxt: bool = True,
        init_zeros_post: bool = False,
    ):
        super().__init__()
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim

        # Residual behavior: mimic MLPBlock semantics
        if do_res == "adjust":
            self.do_res = "adjust"
        else:
            self.do_res = bool(do_res) and (inpt_dim == outp_dim)

        self.pre = MLPBlock(
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

            if use_shake_shake:
                self.mod = ShakeShakeContextOnlyVBetaModulatedFeatureLayer(
                    d=outp_dim,
                    ctxt_dim=ctxt_dim,
                    hidden=mod_hidden,
                    bias_hidden=mod_bias_hidden,
                    act=act,
                    nrm=nrm,
                    drp=drp,
                    gate=gate,
                    gate_uses_ctxt=gate_uses_ctxt,
                    init_zeros=False,
                    use_bias=use_bias,
                )
            else:
                self.mod = ContextOnlyVBetaModulatedFeatureLayer(
                    d=outp_dim,
                    ctxt_dim=ctxt_dim,
                    hidden=mod_hidden,
                    bias_hidden=mod_bias_hidden,
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
                act=act,
                nrm=nrm,
                drp=drp,
                gate=gate,
                init_zeros=False,
                use_bias=use_bias,
            )

        self.post = MLPBlock(
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
    - By default: ctxt injected into input+hidden+output blocks similarly to DenseNetwork.
    """
    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int = 0,
        ctxt_dim: int = 0,
        hddn_dim: int | list = 32,
        num_blocks: int = 1,
        # pre/post depths inside each modulated block
        pre_layers: int = 1,
        post_layers: int = 1,
        act_h: str = "lrlu",
        act_o: str = "none",
        act_i: str | None = None,
        do_out: bool = True,
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
        context_only_vbeta: bool = False,
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
        self.do_out = do_out
        self.unit_in_out_res = unit_in_out_res

        self.outp_dim = outp_dim or inpt_dim if do_out else self.hddn_dim[-1]
        self.hidden_features = self.hddn_dim[-1]  # nflows compat, like DenseNetwork

        # Input block (plain MLPBlock, like DenseNetwork)
        self.input_block = MLPBlock(
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

        # Hidden modulated blocks
        self.hidden_blocks = nn.ModuleList()
        for h_1, h_2 in zip(self.hddn_dim[:-1], self.hddn_dim[1:]):
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
                    context_only_vbeta=context_only_vbeta,
                    use_shake_shake=use_shake_shake,
                    gate_uses_ctxt=gate_uses_ctxt,
                    init_zeros_post=hddn_init_zeros,
                )
            )

        # Output block (plain MLPBlock, like DenseNetwork)
        if do_out:
            self.output_block = MLPBlock(
                inpt_dim=self.hddn_dim[-1],
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

    def forward(self, inputs: T.Tensor, ctxt: T.Tensor | None = None) -> T.Tensor:
        # Broadcast ctxt like DenseNetwork
        if ctxt is not None:
            dim_diff = inputs.dim() - ctxt.dim()
            if dim_diff > 0:
                ctxt = ctxt.view(ctxt.shape[0], *dim_diff * (1,), *ctxt.shape[1:])
                ctxt = ctxt.expand(*inputs.shape[:-1], -1)

        x = self.input_block(inputs, ctxt)

        for blk in self.hidden_blocks:
            x = blk(x, ctxt)

        if self.do_out:
            x = self.output_block(x, ctxt)

        if self.unit_in_out_res and self.do_out:
            if self.inpt_dim <= self.outp_dim:
                x = x + nn.functional.pad(inputs, (0, self.outp_dim - self.inpt_dim), "constant", 0)
            else:
                x = x + inputs[..., : self.outp_dim]

        return x

    def __repr__(self):
        string = ""
        string += "\n  (inp): " + repr(self.input_block) + "\n"
        for i, blk in enumerate(self.hidden_blocks):
            string += f"  (m-{i+1}): {blk.__class__.__name__}({blk.inpt_dim}->{blk.outp_dim})\n"
        if self.do_out:
            string += "  (out): " + repr(self.output_block)
        return string