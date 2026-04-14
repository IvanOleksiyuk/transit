import copy

import torch as T
import torch.nn as nn

from transit.mltools.mlp import MLP
from transit.mattstools.mattstools.modules_myy import DenseNetwork


class ShakeShakeModuleWrapper(nn.Module):
	"""Wrap an arbitrary module with N-way shake-shake regularization.

	The input module is deep-copied into `num_branches` independent branches.
	During training, branch outputs are mixed with random convex coefficients.
	During evaluation, outputs are averaged across branches.

	Notes:
	- This wrapper expects each branch to return a `torch.Tensor`.
	- The wrapped module's `forward` signature is preserved via `*args, **kwargs`.
	"""

	def __init__(
		self,
		module: nn.Module,
		num_branches: int = 2,
		shared_forward_backward_weights: bool = False,
	) -> None:
		super().__init__()

		if not isinstance(module, nn.Module):
			raise TypeError("module must be an instance of torch.nn.Module")
		if num_branches < 1:
			raise ValueError("num_branches must be >= 1")

		self.num_branches = num_branches
		self.shared_forward_backward_weights = shared_forward_backward_weights
		self.branches = nn.ModuleList([copy.deepcopy(module) for _ in range(num_branches)])

	@staticmethod
	def _mix_with_coeffs(branch_outputs: list[T.Tensor], coeffs: T.Tensor) -> T.Tensor:
		return sum(coeffs[i] * branch_outputs[i] for i in range(len(branch_outputs)))

	def _shake_shake(self, branch_outputs: list[T.Tensor]) -> T.Tensor:
		if len(branch_outputs) == 1:
			return branch_outputs[0]

		if not self.training:
			return T.stack(branch_outputs, dim=0).mean(dim=0)

		ref = branch_outputs[0]
		if ref.dim() == 0:
			coeff_shape = [1]
		else:
			coeff_shape = [ref.shape[0]] + [1] * (ref.dim() - 1)

		alpha = T.rand((len(branch_outputs), *coeff_shape), device=ref.device, dtype=ref.dtype)
		alpha = alpha / alpha.sum(dim=0, keepdim=True).clamp_min(1e-12)

		if self.shared_forward_backward_weights:
			return self._mix_with_coeffs(branch_outputs, alpha)

		beta = T.rand((len(branch_outputs), *coeff_shape), device=ref.device, dtype=ref.dtype)
		beta = beta / beta.sum(dim=0, keepdim=True).clamp_min(1e-12)

		forward_mix = self._mix_with_coeffs(branch_outputs, alpha)
		backward_mix = self._mix_with_coeffs(branch_outputs, beta)
		return backward_mix + (forward_mix - backward_mix).detach()

	def forward(self, *args, **kwargs) -> T.Tensor:
		outputs = [branch(*args, **kwargs) for branch in self.branches]

		if not all(isinstance(out, T.Tensor) for out in outputs):
			raise TypeError("ShakeShakeModuleWrapper expects wrapped module to return torch.Tensor")

		return self._shake_shake(outputs)


class ShakeShakeMLP(nn.Module):
	"""Shake-shake analogue of `transit.mltools.mlp.MLP`.

	Constructor arguments intentionally mirror `MLP` so Hydra configs used for
	`MLP` can be reused with this class.
	"""

	def __init__(
		self,
		inpt_dim: int,
		outp_dim: int = 0,
		ctxt_dim: int = 0,
		hddn_dim: int | list = 32,
		num_blocks: int = 1,
		n_lyr_pbk: int = 1,
		act_h: str = "lrlu",
		act_o: str = "none",
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
		num_branches: int = 2,
		shared_forward_backward_weights: bool = False,
	) -> None:
		super().__init__()

		self.inpt_dim = inpt_dim
		self.ctxt_dim = ctxt_dim
		self.hddn_dim = hddn_dim if not isinstance(hddn_dim, int) else num_blocks * [hddn_dim]
		self.num_blocks = len(self.hddn_dim)
		self.do_out = do_out
		self.num_branches = num_branches
		self.shared_forward_backward_weights = shared_forward_backward_weights

		# Keep compatibility with code that expects nflows-like attributes.
		self.hidden_features = self.hddn_dim[-1]
		self.outp_dim = outp_dim or inpt_dim if do_out else self.hddn_dim[-1]

		base_mlp = MLP(
			inpt_dim=inpt_dim,
			outp_dim=outp_dim,
			ctxt_dim=ctxt_dim,
			hddn_dim=hddn_dim,
			num_blocks=num_blocks,
			n_lyr_pbk=n_lyr_pbk,
			act_h=act_h,
			act_o=act_o,
			do_out=do_out,
			nrm=nrm,
			drp=drp,
			nrm_inside=nrm_inside,
			drp_on_output=drp_on_output,
			nrm_on_output=nrm_on_output,
			do_res=do_res,
			ctxt_in_inpt=ctxt_in_inpt,
			ctxt_in_hddn=ctxt_in_hddn,
			ctxt_in_out=ctxt_in_out,
			do_bayesian=do_bayesian,
			init_zeros=init_zeros,
			use_bias=use_bias,
		)

		self.shake = ShakeShakeModuleWrapper(
			module=base_mlp,
			num_branches=num_branches,
			shared_forward_backward_weights=shared_forward_backward_weights,
		)

	def forward(
		self,
		inputs: T.Tensor,
		ctxt: T.Tensor | None = None,
		context: T.Tensor | None = None,
	) -> T.Tensor:
		# Use context as a synonym for ctxt (mirrors MLP API).
		if context is not None:
			ctxt = context
		return self.shake(inputs, ctxt=ctxt)

	def __repr__(self) -> str:
		return (
			"ShakeShakeMLP("
			f"{self.inpt_dim}->{self.outp_dim}, blocks={self.num_blocks}, "
			f"branches={self.num_branches})"
		)


class ShakeShakeDenseNetwork(nn.Module):
	"""Shake-shake analogue of `modules_myy.DenseNetwork`.

	Constructor arguments mirror `DenseNetwork` so existing Hydra configs can be
	reused by changing only the target.
	"""

	def __init__(
		self,
		inpt_dim: int,
		outp_dim: int = 0,
		ctxt_dim: int = 0,
		hddn_dim: int | list = 32,
		num_blocks: int = 1,
		n_lyr_pbk: int = 1,
		act_h: str = "lrlu",
		act_o: str = "none",
		act_i=None,
		do_out: bool = True,
		nrm: str = "none",
		drp: float = 0,
		drp_on_output: bool = False,
		nrm_on_output: bool = False,
		do_res: bool | str = False,
		ctxt_in_inpt: bool = True,
		ctxt_in_hddn: bool = False,
		ctxt_in_out: bool = False,
		do_bayesian: bool = False,
		inpt_init_zeros: bool = False,
		hddn_init_zeros: bool = False,
		output_init_zeros: bool = False,
		use_bias: bool = True,
		scale_output_hidden=None,
		unit_in_out_res: bool = False,
		num_branches: int = 2,
		shared_forward_backward_weights: bool = False,
	) -> None:
		super().__init__()

		self.inpt_dim = inpt_dim
		self.ctxt_dim = ctxt_dim
		self.hddn_dim = hddn_dim if not isinstance(hddn_dim, int) else num_blocks * [hddn_dim]
		self.num_blocks = len(self.hddn_dim)
		self.do_out = do_out
		self.num_branches = num_branches
		self.shared_forward_backward_weights = shared_forward_backward_weights

		# Keep compatibility with code that expects nflows-like attributes.
		self.hidden_features = self.hddn_dim[-1]
		self.outp_dim = outp_dim or inpt_dim if do_out else self.hddn_dim[-1]

		base_net = DenseNetwork(
			inpt_dim=inpt_dim,
			outp_dim=outp_dim,
			ctxt_dim=ctxt_dim,
			hddn_dim=hddn_dim,
			num_blocks=num_blocks,
			n_lyr_pbk=n_lyr_pbk,
			act_h=act_h,
			act_o=act_o,
			act_i=act_i,
			do_out=do_out,
			nrm=nrm,
			drp=drp,
			drp_on_output=drp_on_output,
			nrm_on_output=nrm_on_output,
			do_res=do_res,
			ctxt_in_inpt=ctxt_in_inpt,
			ctxt_in_hddn=ctxt_in_hddn,
			ctxt_in_out=ctxt_in_out,
			do_bayesian=do_bayesian,
			inpt_init_zeros=inpt_init_zeros,
			hddn_init_zeros=hddn_init_zeros,
			output_init_zeros=output_init_zeros,
			use_bias=use_bias,
			scale_output_hidden=scale_output_hidden,
			unit_in_out_res=unit_in_out_res,
		)

		self.shake = ShakeShakeModuleWrapper(
			module=base_net,
			num_branches=num_branches,
			shared_forward_backward_weights=shared_forward_backward_weights,
		)

	def forward(
		self,
		inputs: T.Tensor,
		ctxt: T.Tensor | None = None,
		context: T.Tensor | None = None,
	) -> T.Tensor:
		if context is not None:
			ctxt = context
		return self.shake(inputs, ctxt=ctxt)

	def __repr__(self) -> str:
		return (
			"ShakeShakeDenseNetwork("
			f"{self.inpt_dim}->{self.outp_dim}, blocks={self.num_blocks}, "
			f"branches={self.num_branches})"
		)

