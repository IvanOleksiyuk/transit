import copy

import torch as T
import torch.nn as nn


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
		if num_branches < 2:
			raise ValueError("num_branches must be >= 2 for shake-shake regularization")

		self.num_branches = num_branches
		self.shared_forward_backward_weights = shared_forward_backward_weights
		self.branches = nn.ModuleList([copy.deepcopy(module) for _ in range(num_branches)])

	@staticmethod
	def _mix_with_coeffs(branch_outputs: list[T.Tensor], coeffs: T.Tensor) -> T.Tensor:
		return sum(coeffs[i] * branch_outputs[i] for i in range(len(branch_outputs)))

	def _shake_shake(self, branch_outputs: list[T.Tensor]) -> T.Tensor:
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

