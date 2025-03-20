import torch

class SelectiveDequantizationTransform(torch.nn.Module):
    # Performs dequantization on a subset of features

    def __init__(self, discrete_indices, discrete_shift, discrete_scale):
        super().__init__()
        self.discrete_indices = torch.as_tensor(discrete_indices)
        self.discrete_shift = torch.as_tensor(discrete_shift)
        self.discrete_scale = torch.as_tensor(discrete_scale)

    def forward(self, inputs):
        """Forward pass: Add uniform noise to discrete features."""
        outputs = inputs.clone()  # Clone to avoid modifying the original tensor
        for i, idx in enumerate(self.discrete_indices):
            discrete = outputs[:, idx] * self.discrete_scale[i] + self.discrete_shift[i]
            noise = torch.rand_like(discrete) - 0.5
            outputs[:, idx] = (
                discrete + noise - self.discrete_shift[i]
            ) / self.discrete_scale[i]

        return outputs

    def inverse(self, inputs):
        """Inverse pass: Map discrete features back using floor function."""
        outputs = inputs.clone()
        for i, idx in enumerate(self.discrete_indices):
            # Scale up by discrete_scale
            # apply floor to map to original bins
            # and scale down
            discrete = (
                outputs[:, idx] * self.discrete_scale[i] + self.discrete_shift[i] + 0.5
            )

            outputs[:, idx] = (
                torch.floor(discrete) - self.discrete_shift[i]
            ) / self.discrete_scale[i]
            # Clamp so output non-negative
            # outputs[:, idx] = torch.clamp(
            #     outputs[:, idx],
            #     min=-self.discrete_shift[i] / self.discrete_scale[i],
            # )

        return outputs