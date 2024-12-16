import torch
import torch.nn as nn

class PearsonCorrelation(nn.Module):
    """Distance correlation loss"""

    def __init__(self):
        super(PearsonCorrelation, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n_features_x = x.shape[1]
        n_features_y = y.shape[1]
        
        # Compute the covariance matrix
        covariance_matrix = torch.matmul(torch.t(x-x.mean(dim=0)), y-y.mean(dim=0)) 
        var_x = torch.sum((x-x.mean(dim=0))**2, dim=0)
        var_y = torch.sum((y-y.mean(dim=0))**2, dim=0)
        CorXY = covariance_matrix / torch.sqrt(var_x.view(-1, 1) * var_y.view(1, -1))
        average_correlation = torch.sum(torch.abs(CorXY)) / (n_features_x * n_features_y)
        return average_correlation