import torch
import numpy as np
def pairwise_distances_torch(x):
    #x should be two dimensional
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()

def GaussianKernelMatrix_torch(x, sigma=1):
    pairwise_distances_ = pairwise_distances_torch(x)
    return torch.exp(-pairwise_distances_ /sigma)

def HSIC_torch(x, y, s_x=1, s_y=1, cuda=True):
    m,_ = x.shape #batch size
    K = GaussianKernelMatrix_torch(x,s_x)
    L = GaussianKernelMatrix_torch(y,s_y)
    H = torch.eye(m) - 1.0/m * torch.ones((m,m))
    if cuda:
        H = H.double().cuda()
    HSIC = torch.trace(torch.mm(L,torch.mm(H,torch.mm(K,H))))/((m-1)**2)
    return HSIC

# Numpy equivalent
def pairwise_distances_np(x):
    #x should be two dimensional
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*(x @ x.T) + instances_norm + instances_norm.T

def GaussianKernelMatrix_np(x, sigma=1):
    pairwise_distances_ = pairwise_distances_np(x)
    return np.exp(-pairwise_distances_ /sigma)

def HSIC_np(x, y, s_x=1, s_y=1):
    m,_ = x.shape #batch size
    K = GaussianKernelMatrix_torch(x,s_x)
    L = GaussianKernelMatrix_torch(y,s_y)
    H = np.eye(m) - 1.0/m * np.ones((m,m))
    HSIC = np.trace(L @ (H @ (K @ H)))/((m-1)**2)
    return HSIC