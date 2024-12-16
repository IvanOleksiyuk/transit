import numpy as np
from matplotlib import pyplot as plt

def linear_interpolation(x, y, x_new):
    # Convert lists to numpy arrays for easier manipulation
    x = np.array(x)
    y = np.array(y)

    # Ensure that x_new is within the range of x
    if x_new < np.min(x) or x_new > np.max(x):
        raise ValueError("x_new is out of bounds of x.")

    # Find indices of the two closest values in x that bound x_new
    idx = np.searchsorted(x, x_new)
    
    # Handle edge case where x_new matches exactly the smallest or largest x value
    if x_new == x[idx - 1]:
        return y[idx - 1]
    elif x_new == x[idx]:
        return y[idx]

    # Get the two x and y values surrounding x_new
    x0, x1 = x[idx - 1], x[idx]
    y0, y1 = y[idx - 1], y[idx]

    # Perform linear interpolation
    y_new = y0 + (y1 - y0) * (x_new - x0) / (x1 - x0)

    return y_new
    
    
fpr_target = 1e-3
n_sig_list = [0, 50, 100, 333, 500, 667, 1000, 3000]
SI_at_fpr = []

for n_sigin in n_sig_list:
    curve = np.load()
    tpr = curve[0, :]
    fpr = curve[1, :]
    SI = tpr/np.sqrt(fpr)
    SI_at_fpr.append(linear_interpolation(fpr, SI, fpr_target))

plt.plot(n_sig_list, SI_at_fpr)