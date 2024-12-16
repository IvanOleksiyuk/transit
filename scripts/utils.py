import numpy as np

def linear_interpolation(x, y, x_new):
    # Convert lists to numpy arrays for easier manipulation
    x = np.array(x)
    y = np.array(y)

    # Ensure that x_new is within the range of x
    if np.any(x_new < np.min(x)) or np.any(x_new > np.max(x)):
        raise ValueError("x_new is out of bounds of x.")

    if hasattr(x_new, "__len__"):
        x_news=[]
        for x_n in x_new:
            # Find indices of the two closest values in x that bound x_new
            idx = np.searchsorted(x, x_n)
            if x_n == x[idx - 1]:
                x_news.append(y[idx - 1])
            elif x_n == x[idx]:
                x_news.append(y[idx])
            else:
                x0, x1 = x[idx - 1], x[idx]
                y0, y1 = y[idx - 1], y[idx]
                x_news.append(y0 + (y1 - y0) * (x_n - x0) / (x1 - x0))
        y_new = np.array(x_news)
    else:
        idx = np.searchsorted(x, x_new)
        if x_new == x[idx - 1]:
            y_new = y[idx - 1]
        elif x_new == x[idx]:
            y_new = y[idx]
        else:
            x0, x1 = x[idx - 1], x[idx]
            y0, y1 = y[idx - 1], y[idx]
            y_new = y0 + (y1 - y0) * (x_new - x0) / (x1 - x0)

    return y_new

def get_y_and_std_curves(curves, x):
    y = []
    if hasattr(x, "__len__"):
        for curve in curves:
            y += [list(linear_interpolation(curve[0], curve[1], x))]
        y = np.array(y)
        return np.mean(y, axis=0), np.std(y, axis=0)
    else:
        for curve in curves:
            y += [linear_interpolation(curve[0], curve[1], x)]
        return np.mean(y), np.std(y)

def filter_finite_values(x):
    return x[np.isfinite(x)]

def get_common_x(curves, mode="concat"):
    min_x = max([np.min(filter_finite_values(curve[0])) for curve in curves])
    max_x = min([np.max(filter_finite_values(curve[0])) for curve in curves])
    if mode == "concat":
        common_x = np.sort(np.unique(np.concatenate([filter_finite_values(curve[0]) for curve in curves])))
        common_x = common_x[(common_x >= min_x) & (common_x <= max_x)]
    else:
        common_x = np.linspace(min_x, max_x, 1000)
    return common_x

def flip_curve(curve):
    curve_new = []
    for c in curve:
        curve_new.append(c[::-1])
    return curve_new