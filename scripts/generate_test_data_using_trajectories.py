from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def random_gaussian_blobs_parametrers(n_blobs, n_vars):
    gaussian_parameters = []
    for _ in range(n_blobs):
        mean = np.random.uniform(-2, 2, size=n_vars)
        A = np.random.rand(n_vars, n_vars)
        cov = np.dot(A, A.T)/4  # To ensure the covariance matrix is positive definite
        # randomly make come correlations negative
        for i in range(n_vars):
            for j in range(i):
                if np.random.rand() < 0.5:
                    cov[i, j] = -cov[i, j]
                    cov[j, i] = -cov[j, i]
        weight = np.random.uniform(0.1, 1.0)
        gaussian_parameters.append({"mean": mean, "cov": cov, "weight": weight})
    # Normalize weights
    total_weight = sum(gp["weight"] for gp in gaussian_parameters)
    for gp in gaussian_parameters:
        gp["weight"] /= total_weight
    return gaussian_parameters

def sample_from_gaussian_mixture(gaussian_parameters, n_points):
    n_blobs = len(gaussian_parameters)
    weights = [gp["weight"] for gp in gaussian_parameters]
    chosen_blobs = np.random.choice(n_blobs, size=n_points, p=weights)
    samples = []
    for i in range(n_blobs):
        n_samples = np.sum(chosen_blobs == i)
        if n_samples > 0:
            gp = gaussian_parameters[i]
            samples_blob = np.random.multivariate_normal(gp["mean"], gp["cov"], size=n_samples)
            samples.append(samples_blob)
    return np.vstack(samples)

def trajectory_constant(point, t):
    return point

def trajectory_linear(point, t, velocity="random", mask=None):
    len_= len(point[mask])
    if velocity == None:
        velocity = np.ones(len_)
    if velocity == "random":
        velocity = np.random.uniform(-0.5, 0.5, size=len_)
    return point[mask] + velocity * t

def create_random_quadratic_trajectory_params(len_):
    a = np.random.uniform(-1, 1, size=len_)
    peak_t = np.random.uniform(-1, 1, size=len_)
    lin = np.random.uniform(-1, 1, size=len_)
    return {"a": a, "peak_t": peak_t, "lin": lin}

def trajectory_quadratic(point, t, a=0, peak_t=0, lin=0, mask=None):
    return point[mask] + lin * t + 0.5 * a * (t - peak_t)**2

def randomt_trajectory_displacement(point_at0, t_interval, trajectory_type="linear", parameters=None):
    t0, t1 = t_interval
    t = np.random.uniform(t0, t1)
    if trajectory_type == "constant":
        return t, trajectory_constant(point_at0, t)
    elif trajectory_type == "linear":
        return t, trajectory_linear(point_at0, t)
    elif trajectory_type == "quadratic":
        return t, trajectory_quadratic(point_at0, t, **parameters)
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")

def generate_test_data_using_trajectories(n_vars=5, n_points=1000, t_interval=[-1, 1], gaussian_parameters=None, traj_type=None, traj_params=None, plot_path=""):
    # Step 0 handle some special cases for gaussian_parameters input
    if gaussian_parameters is None:
        gaussian_parameters = [
            {"mean": np.zeros(n_vars), "cov": np.eye(n_vars), "weight": 1.0}
        ]
    if gaussian_parameters=="random":
        gaussian_parameters = random_gaussian_blobs_parametrers(n_blobs=3, n_vars=n_vars)
    if traj_type is None:
        traj_type = "constant"

    # Step 1 generate gaussian blobs usign gaussian parameters such as mean and covariance
    points = sample_from_gaussian_mixture(gaussian_parameters, n_points)
    for i in range(n_vars):
        for j in range(i):
            plt.figure()
            plt.scatter(points[:, j], points[:, i], s=1)
            plt.xlabel(f"var_{j}")
            plt.ylabel(f"var_{i}")
            plt.title(f"Variable {i} vs Variable {j}")
            plt.grid()
            os.makedirs(f"{plot_path}/0_t", exist_ok=True)
            plt.savefig(f"{plot_path}/0_t/variable_{i}_vs_variable_{j}.png")
            plt.close()

    # Step 2 transport points alomg trajectories 
    t=[]
    for point in points:
        t_val, new_point = randomt_trajectory_displacement(point, t_interval, trajectory_type=traj_type, parameters=traj_params)
        point[:] = new_point  # Update point in place
        t.append(t_val)

    # Step 3 plot dataset
    for i in range(n_vars):
        plt.figure()
        plt.scatter(t, points[:, i], s=1)
        plt.xlabel("t")
        plt.ylabel(f"var_{i}")
        plt.title(f"Variable {i} vs Time")
        plt.grid()
        plt.savefig(f"{plot_path}variable_{i}_vs_time.png")

    return points, np.asarray(t)


def save_in_PAD_format(points, t, output_path, frame_key="data", time_key="t", original_key="orig"):
    """Persist generated samples in separate HDF5 frames for features and time."""
    points = np.asarray(points, dtype=np.float32)
    t = np.asarray(t, dtype=np.float32)
    if points.ndim != 2:
        raise ValueError("Points array must be 2-dimensional.")
    if points.shape[0] != t.shape[0]:
        raise ValueError("Points and time arrays must have the same length.")

    columns = [f"var_{i}" for i in range(points.shape[1])]
    df_features = pd.DataFrame(points, columns=columns).astype(np.float32)
    df_time = pd.DataFrame({time_key: t}).astype(np.float32)
    df_orig = pd.DataFrame({time_key: t}).astype(np.float32)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.HDFStore(output_path, mode="w") as store:
        store.put(frame_key, df_features)
        store.put(time_key, df_time)
        store.put(original_key, df_orig)


def split_sb_sr(points, t, sr_interval=[-0.3, 0.3]):
    """Split data into search space (ss) and search region (sr) based on time interval."""
    sr_mask = (t >= sr_interval[0]) & (t <= sr_interval[1])
    sb_points = points[~sr_mask]
    sb_t = t[~sr_mask]
    sr_points = points[sr_mask]
    sr_t = t[sr_mask]
    return (sb_points, sb_t), (sr_points, sr_t)

if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)

    output_path="/home/users/o/oleksiyu/WORK/weakly-supervised-search/user/test_runs/toy3/data/"
    os.makedirs(output_path+"plots/", exist_ok=True)
    points, t = generate_test_data_using_trajectories(
        n_vars=4,
        traj_params=create_random_quadratic_trajectory_params(len_=4),
        n_points=200000,
        t_interval=[-1, 1],
        gaussian_parameters="random",
        traj_type="quadratic",
        plot_path=output_path+"plots/"
    )
    sb_data, sr_data = split_sb_sr(points, t, sr_interval=[-0.3, 0.3])
    save_in_PAD_format(sb_data[0], sb_data[1], output_path+"data/sb.h5")
    save_in_PAD_format(sr_data[0], sr_data[1], output_path+"data/sr.h5")
