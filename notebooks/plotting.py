"Here we gather useful plotting functions"

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, pi


def save_fig(save_path, fig_name):
    """
    Save figure to file
    """
    if save_path is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        name = os.path.join(save_path, fig_name)
        plt.savefig(f"{name}.png", bbox_inches='tight')
        plt.savefig(f"{name}.pdf", bbox_inches='tight')


def plot_fields(field, t, plot_keys=None, cmap='RdBu',
                save_path=None):
    """
    Plot specified field components
    """
    E, B = field.calculate_field(t=t, mode='real')
    nx, ny, nz = field.grid_shape
    x, y, z = [ax*1e6 for ax in field.grid_xyz.grid]

    I = (E[0]**2 + E[1]**2 + E[2]**2 + B[0]**2 + B[1]**2 + B[2]**2)/2
    field_comps = {
        "Ex": E[0],
        "Ey": E[1],
        "Ez": E[2],
        "Bx": B[0],
        "By": B[1],
        "Bz": B[2],
        "I": I,
    }

    # 1st plot: xy and xz profiles at focus for given components
    plot_keys = plot_keys if plot_keys is not None else field_comps.keys()
    n_rows = len(plot_keys)
    n_cols = 2

    fig = plt.figure(figsize=(10, 5*n_rows), layout="constrained")
    for i,key in enumerate(plot_keys):
        plt.subplot(n_rows, n_cols, i*n_cols+1)
        plt.pcolormesh(y, x, field_comps[key][:, :, nz//2], shading=None,
                       rasterized=True, cmap=cmap)
        plt.xlabel("y [$\\mu$m]")
        plt.ylabel("x [$\\mu$m]")
        plt.title(f"{key} at z=0")

        plt.subplot(n_rows, n_cols, i*n_cols+2)
        plt.pcolormesh(z, x, field_comps[key][:, ny//2, :], shading=None,
                       rasterized=True, cmap=cmap)
        plt.xlabel("z [$\\mu$m]")
        plt.ylabel("x [$\\mu$m]")
        plt.title(f"{key} at z=0")
    save_fig(save_path, "field_profiles_focus")
    plt.show()

