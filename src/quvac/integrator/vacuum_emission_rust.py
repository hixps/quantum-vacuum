'''
Calculation of vacuum emission integral in Rust (box diagram, F^4).
'''

import os
from pathlib import Path

import numpy as np


class VacuumEmissionRust:
    '''
    Empty
    '''
    def __init__(self, field, grid, nthreads=None):
        self.field = field
        self.grid_xyz = grid
        # Update local dict with variables from GridXYZ class
        self.__dict__.update(self.grid_xyz.__dict__)

    def save_rust_input(self, path):
        data = {
            "x": self.grid[0],
            "y": self.grid[1],
            "z": self.grid[2],
            "t": self.t_grid,
            "a1": self.field.a1,
            "a2": self.field.a2,
        }
        save_path = os.path.join(path, "rust", "rust_input.npz")
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        np.savez(save_path, **data)


    def calculate_amplitudes(
        self, t_grid, integration_method="trapezoid", save_path=None
    ):
        """
        Calculate the vacuum emission amplitudes and save the result.
        """
        self.t_grid = t_grid
        path = os.path.dirname(save_path)

        self.save_rust_input(path)

        self.launch_rust_calculation()

        self.read_rust_output()