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
        self.rust_input = os.path.join(path, "rust", "rust_input.npz")
        Path(os.path.dirname(self.rust_input)).mkdir(parents=True, exist_ok=True)
        np.savez(self.rust_input, **data)

    def launch_rust_calculation(self):
        # this is just a placeholder for now
        data = np.load(self.rust_input)
        print(list(data.keys()))
        print("Rust input loaded!")
        S1 = np.ones_like(data["a1"])
        S2 = np.zeros_like(data["a1"])

        data_to_save = {
            "x": data["x"],
            "y": data["y"],
            "z": data["z"],
            "S1": S1,
            "S2": S2,
        }
        self.rust_output = os.path.join(os.path.dirname(self.rust_input),
                                        "rust_output.npz")
        np.savez(self.rust_output, **data_to_save)

    def read_rust_output(self):
        data = np.load(self.rust_output)
        data_keys = list(data.keys())
        required_keys = "x y z S1 S2".split()

        err_msg = (f"Not all required keys ({required_keys}) are present in the Rust"
                   f"amplitude file ({data_keys})")
        assert all(key in data_keys for key in required_keys), err_msg

        os.rename(self.rust_output, self.save_path)

    def calculate_amplitudes(
        self, t_grid, integration_method="trapezoid", save_path=None
    ):
        """
        Calculate the vacuum emission amplitudes and save the result.
        """
        self.t_grid = t_grid
        self.save_path = save_path
        path = os.path.dirname(save_path)

        self.save_rust_input(path)

        self.launch_rust_calculation()

        self.read_rust_output()