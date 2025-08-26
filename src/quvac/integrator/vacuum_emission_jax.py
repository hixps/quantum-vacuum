"""
Calculation of vacuum emission integral (box diagram, F^4) in Jax.

Currently supports:

1. Calculation of total vacuum emission signal for given field configuration. 
All fields are treated as external.

2. Separation of fields into pump and probe with subsequent calculation of
probe channel signal.

.. note::
    For details on implementation check out :ref:`implementation` section.
"""

from dataclasses import dataclass
from typing import NamedTuple
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numexpr as ne
import numpy as np
import pyfftw
from scipy.constants import alpha, c, e, hbar, m_e, pi

from quvac import config

BS = m_e**2 * c**2 / (hbar * e)  # Schwinger magnetic field


# @dataclass
# class GridParams:
#     kabs: np.ndarray
#     e1x: np.ndarray
#     e1y: np.ndarray
#     e1z: np.ndarray
#     e2x: np.ndarray
#     e2y: np.ndarray
#     e2z: np.ndarray
#     dVk: float


class IntegratorState(NamedTuple):
    kabs: np.ndarray
    e1x: np.ndarray
    e1y: np.ndarray
    e1z: np.ndarray
    e2x: np.ndarray
    e2y: np.ndarray
    e2z: np.ndarray
    dVk: float
    a1: np.ndarray
    a2: np.ndarray
    t0: float


class VacuumEmissionJax:
    """
    Calculator of Vacuum Emission amplitude from given fields

    Parameters
    ----------
    field : quvac.Field
        External fields.
    grid : quvac.grid.GridXYZ
        Spatial and spectral grid.
    nthreads : int, optional
        Number of threads to use for calculations. If not provided, 
        defaults to the number of CPU cores.
    channels : bool, optional
        Whether to calculate a particular channel in vacuum emission amplitude. 
        Default is False.

    """

    def __init__(self, field, grid, nthreads=None, channels=False):
        self.field = field
        self.grid_xyz = grid

        self.integrator_state = VacuumEmissionJax.create_state(field, grid)
        # Update local dict with variables from GridXYZ class
        self.__dict__.update(self.grid_xyz.__dict__)
        # self.channels = channels

        self.c = c
        self.nthreads = nthreads if nthreads else os.cpu_count()

        # Define symbolic expressions to evaluate later
        # self.F_expr = "(Bx**2 + By**2 + Bz**2 - Ex**2 - Ey**2 - Ez**2)/2"
        # self.G_expr = "-(Ex*Bx + Ey*By + Ez*Bz)"

        # self.F, self.G = [
        #     np.zeros(self.grid_shape, dtype=config.FDTYPE) for _ in range(2)
        # ]

        # if not self.channels:
        #     self.U1 = [f"(4*E{ax}*F + 7*B{ax}*G)" for ax in "xyz"]
        #     self.U2 = [f"(4*B{ax}*F - 7*E{ax}*G)" for ax in "xyz"]
        # else:
        #     self._define_channel_variables()

        self.I_ij = {
            f"{i}{j}": f"(e{i}x*U{j}_acc_x + e{i}y*U{j}_acc_y + e{i}z*U{j}_acc_z)"
            for i in range(1, 3)
            for j in range(1, 3)
        }
        for key, val in self.I_ij.items():
            self.__dict__[f"I_{key}_expr"] = val

    def create_state(field, grid):
        field = field.fields[0]
        # grid_params = GridParams(
        #     grid.kabs,
        #     grid.e1x, 
        #     grid.e1y, 
        #     grid.e1z, 
        #     grid.e2x, 
        #     grid.e2y, 
        #     grid.e2z,
        #     grid.dVk
        # )
        state = IntegratorState(
            grid.kabs,
            grid.e1x, 
            grid.e1y, 
            grid.e1z, 
            grid.e2x, 
            grid.e2y, 
            grid.e2z,
            grid.dVk,
            field.a1,
            field.a2,
            field.t0
        )
        return state

    @jax.jit
    def propagate_fields(state, t):
        # kabs, dVk = state.kabs, state.dVk
        # e1x, e1y, e1z = state.e1x, state.e1y, state.e1z
        # e2x, e2y, e2z = state.e2x, state.e2y, state.e2z
        # a1, a2, t0 = state.a1, state.a2, state.t0 

        norm_ifft = state.dVk / (2.0 * pi) ** 3
    
        phase = jnp.exp(-1j*state.kabs*c*(t-state.t0))
        
        a1t = phase * state.a1 * norm_ifft
        a2t = phase * state.a2 * norm_ifft

        # Ex = (state.e1x*a1t + state.e2x*a2t)
        # Ey = (state.e1y*a1t + state.e2y*a2t)
        # Ez = (state.e1z*a1t + state.e2z*a2t)

        # Bx = (state.e2x*a1t - state.e1x*a2t)
        # By = (state.e2y*a1t - state.e1y*a2t)
        # Bz = (state.e2z*a1t - state.e1z*a2t)

        # Ex = np.real(jnp.fft.fftn(Ex))
        # Ey = np.real(jnp.fft.fftn(Ey))
        # Ez = np.real(jnp.fft.fftn(Ez))
        # Bx = np.real(jnp.fft.fftn(Bx))
        # By = np.real(jnp.fft.fftn(By))
        # Bz = np.real(jnp.fft.fftn(Bz))

        def fft_real(expr):
            # use rfftn/irfftn if your inputs are real-valued
            return jnp.real(jnp.fft.fftn(expr, norm="backward"))

        Ex = fft_real(state.e1x * a1t + state.e2x * a2t)
        Ey = fft_real(state.e1y * a1t + state.e2y * a2t)
        Ez = fft_real(state.e1z * a1t + state.e2z * a2t)
        Bx = fft_real(state.e2x * a1t - state.e1x * a2t)
        By = fft_real(state.e2y * a1t - state.e1y * a2t)
        Bz = fft_real(state.e2z * a1t - state.e2x * a2t)
        E = jnp.stack([Ex, Ey, Ez])
        B = jnp.stack([Bx, By, Bz])

        return E, B

    @jax.jit
    def calculate_one_time_step(state, t, U1_acc, U2_acc):
        """
        Calculate the field and U terms (integrand) for one time step.
        """
        # kabs = state.kabs
        # (Ex, Ey, Ez), (Bx, By, Bz) = VacuumEmissionJax.propagate_fields(state, t)
        E, B = VacuumEmissionJax.propagate_fields(state, t)
        # (Ex, Ey, Ez), (Bx, By, Bz) = E, B

        F = ((B**2).sum(axis=0) - (E**2).sum(axis=0))/2
        G = -jnp.einsum("ijkl,ijkl->jkl", E, B)

        prefactor = jnp.exp(1j*state.kabs*c*t)
        # E = jnp.stack([Ex, Ey, Ez])
        # B = jnp.stack([Bx, By, Bz])
        
        # U1_acc[0] += jnp.fft.fftn(4*Ex*F + 7*Bx*G) * prefactor
        # U1_acc[1] += jnp.fft.fftn(4*Ey*F + 7*By*G) * prefactor
        # U1_acc[2] += jnp.fft.fftn(4*Ez*F + 7*Bz*G) * prefactor

        # U2_acc[0] += jnp.fft.fftn(4*Bx*F - 7*Ex*G) * prefactor
        # U2_acc[1] += jnp.fft.fftn(4*By*F - 7*Ey*G) * prefactor
        # U2_acc[2] += jnp.fft.fftn(4*Bz*F - 7*Ez*G) * prefactor

        U1_update = jnp.fft.fftn(4*E*F + 7*B*G, norm="backward") * prefactor
        U2_update = jnp.fft.fftn(4*B*F - 7*E*G, norm="backward") * prefactor

        # Accumulate safely
        U1_acc = U1_acc + U1_update
        U2_acc = U2_acc + U2_update

        return U1_acc, U2_acc

    def calculate_time_integral(self, t_grid, integration_method="trapezoid"):
        """
        Calculate the time integral.
        """
        # dt = t_grid[1] - t_grid[0]
        shape = [3] + list(self.grid_shape)
        U1_acc = np.zeros(shape, dtype=config.CDTYPE)
        U2_acc = np.zeros(shape, dtype=config.CDTYPE)
        # U1_acc = [np.zeros(self.grid_shape, dtype=config.CDTYPE) for _ in range(3)]
        # U2_acc = [np.zeros(self.grid_shape, dtype=config.CDTYPE) for _ in range(3)]

        if integration_method == "trapezoid":
            for _, t in enumerate(t_grid):
                # weight = 0.5 if i in end_pts else 1.
                # weight = 1
                U1_acc, U2_acc = VacuumEmissionJax.calculate_one_time_step(
                    self.integrator_state,
                    t, 
                    U1_acc, 
                    U2_acc
                )
        else:
            err_msg = (
                "integration_method should be one of ['trapezoid'] but you "
                f"passed {integration_method}"
            )
            raise NotImplementedError(err_msg)
        
        return U1_acc, U2_acc

    def calculate_amplitudes(
        self, t_grid, integration_method="trapezoid", save_path=None
    ):
        """
        Calculate the vacuum emission amplitudes and save the result.
        """
        # Allocate resources
        # self._allocate_result_arrays()
        # self._allocate_fft()
        dt = t_grid[1] - t_grid[0]
        dV = self.grid_xyz.dV

        time_integral_start = time.perf_counter()
        U1_acc, U2_acc = self.calculate_time_integral(t_grid, integration_method)
        time_integral_end = time.perf_counter()
        time_integral = time_integral_end - time_integral_start
        U1_acc_x, U1_acc_y, U1_acc_z = U1_acc
        U2_acc_x, U2_acc_y, U2_acc_z = U2_acc
        # self._free_resources()

        # Results should be in U1_acc and U2_acc
        dims = 1 / BS**3 * m_e**2 * c**3 / hbar**2 * dt * dV
        prefactor = -1j * np.sqrt(alpha * self.kabs) / (2 * pi) ** 1.5 / 45 * dims # noqa: F841
        # Next time need to be careful with f-strings and brackets
        self.S1 = ne.evaluate(
            f"prefactor * ({self.I_11_expr} - {self.I_22_expr})",
            global_dict=self.__dict__,
        ).astype(config.CDTYPE)
        self.S2 = ne.evaluate(
            f"prefactor * ({self.I_12_expr} + {self.I_21_expr})",
            global_dict=self.__dict__,
        ).astype(config.CDTYPE)
        # Save amplitudes
        if save_path:
            self.save_amplitudes(save_path)
        return time_integral

    def save_amplitudes(self, save_path):
        """
        Save the calculated amplitudes to a file.
        """
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        data = {
            "x": self.grid[0],
            "y": self.grid[1],
            "z": self.grid[2],
            "S1": self.S1,
            "S2": self.S2,
        }
        np.savez(save_path, **data)
