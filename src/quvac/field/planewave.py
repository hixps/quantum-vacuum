"""
Analytic expression for plane wave.
"""

import numexpr as ne
from scipy.constants import c, pi

from quvac.field.abc import ExplicitField


class PlaneWave(ExplicitField):
    """
    Analytic expression for paraxial Gaussian beam.

    Parameters
    ----------
    field_params : dict
        Dictionary containing the field parameters. Required keys are:
            - 'theta' : float
                Polar angle of k-vector (in degrees).
            - 'phi' : float
                Azimuthal angle of k-vector (in degrees).
            - 'beta' : float
                Polarization angle (in degrees).
            - 'lam' : float
                Wavelength of the pulse.
            - 'tau' : float
                Duration.
            - 'phase0' : float
                Phase delay at focus.
            - 'E0' : float, optional
                Amplitude (either E0 or W is required).
            - 'W' : float, optional
                Energy (either E0 or W is required).
    grid : quvac.grid.GridXYZ
        Spatial and grid.
    """
    def __init__(self, field_params, grid):
        super().__init__(field_params, grid)

        self.B0 = self.E0 / c

        # Set up correct field amplitude
        if "W" in field_params:
            self.check_energy()

    def check_energy(self):
        """
        Check and adjust the field energy.
        """
        self._check_energy()

        if self.modify_energy:
            self.E0 *= self.W_correction
            self.B0 = self.E0 / c

    def calculate_field(self, t, E_out=None, B_out=None, mode="real"):
        """
        Calculates the electric and magnetic fields at a given time step.
        """
        k = 2.0 * pi / self.lam # noqa: F841
        self.psi_plane = ne.evaluate("(omega*(t-t0) - k*z + phase0)",
                                      global_dict=self.__dict__)

        Et = ne.evaluate(
            "E0 * exp(-(2.*psi_plane/(omega*tau))**2) * exp(-1.j*psi_plane)",
            global_dict=self.__dict__,
        )

        self.Ex = self.By = Et.copy()
        self.Ey = self.Ez = self.Bx = self.Bz = 0.0

        if mode == "real":
            self.convert_fields_to_real()

        E_out, B_out = self.rotate_fields_back(E_out, B_out, mode)
        return E_out, B_out