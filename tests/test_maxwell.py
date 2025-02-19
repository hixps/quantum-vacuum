"""
Here we provide a test for maxwell representation of fields:
for some field configurations we compare analytic and maxwell fields
"""

import os
from pathlib import Path

import numpy as np
import pytest

from quvac.grid import setup_grids
from quvac.field.gaussian import GaussianAnalytic
from quvac.field.maxwell import MaxwellMultiple
from quvac.utils import read_yaml, write_yaml
from config_for_tests import DEFAULT_CONFIG_PATH


def get_intensity(field, t):
    E, B = field.calculate_field(t=t)
    E, B = [np.real(Ex) for Ex in E], [np.real(Bx) for Bx in B]
    I = (E[0]**2 + E[1]**2 + E[2]**2 + B[0]**2 + B[1]**2 + B[2]**2)/2
    return I


def test_maxwell_gauss():
    ini_data = read_yaml(DEFAULT_CONFIG_PATH)

    field_params = ini_data["fields"]["field_1"]
    grid_params = ini_data["grid"]

    grid_xyz, grid_t = setup_grids([field_params], grid_params)
    grid_xyz.get_k_grid()
    
    gauss = GaussianAnalytic(field_params, grid_xyz)
    gauss_mw = MaxwellMultiple([field_params], grid_xyz)

    t = 0.
    I = get_intensity(gauss, t)
    I_mw = get_intensity(gauss_mw, t)

    I = np.clip(I, a_min=I.max()*1e-10, a_max=None)
    I_mw = np.clip(I, a_min=I_mw.max()*1e-10, a_max=None)

    err_msg = "Maxwell field does not match analytic field (up to 10% relative error)"
    assert np.allclose(I, I_mw, rtol=1e-1), err_msg