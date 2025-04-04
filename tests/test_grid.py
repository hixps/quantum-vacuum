"""
Collection of tests for numerical grid creation.
"""

import numpy as np
import pytest

from quvac.grid import GridXYZ, get_ek, get_pol_basis


@pytest.mark.parametrize(
    ("theta", "phi", "expected_ek"),
    [
        (0,0,np.array([0,0,1])),
        (90,0,np.array([1,0,0])),
        (90,90,np.array([0,1,0]))
    ],
)
def test_ek_and_pol_basis(theta, phi, expected_ek, expected_e1, expected_e2):
    theta, phi = np.radians(theta), np.radians(phi)
    ek = get_ek(theta, phi)
    e1, e2 = get_pol_basis(theta, phi)
    assert np.allclose(ek, expected_ek)
    assert np.allclose(e1, expected_e1)
    assert np.allclose(e2, expected_e2)

    