'''
Here we provide a test for vacuum emission integrator:
    - Given two colliding paraxial gaussians we calculate the total
    vacuum signal and compare it with analytic calculation
'''

import numpy as np

from quvac.paraxial_gaussian import ParaxialGaussianAnalytic
from quvac.field import ExternalField
from quvac.vacuum_emission import VacuumEmission

