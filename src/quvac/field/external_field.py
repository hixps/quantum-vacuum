'''
This script provides uniform ExternalField class to unite all
participating fields in one interface
'''
import os

import numpy as np

from quvac.field.abc import Field
from quvac.field.gaussian import GaussianAnalytic
from quvac.field.maxwell import MaxwellMultiple


class ExternalField(Field):
    '''
    Class to unite several participating fields under
    one interface

    Parameters
    ----------
    fields_params: list of dicts (field_params)
        External fields
    grid: (1d-np.array, 1d-np.array, 1d-np.array)
        xyz spatial grid to calculate fields on 
    '''
    def __init__(self, fields_params, grid, nthreads=None):
        self.fields = []
        self.grid = grid
        self.grid.get_k_grid()

        self.nthreads = nthreads if nthreads else os.cpu_count()

        maxwell_params = [params for params in fields_params
                          if params['field_type'].endswith('maxwell')]
        new_params = [params for params in fields_params
                      if not params['field_type'].endswith('maxwell')]
        if maxwell_params:
            new_params.append(maxwell_params)

        for field_params in new_params:
            self.setup_field(field_params)

    def setup_field(self, field_params):
        if isinstance(field_params, list):
            field_type = 'maxwell'
        else:
            field_type = field_params["field_type"]

        match field_type:
            case "paraxial_gaussian_analytic":
                field = GaussianAnalytic(field_params, self.grid)
            case "maxwell":
                field = MaxwellMultiple(field_params, self.grid, nthreads=self.nthreads)
            case _:
                raise NotImplementedError(f"We do not support '{field_type}' field type")
        self.fields.append(field)
            
    def calculate_field(self, t, E_out=None, B_out=None):
        for field in self.fields:
            E_out, B_out = field.calculate_field(t, E_out=E_out, B_out=B_out)
        return E_out, B_out

