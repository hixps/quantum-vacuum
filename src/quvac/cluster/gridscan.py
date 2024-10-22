'''
Script to run gridscan simulations on cluster with Slurm
'''
import itertools
import os
from pathlib import Path
from copy import deepcopy

import numpy as np
import submitit

from quvac.cluster.config import DEFAULT_SUBMITIT_PARAMS
from quvac.simulation import quvac_simulation
from quvac.utils import write_yaml, read_yaml


def create_parameter_grids(variables):
    '''
    Create a grid from (start, end , npts) specified for each parameter
    '''
    variables_grid = {}
    for category_key,category in variables.items():
        variables_grid[category_key] = {}
        for param_key,param in category.items():
            start, end, npts = param
            param_grid = list(np.linspace(start, end, npts))
            variables_grid[category_key][param_key] = param_grid
    return variables_grid


def restructure_variables_grid(variables):
    variables_grid = {}
    for key,val in variables['fields'].items():
        variables[key] = val
    variables.pop('fields')

    for category_key,category in variables.items():
        for param_key,param in category.items():
            new_key = f'{category_key}:{param_key}'
            variables_grid[new_key] = param
    
    param_names = list(variables_grid.keys())
    param_names = [param.split(':') for param in param_names]
    param_grids = list(variables_grid.values())
    return param_names, param_grids


def create_ini_files_for_gridscan(ini_default, param_names,
                                  param_grids, save_path):
    ini_files = []
    for parametrization in itertools.product(*param_grids):
        ini_current = deepcopy(ini_default)
        name_local = ''
        for name,param in zip(param_names,parametrization):
            category, param_name = name
            if category.startswith('field'):
                ini_current['field'][category][param_name] = param
            else:
                ini_current[category][param_name] = param
            param_str = str(param) if isinstance(param, int) else f'{param:.2f}'
            param_str = f'{category}:{param_name}_{param_str}'
            name_local = '#'.join(name_local, param_str)
        save_path_local = os.path.join(save_path, name_local, 'ini.yml')
        write_yaml(save_path_local, ini_current)
        ini_files.append(save_path_local)
    return ini_files


def cluster_gridscan(ini_file, variables_file, save_path=None):
    '''
    Launch a gridscan of quvac simulation for given default <ini>.yml file
    and <variables>.yml file

    Parameters:
    -----------
    ini_file: str (format <path>/<file_name>.yaml)
        Default initial configuration file containing all simulation parameters.
        These parameters (apart from variables) would remain the same for the 
        whole gridscan.
        Note: This file might also contain parameters for cluster computation
    variables_file: str (format <path>/<file_name>.yaml)
        File containing all parameters to vary
    save_path: str
        Path to save simulation results to
    '''
    # Check that ini file and save_path exists
    err_msg = f"{ini_file} or {variables_file} is not a file or does not exist"
    assert os.path.isfile(ini_file) and os.path.isfile(variables_file), err_msg
    if save_path is None:
        save_path = os.path.dirname(ini_file)
    if not os.path.exists(save_path):
        Path(save_path).mkdir(parents=True, exist_ok=True)

    ini_default = read_yaml(ini_file)
    cluster_params = ini_default.get('cluster', {})
    if cluster_params:
        ini_default.pop('cluster')
    variables = read_yaml(variables_file)

    # Create parameter grids if required
    if variables.get('create_grids', False):
        variables_grid = create_parameter_grids(variables)
    else:
        variables_grid = variables

    # Restructure variables dict
    param_names, param_grids = restructure_variables_grid(variables_grid)

    # Create yaml files for the grid scan
    ini_files = create_ini_files_for_gridscan(ini_default, param_names,
                                              param_grids, save_path)

    # Set up scheduler
    cluster = cluster_params.get('cluster', 'local')
    log_folder = os.path.join(save_path, 'submitit_logs')
    sbatch_params = cluster_params.get('sbatch_params', DEFAULT_SUBMITIT_PARAMS)
    max_jobs = cluster_params.get('max_jobs', 5)
    executor = submitit.AutoExecutor(folder=log_folder, cluster=cluster,
                                     slurm_array_parallelism=max_jobs, **sbatch_params)
    jobs = executor.map_array(quvac_simulation, ini_files)


    



    

