Usage
=====

Command line interface
----------------------

Quvac has its own small command line interface and after installation provides the following commands:

.. code-block:: bash

    quvac-simulation
    quvac-simulation-parallel
    quvac-gridscan
    quvac-optimization

All functions have the same calling signature (``-h`` for help):

.. code-block:: bash

    quvac-simulation [-h] [--input INPUT] [--output OUTPUT] [--wisdom WISDOM]


The difference is in the input file format (see Tutorials section for details):

1. ``quvac-simulation`` requires field, grid and postprocessing parameters.
2. ``quvac-simulation-parallel`` additionally requires a section about parallelization parameters.
3. ``quvac-gridscan`` requires a section about gridscan parameters.
4. ``quvac-optimization`` requires a section about optimization parameters. 


Launching scripts directly
--------------------------

Python scripts could be launched directly with (make sure to provide correct path to the script):

.. code-block:: bash

    python src/quvac/simulation.py <args>
    python src/quvac/simulation_parallel.py <args>
    python src/quvac/cluster/gridscan.py <args>
    python src/quvac/cluster/optimization.py <args>

Script arguments are the same as in the previous section:

.. code-block:: bash

    python src/quvac/simulation.py -i <input_yml_file> -o <output_dir>