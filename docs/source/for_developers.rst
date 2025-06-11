For developers
==============

A collection of remarks about code.

Core libraries
--------------

- Regular arithmetic and matrix operations are performed with `numexpr <https://numexpr.readthedocs.io/en/latest/user_guide.html>`_,

- 3d FFTs with `pyfftw <https://pyfftw.readthedocs.io/en/latest/>`_,

- Bayesian optimization with `ax <https://ax.dev/docs/tutorials/quickstart/>`_,

- Job submission to Slurm with `submitit <https://github.com/facebookincubator/submitit>`_,

- Documentation generation with `sphinx <https://www.sphinx-doc.org/en/master/>`_ (automatic API generation with `sphinx-autoapi <https://sphinx-autoapi.readthedocs.io/en/latest/>`_),

- For tests `pytest <https://docs.pytest.org/en/stable/>`_,

- For code style `ruff <https://docs.astral.sh/ruff/>`_.

Implementation details
----------------------

The vacuum emission integral is a four-dimensional (4D) space-time integral. The time component of the integral 
is computed using the rectangle integration rule. For the spatial components, Fast Fourier Transforms (FFTs) are used.
Optionally (``simulation_parallel.py`` script), to speed up the process, the time grid is divided into segments, and each segment 
is assigned to a separate job for parallel computation.

quvac supports lower-precision computations (``float32/complex64`` instead of default ``float64/complex128``) which allows to save
memory. 

.. warning::
    However, the implementation is not totally consistent since ``numexpr`` doesn't support ``complex64``, that's why for some computations
    some of the arrays are converted to higher-precision ``complex64 -> complex128`` and back-converted after the computation 
    ``complex128 -> complex64`` which is not optimal.

