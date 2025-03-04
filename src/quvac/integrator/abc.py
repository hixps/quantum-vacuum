"""
Abstract interfaces for different integrators.

.. warning::
    Currently not used...
"""

from abc import ABC, abstractmethod


class Integrator(ABC):
    """
    Abstract integrator class.
    """
    @abstractmethod
    def calculate_amplitudes(self, t_grid):
        """ """
        ...
