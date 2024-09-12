'''
This script implements calculation of vacuum emission integral
It is planned to add support for two versions:
    - Calculation of total vacuum emission signal for given field configuration
    - Separation of fields into pump and probe with subsequent calculation of probe channel signal
'''
'''
TODO:
    - 
'''

import numpy as np
import numexpr as ne
from scipy.constants import pi, c, epsilon_0
import pyfftw


class VacuumEmission(object):
    '''
    Calculator of Vacuum Emission amplitude from given fields

    Field parameters
    ----------------
    field: quvac.Field
        External fields
    '''
    def __init__(self, field):
        self.field = field
        self.allocate_fields()
        self.allocate_result_arrays()
        self.allocate_fft()
        self.setup_k_grid()

        angles = "theta phi beta".split()
        # for angle in angles:
        #     self.__dict__[angle] = self.field.__dict__[angle]
        
        # # Define two perpendicular polarizations
        # self.e1 = np.array([np.cos(self.phi)*np.cos(self.theta),
        #                     np.sin(self.phi)*np.cos(self.theta),
        #                     -np.sin(self.theta)])
        # self.e2 = np.array([-np.sin(self.phi),np.cos(self.phi),0])

        # Define symbolic expressions to evaluate later
        self.F = F = "0.5 * (Bx**2 + By**2 + Bz**2 - Ex**2 - Ey**2 - Ez**2)"
        self.G = G ="-(Ex*Bx + Ey*By + Ez*Bz)"
        self.U1 = [f"4*E{ax}*{F} + 7*B{ax}*{G}" for ax in "xyz"]
        self.U2 = [f"-4*B{ax}*{F} + 7*E{ax}*{G}" for ax in "xyz"]
        # self.I_ij = {f"{i}{j}": f"e{i}[0]*U{j}[0] + e{i}[1]*U{j}[1] + e{i}[2]*U{j}[2]"
        #              for i in range(2) for j in range(2)}
        # for key,val in self.I_ij.items():
        #     self.__dict__[f"I_{key}"] = val
        # self.I = "cos(beta_p)*(I_11 - I_22) + sin(beta_p)*(I_12 + I_21)"
    
    def allocate_fields(self):
        self.E_out = (np.zeros(self.field.grid_shape) for _ in range(3))
        self.B_out = (np.zeros(self.field.grid_shape) for _ in range(3))
        self.Ex, self.Ey, self.Ez = self.E_out
        self.Bx, self.By, self.Bz = self.B_out

    def allocate_result_arrays(self):
        self.U1_acc = (np.zeros(self.field.grid_shape) for _ in range(3))
        self.U2_acc = (np.zeros(self.field.grid_shape) for _ in range(3))
        self.U1_acc_x, self.U1_acc_y, self.U1_acc_z = self.U1_acc
        self.U2_acc_x, self.U2_acc_y, self.U2_acc_z = self.U2_acc

    def allocate_fft(self):
        self.tmp = [pyfftw.zeros_aligned(self.field.grid_shape,  dtype='complex128') for _ in range(3)]
        # Add number of threads
        self.tmp_fftw = [pyfftw.FFTW(a, a, axes=(0, 1, 2),
                                    direction='FFTW_FORWARD',
                                    flags=('FFTW_MEASURE', ),)
                        for a in self.tmp]
    
    def setup_k_grid(self):
        for i,ax in enumerate('xyz'):
            grid = self.field.grid[i]
            self.__dict__[f'd{ax}'] = step = grid[1] - grid[0]
            self.__dict__[f'k{ax}'] = 2*pi*np.fft.fftfreq(grid.size, step)
        self.kmeshgrid = np.meshgrid(self.kx, self.ky, self.kz, indexing='ij', sparse=True)
        k_dict = {f'k{ax}': self.kmeshgrid[i] for i,ax in enumerate('xyz')}
        self.kabs = ne.evaluate("sqrt(kx**2 + ky**2 + kz**2)", local_dict=k_dict)
        for i,ax in enumerate('xyz'):
            self.__dict__[f'k{ax}_unit'] = self.kmeshgrid[i] / self.kabs

    def free_resources(self):
        del self.E_out, self.B_out
        del self.tmp, self.tmp_fftw

    def calculate_one_time_step(self, t, weight=1):
        # Calculate fields
        self.field.calculate_field(t, E_out=self.E_out, B_out=self.B_out)
        # Evaluate U1 and U2 expressions
        ax = 'xyz'
        for idx,U_expr in enumerate([self.U1, self.U2]):
            for i,expr in enumerate(U_expr):
                ne.evaluate(expr, global_dict=self.__dict__, out=self.tmp[i])
                self.tmp_fftw[i].execute()
                self.U = self.tmp[i]
                ne.evaluate(f"U{idx+1}_acc_{ax[i]} + U*exp(1j*kabs*t)*dt*weight",
                            global_dict=self.__dict__)

    def calculate_time_integral(self, t_grid, integration_method="trapezoid"):
        self.dt = t_grid[1] - t_grid[0]
        if integration_method == "trapezoid":
            end_pts = (0,len(t_grid)-1)
            for i,t in enumerate(t_grid):
                if i in end_pts:
                    weight = 0.5
                self.calculate_one_time_step(t, weight=weight)
        else:
            raise NotImplementedError(f"""integration_method should be one of ['trapezoid'] but 
                                      you passed {integration_method}""")

    def calculate_vacuum_current(self, t_grid, integration_method="trapezoid",
                                 filename="current.npz"):
        self.calculate_time_integral(t_grid, integration_method)
        self.free_resources()
        # Results should be in U1_acc and U2_acc
        current = [0, 0, 0]
        current[0] = ne.evaluate("-U1_acc_x + ky_unit*U2_acc_z - kz_unit*U2_acc_y",
                                 global_dict=self.__dict__)
        current[1] = ne.evaluate("-U1_acc_y - kx_unit*U2_acc_z + kz_unit*U2_acc_x",
                                 global_dict=self.__dict__)
        current[1] = ne.evaluate("-U1_acc_z + kx_unit*U2_acc_y - ky_unit*U2_acc_x",
                                 global_dict=self.__dict__)
        np.savez(filename, jx=current[0], jy=current[1], jz=current[2])
        
        




