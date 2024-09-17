'''
Here we provide utility functions to calculate analytic 
scalings from articles
'''

import numpy as np
from scipy.constants import pi, c, alpha, m_e, hbar


W_e = m_e * c**2 # electron's rest energy
lam_C = hbar / (m_e*c)


def get_two_paraxial_scaling(fields):
    theta_c = (fields[1]['theta'] - fields[0]['theta']) * pi / 180
    beta = (fields[1]['beta'] - fields[0]['beta']) * pi / 180
    # assert beta != 0, "This formula doesn't work for 0 relative polarization angle"
    # tau1, tau2 = fields[0]['tau'], fields[1]['tau']
    # w01, w02 = fields[0]['w0'], fields[1]['w0']
    # W1, W2 = fields[0]['W'], fields[1]['W']

    # # Define additional parameters
    # T1 = tau1*np.sqrt(1+2*(tau1/tau2)**2)
    # T2 = tau2*np.sqrt(1+0.5*(tau2/tau1)**2)
    # w1 = w01*np.sqrt(1+2*(w01/w02)**2)
    # w2 = w02*np.sqrt(1+0.5*(w02/w01)**2)
    # sin_term = T1*T2*c**2/(w1*w2)*np.sin(theta_c)**2
    # cos_term = 4*(1 - np.cos(theta_c))**2
    # fields[0]['T'] = T1
    # fields[1]['T'] = T2
    # fields[0]['w'] = w1
    # fields[1]['w'] = w2


    omega = 2*pi*c/fields[1]['lam']
    N_signal, N_perp = 0, 0
    # Iterate through pump/probe channels
    for field1,field2 in [fields, fields[::-1]]:
        tau1, tau2 = field1['tau'], field2['tau']
        w01, w02 = field1['w0'], field2['w0']
        W1, W2 = field1['W'], field2['W']
        # T1, T2 = field1['T'], field2['T']
        # w1, w2 = field1['w'], field2['w']

        # # Define additional parameters
        T1 = tau1*np.sqrt(1+2*(tau1/tau2)**2)
        T2 = tau2*np.sqrt(1+0.5*(tau2/tau1)**2)
        w1 = w01*np.sqrt(1+2*(w01/w02)**2)
        w2 = w02*np.sqrt(1+0.5*(w02/w01)**2)
        sin_term = T1*T2*c**2/(w1*w2)*np.sin(theta_c)**2
        cos_term = 4*(1 - np.cos(theta_c))**2

        # Calculate N_signal and N_perp
        theta_term = (1 - np.cos(theta_c))**4 / np.sqrt(cos_term + sin_term)
        theta_term *= 1/np.sqrt((T1/tau1*w01/w1)**2*cos_term + sin_term)
        prefactor = 8*pi**2/225 * (alpha/pi)**4 * W1*W2**2/W_e**3 * (hbar*omega/W_e)
        prefactor *= T1/tau1 * w1/w01 * lam_C**4/(w1*w2)**2 / 9
        N_perp += prefactor * theta_term * 9*np.sin(2*beta)**2
        N_signal += prefactor * theta_term * (130 - 66*np.cos(2*beta))
    return N_signal, N_perp

