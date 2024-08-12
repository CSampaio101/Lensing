import sys, os
import numpy as np



import matplotlib as mpl
import matplotlib.pyplot as plt

#plt.rcParams['text.usetex'] = True

from astropy.cosmology import Planck18
from astropy import units
from IPython.display import clear_output

from functools import partial


from scipy import special

Gpc_in_km = 3.08568e+22

# Present day critical density
rho_c = Planck18.critical_density0.to('M_sun/km^3').value # Solar masses / km^3

def get_halo_information(M, c):
    rs = ((3*M)/(800*c**3*np.pi*rho_c))**(1./3.) # km
    rMax = c*rs # km
    rho_s = ((1 + c)*M)/(16.*np.pi*rs**3*(-c + np.log(1 + c) + c*np.log(1 + c))) # Solar masses/ km^3

    return rs, rMax, rho_s # [km, km, Solar Masses / km^3]

def g_truncated_nfw(x, c, asymptotic_switch=1e5):
    """
    A function needed for the Fourier transform of the truncated NFW density profile as a function of x=c*r_s, with the truncation radius at c*r_s
    """

    if c*x < asymptotic_switch:
        si_x, ci_x = special.sici(x)
        si_1_plus_cx, ci_1_plus_cx = special.sici((1 + c) * x)
        return -np.sin(c * x)/((1 + c) * x) \
            - np.cos(x) * (ci_x - ci_1_plus_cx) \
            - np.sin(x) * (si_x - si_1_plus_cx)
    else:
        return (1.-np.cos(c*x)/(1 + c)**2) / x**2

def rhoTildeSq(q, M, c):
    """
    Fourier transform of the truncated NFW profile
    Args:
        q: Fourier wavenumber, conjugate to the physical length r
        M: virial mass of the minihalo in solar masses
        c: concentration parameter
    """

    rs, rMax, rho_s = get_halo_information(M, c)
    x = q * rs

    result = (16 * np.pi * rs**3 * rho_s * g_truncated_nfw(x, c))**2

    return result



def Pkappa(q, params):
    M = params['M']
    c = params['c']
    DL = params['DL'] # Gpc
    DS = params['DS'] # Gpc
    DLS = params['DLS'] # Gpc
    f = params['dm_mass_fraction']



    Sigma_cr = 1.74e-24 / (DL*DLS/DS) # [M_Sun / km^2]
    Sigma_cl = 0.83*Sigma_cr # [M_Sun / km^2]

    return f * Sigma_cl / Sigma_cr**2  * rhoTildeSq(q, M=M, c = c)/M

def Pkappa_angular(l,params):
    DL = params['DL']*Gpc_in_km
    return(Pkappa(l/DL, params)/DL**2)


params = {'M':1e-6, 'c': 10, 'DL': 1.35, 'DS':1.79, 'DLS':0.95, 'dm_mass_fraction':1}
qPerp = np.geomspace(1e-20, 1e11, 1000)
k_rs = get_halo_information(params['M'], params['c'])[0]*qPerp
pkappa = [q**2 / 2 / np.pi * Pkappa(q, params) for q in qPerp]
plt.plot(k_rs, pkappa, label = '$c = 10$')

params = {'M':1e-6, 'c': 100, 'DL': 1.35, 'DS':1.79, 'DLS':0.95, 'dm_mass_fraction':1}
qPerp = np.geomspace(1e-20, 1e11, 1000)
pkappa = [q**2 / 2 / np.pi * Pkappa(q,params) for q in qPerp]
k_rs = get_halo_information(params['M'], params['c'])[0]*qPerp
plt.plot(k_rs, pkappa, label = '$c = 100$')

params = {'M':1e-6, 'c': 1000, 'DL': 1.35, 'DS':1.79, 'DLS':0.95, 'dm_mass_fraction':1}
qPerp = np.geomspace(1e-20, 1e11, 1000)
pkappa = [q**2 / 2 / np.pi * Pkappa(q, params) for q in qPerp]
k_rs = get_halo_information(params['M'], params['c'])[0]*qPerp
plt.plot(k_rs, pkappa, label = '$c = 1000$')

# plt.ylim(float(pkappa[0]), 1e-5)
# plt.xlim(k_rs[0], 1e11)
# plt.axvline(1, color = 'black', ls = '--')
# plt.axvline(1e6, color = 'black', ls = ':')
# plt.xscale('log')
# plt.yscale('log')
# plt.ylabel('$q^2 P_\kappa/(2\pi)$', fontsize = 20)
# plt.xlabel('$q r_s$', fontsize = 20)
# plt.legend()
# plt.tight_layout()
# plt.show()

for c in [10, 100, 1000]:
    params = {'M':1e-6, 'c': c, 'DL': 1.35, 'DS':1.79, 'DLS':0.95, 'dm_mass_fraction':1}
    rs = get_halo_information(params['M'], params['c'])[0]
    lPerp = np.geomspace(1e-10*params['DL']*Gpc_in_km/rs, 1e10*params['DL']*Gpc_in_km/rs, 1000)
    l_rs_over_DL = lPerp*rs/(params['DL']*Gpc_in_km)
    pkappa = [l**2 / 2 / np.pi * Pkappa_angular(l,params) for l in lPerp]
    plt.plot(l_rs_over_DL, pkappa, label = '$c = '+str(c)+'$')


# plt.ylim(float(pkappa[0]), 1e-5)
# #plt.xlim(l_rs_over_DL[0], 1e11)
# plt.axvline(1, color = 'black', ls = '--')

# plt.xscale('log')
# plt.yscale('log')
# plt.ylabel('$\ell^2 P_\kappa(\ell)/(2\pi)$', fontsize = 20)
# plt.xlabel('$\ell r_s / D_L$', fontsize = 20)
# plt.legend()
# plt.tight_layout()

# #plt.savefig("graphics/angular_convergence_ps.pdf", bbox_inches='tight')


# plt.show()