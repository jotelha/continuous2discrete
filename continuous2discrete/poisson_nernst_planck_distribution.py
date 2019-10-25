""" Calculate ionic concentrations consistent with general
Poisson-Nernst-Planck (PNP) equations.

Copyright 2019 IMTEK Simulation
University of Freiburg

Authors:
  Johannes Hoermann <johannes.hoermann@imtek-uni-freiburg.de>
"""
import numpy as np
import scipy.constants as sc


def ionic_strength(c,z):
  """Compute ionic strength from charges and concentrations

  Arguments:
  c: bulk concentrations [concentration unit, i.e. mol m^-3]
  z: number charges [number charge unit, i.e. 1]

  Returns:
  float: I, ionic strength ( 1/2 * sum(z_i^2*c_i) ) [concentration unit, i.e. mol m^-3]
  """
  return 0.5*np.sum( np.square(z) * c )

def debye(c, z,
  T=298.15,
  relative_permittivity=79,
  vacuum_permittivity=sc.epsilon_0,
  R = sc.value('molar gas constant'), F=sc.value('Faraday constant'),
  ):
  """Calculate the Debye length (in SI units per default).
  The Debye length indicates at which distance a charge will be screened off.

  Arguments:
  c:            bulk concentrations of each ionic species [mol/m^3]
  z:            charge of each ionic species [1]
  T:            temperature of the solution [K] (default: 298.15)
  relative_permittivity:
                relative permittivity of the ionic solution [1] (default: 79)
  vacuum_permittivity:
                vacuum permittivity [F m^-1] (default: 8.854187817620389e-12 )
  R:            molar gas constant [J mol^-1 K^-1] (default: 8.3144598)
  F:            Faraday constant [C mol^-1] (default: 96485.33289)

  Returns:
  float: lambda_D, Debye length, sqrt( epsR*eps*R*T/(2*F^2*I) ) [m]
  """

  I = ionic_strength(c,z)
  return np.sqrt(relative_permittivity*vacuum_permittivity*R*T/(2.0*F**2*I))

class PoissonNernstPlanckSystem:
  def I(self): # ionic strength
    return 0.5*np.sum( np.square(self.z) * self.c )

  def lambda_D(self):
    return np.sqrt(
      self.relative_permittivity*self.vacuum_permittivity*self.R*self.T/(
        2.0*self.F**2*self.I() ) )

  # default 0.1 mM NaCl aqueous solution
  def __init__(self,
    c = [0.1,0.1],
    z = [+1,-1],
    L = 100e-9, # 100 nm
    T = 298.15,
    delta_u = 1, # potential difference [V]
    relative_permittivity = 79,
    vacuum_permittivity   = sc.epsilon_0,
    R = sc.value('molar gas constant'),
    F = sc.value('Faraday constant') ):

    self.c  = c # concentrations
    self.z  = z # number charges
    self.T  = T # temperature
    self.L  = L # 1d domain size
    self.delta_u = delta_u # potential difference

    self.relative_permittivity = relative_permittivity
    self.vacuum_permittivity   = vacuum_permittivity
    # R = N_A * k_B
    # (universal gas constant = Avogadro constant * Boltzmann constant)
    self.R                     = R
    self.F                     = F

    self.f                     = F / (R*T) # for convenience

    # scaled units for dimensionless formulation

    # length unit chosen as Debye length lambda
    self.l_unit = self.lambda_D()

    # concentration unit is ionic strength
    self.c_unit =  self.I()

    # no time unit for now, only steady state
    # self.t_unit = l_unit**2 / Dn # fixes Dn_scaled = 1

    # u = psi * q / kB T
    # u_unit = kB * T / q
    self.u_unit = self.R * self.T / self.F # thermal voltage

    # domain
    self.L_scaled = self.L / self.l_unit

    # bulk conectrations
    self.c_scaled = self.c / self.c_unit

    # potential difference
    self.delta_u_scaled = self.delta_u / self.u_unit

    # relaxation time
    # tau_scaled   = tau / t_unit

    # should be 1
    # Dn_scaled    = Dn * t_unit / l_unit**2
