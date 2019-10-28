""" Calculate ionic concentrations consistent with general
Poisson-Nernst-Planck (PNP) equations.

Copyright 2019 IMTEK Simulation
University of Freiburg

Authors:
  Johannes Hoermann <johannes.hoermann@imtek-uni-freiburg.de>
"""
import logging

import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
# Drücke nicht-linearen Teil der Transportgleichung (genauer, des Flusses) über
# Bernoulli-Funktionen
#
# $$ B(x) = \frac{x}{\exp(x)-1} $$
#
# aus. Damit wir in der Nähe von 0 nicht "in die Bredouille geraten", verwenden
# wir hier lieber die Taylorentwicklung. In der Literatur (Selbherr, S. Analysis
# and Simulation of Semiconductor Devices, Spriger 1984) wird eine noch
# aufwendigere stückweise Definition empfohlen, allerdings werden wir im
# Folgenden sehen, dass unser Ansatz für dieses stationäre Problem genügt.
def B(x):
    return np.where( np.abs(x) < 1e-9,
        1 - x/2 + x**2/12 - x**4/720, # Taylor
        x / ( np.exp(x) - 1 ) )

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

# "lazy" Ansatz for approximating Jacobian
def jacobian(f, x0, dx=np.NaN):
    """Naive way to construct N x N Jacobin Fij from N-valued function
    f of N-valued vector x0."""
    N = len(x0)
    # choose step as small as possible
    if np.isnan(dx).any():
        res = np.finfo('float64').resolution
        dx = np.abs(x0) * np.sqrt(res)
        dx[ dx < res ] = res

    if np.isscalar(dx):
        dx = np.ones(N) * dx

    F = np.zeros((N,N)) # Jacobian Fij

    # convention: dFi_dxj
    # i are rows, j are columns
    for j in range(N):
        dxj = np.zeros(N)
        dxj[j] = dx[j]

        F[:,j] =  (f(x0 + dxj) - f(x0)) / dxj[j]

    return F

class PoissonNernstPlanckSystem:

  e      = 1e-10 # Newton solver default tolerance
  maxit  = 200 # Newton solver maximum iterations
  output = False
  outfreq = 1

  def newton(self,f,xij):
    # xij, e=1e-10, maxit=200, output=False, outfreq=1
    #N = len(xij)
    Ni = self.Ni
    N = Ni -1

    self.logger.debug('Newton solver, grid points N = {:d}'.format(N))
    self.logger.debug('Newton solver, tolerance e = {:> 8.4g}'.format(self.e))
    self.logger.debug('Newton solver, maximum number of iterations M = {:d}'.format(self.maxit))

    i = 0
    delta_rel = 2*self.e

    if self.output:
        fig = plt.figure(figsize=(16,15))
        ax1 = plt.subplot(221)
        ax1.set_ylabel('u')
        ax1.set_xlabel('x')
        ax1.set_title('potential u')

        ax2 = plt.subplot(322)
        ax2.set_ylabel('n')
        ax2.set_xlabel('x')
        ax2.set_title('density n')

        ax3 = plt.subplot(325)
        ax3.set_ylabel('du')
        ax3.set_xlabel('x')
        ax3.set_title('step du')

        ax4 = plt.subplot(326)
        ax4.set_ylabel('dn')
        ax4.set_xlabel('x')
        ax4.set_title('step dn')

    self.logger.info("Convergence criterion: norm(dx) < {:4.2e}".format(self.e))

    self.convergenceStepAbsolute = np.zeros(self.maxit)
    self.convergenceStepRelative = np.zeros(self.maxit)
    self.convergenceResidualAbsolute = np.zeros(self.maxit)

    dxij = np.zeros(N)
    while delta_rel > self.e and i < self.maxit:
        self.logger.info('*** Newton solver iteration {:d} ***'.format(i))
        J = jacobian(f, xij)
        rank = np.linalg.matrix_rank(J)
        self.logger.debug('    Jacobian rank {:d}'.format(rank))

        if rank < N:
            if self.output:
                self.logger.warn("Singular jacobian of rank"
                      + "{:d} < {:d} at step {:d}".format(
                      rank, N, i ))
            break

        F = f(xij)
        invJ = np.linalg.inv(J)

        dxij = np.dot( invJ, F )

        delta = np.linalg.norm(dxij)
        delta_rel = delta / np.linalg.norm(xij)

        xij -= dxij

        normF = np.linalg.norm(F)

        self.logger.debug('    convergence norm(dxij), absolute {:> 8.4g}'.format(delta))
        self.logger.debug('    convergence norm(dxij), realtive {:> 8.4g}'.format(delta_rel))
        self.logger.debug('          residual norm(F), absolute {:> 8.4g}'.format(normF))

        self.convergenceStepAbsolute[i] = delta
        self.convergenceStepRelative[i] = delta_rel
        self.convergenceResidualAbsolute[i] = normF

        if i % self.outfreq == 0 and self.output:
            self.logger.info("Step {:4d}: norm(dx)/norm(x) = {:4.2e}, norm(dx) = {:4.2e}, norm(F) = {:4.2e}".format(
                i, delta_rel, delta, normF) )
            duij = dxij[:Ni]
            dnij = dxij[Ni:]
            uij = xij[:Ni]
            nij = xij[Ni:]
            ax1.plot(self.X, uij, '-', label='u Step {:2d}'.format(i))
            ax2.plot(self.X, nij, '-', label='n Step {:2d}'.format(i))
            ax3.plot(self.X, duij, '-', label='du Step {:2d}'.format(i))
            ax4.plot(self.X, dnij, '-', label='dn Step {:2d}'.format(i))
            ax1.legend(loc='best')
            ax2.legend(loc='best')
            ax3.legend(loc='best')
            ax4.legend(loc='best')

        i += 1

    if i == self.maxit and self.output:
        self.logger.warn("Maximum number of iterations reached")

    if self.output:
        fig.tight_layout()

        self.logger.info("Ended after {:d} steps.".format(i))
        fig = plt.figure(figsize=(16,10))
        ax1 = plt.subplot(221)
        ax1.set_ylabel(r'\epsilon = $\frac{|x_j - x_{j-1}|}{|x_{j-1}}$')
        ax1.set_xlabel('j')
        ax1.set_title('step convergence, relative')
        ax2 = plt.subplot(222)
        ax2.set_ylabel('\Epsilon = $|x_j - x_{j-1}|')
        ax2.set_xlabel('j')
        ax2.set_title('step convergence, absolute')
        ax3 = plt.subplot(223)
        ax3.set_ylabel('$R = F(x_j)$')
        ax3.set_xlabel('j')
        ax3.set_title('residue convergence, absolute')
        ax1.plot(self.convergenceStepRelative[:i])
        ax2.plot(self.convergenceStepAbsolute[:i])
        ax3.plot(self.convergenceResidualAbsolute[:i])
        fig.tight_layout()

    # convergenceStepAbsolute[:i], convergenceStepRelative[:i], convergenceResidualAbsolute[:i]
    return xij

  def solve(self, N = 1000):
    # indices
    self.Ni = N+1
    I = np.arange(N+1)

    self.logger.info('discretization segments N:    {:> 8d}'.format(N))
    self.logger.info('grid points Ni:               {:> 8d}'.format(self.Ni))
    #N      = 1000 # 1000 segments, 1001 points

    # discretization
    dx      = self.L_scaled / N # spatial step
    self.dx = dx
    # maximum time step allowed
    # (irrelevant for our steady state case)
    # D * dt / dx^2 <= 1/2
    # dt <= dx^2 / ( 2*D )
    # dt_max = dx**2 / 2 # since D = 1
    # dt_max = dx**2 / (2*self.Dn_scaled)

    # dx2overtau = dx**2 / self.tau_scaled
    dx2overtau = 10.0
    self.dx2overtau = dx2overtau

    self.logger.info('dx:                           {:> 8.4g}'.format(self.dx))
    self.logger.info('dx2overtau:                   {:> 8.4g}'.format(self.dx2overtau))

    # positions (scaled)
    X = I*dx
    self.X = X

    # Bounary & initial values

    # Potential Dirichlet BC
    self.u0 = 0
    self.u1 = self.delta_u_scaled
    self.logger.info('Left hand side Dirichlet boundary condition:  u0 = {:> 8.4g}'.format(self.u0))
    self.logger.info('Right hand side Dirichlet boundary condition: u1 = {:> 8.4g}'.format(self.u1))

    # internally:
    #   n: dimensionless concentrations
    #   u: dimensionless potential
    #   i: spatial index
    #   j: temporal index
    # initial concentrations equal to bulk concentrations
    ni0 = np.ones(I.shape)*self.c_scaled

    # system matrix of spatial poisson equation
    Au = np.zeros((self.Ni,self.Ni))
    bu = np.zeros(self.Ni)
    Au[0,0]   = 1
    Au[-1,-1] = 1
    bu[0]  = self.u0
    bu[-1] = self.u1

    # get initial potential distribution by solving Poisson equation
    for i in range(1,N):
        Au[i,[i-1,i,i+1]] = [1.0, -2.0, 1.0] # 1D Laplace operator, 2nd order
        bu[i] = ni0[i]*dx**2 # ni0 == Ni => Poisson equation

    ui0 = np.dot( np.linalg.inv(Au), bu) # A u - b = 0 <=> u = A^-1 b

    xi0 = np.concatenate([ui0,ni0])

    xij = self.newton(self.G,xi0.copy())

    self.uij = xij[:self.Ni]
    self.nij = xij[self.Ni:]

    return xij

  # the non-linear system, "controlled volume"
  def G(self, x):
    # global NDi, Ni, u0, u1

    uij1 = x[:self.Ni]
    nij1 = x[self.Ni:]

    Fu = ( np.roll(uij1, -1) - 2*uij1 + np.roll(uij1, 1) ) - nij1*self.dx**2

    Fn =    + B(np.roll(uij1, 1) - uij1)  * np.roll(nij1, 1) \
            - B(uij1 - np.roll(uij1, 1))  * nij1 \
            + B(np.roll(uij1, -1) - uij1) * np.roll(nij1, -1) \
            - B(uij1 - np.roll(uij1, -1)) * nij1 \
            - nij1 * self.dx2overtau

    # Dirichlet BC:
    Fu[0]  = uij1[0] - self.u0
    Fu[-1] = uij1[-1] - self.u1

    self.logger.debug('Dirichlet BC  Fu[0] = uij[0]  - u0 = {:> 8.4g}'.format(Fu[0]))
    self.logger.debug('Dirichlet BC Fu[-1] = uij[-1] - u1 = {:> 8.4g}'.format(Fu[-1]))

    # implement flux BC (Neumann BC) here
    #
    # right-hand side first derivative of second order error
    # df0dx = 1 / (2*dx) * (-3 f0 + 4 f1 - f2 ) + O(dx^2) = 0
    # f0 = (4 f1 - f2) / 3
    # right-hand side first derivative of second order error
    # dfndx = 1 / (2*dx) * (+3 fn - 4 fn-1 + fn-2 ) + O(dx^2) = 0
    #
    # Fn[0]  = nij1[0] - self.NDi[0]
    Fn[0] = -3.0*nij1[0] + 4.0*nij1[1] - nij1[2]
    Fn[int(self.Ni/2)] = nij1[int(self.Ni/2)] - self.c_scaled # BC
    Fn[-1] = 3.0*nij1[-1] - 4.0*nij1[-2] + nij1[-3]

    self.logger.debug('Neumann BC Fn[0]  = -3*nij[0]  + 4*nij[1]  - nij[2]  = {:> 8.4g}'.format(Fn[0]))
    # self.logger.debug('Neumann BC Fn[-1] = -3*nij[-1] + 4*nij[-2] - nij[-3] = {:> 8.4g}'.format(Fn[-1]))
    self.logger.debug('Dirichlet BC Fn[-1] = nij[-1] - n1 = {:> 8.4g}'.format(Fn[-1]))
    # self.logger.debug('Dirichlet BC Fn[-1] = -3*nij[-1] + 4*nij[-2] - nij[-3] = {:> 8.4g}'.format(Fn[-1]))

    return np.concatenate([Fu,Fn])

  def I(self): # ionic strength
    return 0.5*np.sum( np.square(self.z) * self.c )

  def lambda_D(self):
    return np.sqrt(
      self.relative_permittivity*self.vacuum_permittivity*self.R*self.T/(
        2.0*self.F**2*self.I() ) )

  # default 0.1 mM NaCl aqueous solution
  def __init__(self,
    c = 0.1,
    z = 1,
    L = 100e-9, # 100 nm
    T = 298.15,
    delta_u = 1, # potential difference [V]
    relative_permittivity = 79,
    vacuum_permittivity   = sc.epsilon_0,
    R = sc.value('molar gas constant'),
    F = sc.value('Faraday constant') ):

    self.logger = logging.getLogger(__name__)


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

    self.logger.info('bulk concentration c:         {:> 8.4g}'.format(self.c))
    self.logger.info('charge number z:              {:> 8.4g}'.format(self.z))
    self.logger.info('temperature T:                {:> 8.4g}'.format(self.T))
    self.logger.info('domain size L:                {:> 8.4g}'.format(self.L))
    self.logger.info('potential difference delta_u: {:> 8.4g}'.format(self.delta_u))
    self.logger.info('relative permittivity eps_R:  {:> 8.4g}'.format(self.relative_permittivity))
    self.logger.info('vacuum permittivity eps_0:    {:> 8.4g}'.format(self.vacuum_permittivity))
    self.logger.info('universal gas constant R:     {:> 8.4g}'.format(self.R))
    self.logger.info('Faraday constant F:           {:> 8.4g}'.format(self.F))
    self.logger.info('f = F / (RT)                  {:> 8.4g}'.format(self.f))

    # scaled units for dimensionless formulation

    # length unit chosen as Debye length lambda
    self.l_unit = self.lambda_D()

    # concentration unit is ionic strength
    self.c_unit =  self.I()

    # no time unit for now, only steady state
    # self.t_unit = self.l_unit**2 / self.Dn # fixes Dn_scaled = 1

    # u = psi * q / kB T
    # u_unit = kB * T / q
    self.u_unit = self.R * self.T / self.F # thermal voltage

    self.logger.info('spatial unit [l]:             {:> 8.4g}'.format(self.l_unit))
    self.logger.info('concentration unit [c]:       {:> 8.4g}'.format(self.c_unit))
    self.logger.info('potential unit [u]:           {:> 8.4g}'.format(self.u_unit))

    # domain
    self.L_scaled = self.L / self.l_unit

    # bulk conectrations
    self.c_scaled = self.c / self.c_unit

    # potential difference
    self.delta_u_scaled = self.delta_u / self.u_unit

    # relaxation time
    # self.tau_scaled   = self.tau / self.t_unit

    # should be 1
    # Dn_scaled    = Dn * t_unit / l_unit**2
    self.logger.info('reduced domain size L*:       {:> 8.4g}'.format(self.L_scaled))
    self.logger.info('reduced concentation c*:      {:> 8.4g}'.format(self.c_scaled))
    self.logger.info('reduced potential delta_u*:   {:> 8.4g}'.format(self.delta_u_scaled))
