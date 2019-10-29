""" Calculate ionic concentrations consistent with general
Poisson-Nernst-Planck (PNP) equations.

Copyright 2019 IMTEK Simulation
University of Freiburg

Authors:
  Johannes Hoermann <johannes.hoermann@imtek-uni-freiburg.de>
"""
import logging
import itertools as it
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Druecke nicht-linearen Teil der Transportgleichung (genauer, des Flusses) ueber
# Bernoulli-Funktionen
#
# $$ B(x) = \frac{x}{\exp(x)-1} $$
#
# aus. Damit wir in der Naehe von 0 nicht "in die Bredouille geraten", verwenden
# wir hier lieber die Taylorentwicklung. In der Literatur (Selbherr, S. Analysis
# and Simulation of Semiconductor Devices, Spriger 1984) wird eine noch
# aufwendigere stueckweise Definition empfohlen, allerdings werden wir im
# Folgenden sehen, dass unser Ansatz fuer dieses stationaere Problem genuegt.
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
  # properties "offer" the solution in physical units:
  @property
  def grid(self):
    return self.X*self.l_unit

  @property
  def potential(self):
    return self.uij*self.u_unit

  @property
  def concentration(self):
    return self.nij*self.c_unit

  def newton(self,f,xij):
    """Newton solver expects system f and initial value xij"""
    self.xij = []

    self.logger.debug('Newton solver, grid points N = {:d}'.format(self.N))
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

    dxij = np.zeros(self.N)
    while delta_rel > self.e and i < self.maxit:
        self.logger.debug('*** Newton solver iteration {:d} ***'.format(i))

        # avoid cluttering log
        self.logger.disabled = True
        J = jacobian(f, xij)
        self.logger.disabled = False

        rank = np.linalg.matrix_rank(J)
        self.logger.debug('    Jacobian ({}) rank {:d}'.format(J.shape, rank))

        if rank < self.N:
            if self.output:
                self.logger.warn("Singular jacobian of rank"
                      + "{:d} < {:d} at step {:d}".format(
                      rank, self.N, i ))
            break

        F = f(xij)
        invJ = np.linalg.inv(J)

        dxij = np.dot( invJ, F )

        delta = np.linalg.norm(dxij)
        delta_rel = delta / np.linalg.norm(xij)

        xij -= dxij
        self.xij.append(xij)

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
            duij = dxij[:self.Ni]
            dnij = dxij[self.Ni:(self.M+1)*self.Ni].reshape(self.M,self.Ni)

            uij = xij[:self.Ni]
            nij = xij[self.Ni:(self.M+1)*self.Ni].reshape(self.M,self.Ni)

            ax1.plot(self.X, uij, '-', label='u step {:02d}'.format(i))
            for k in range(self.M):
              ax2.plot(self.X, nij[k,:], '-', label='n step {:02d}, species {:02d}'.format(i,k))
            ax3.plot(self.X, duij, '-', label='du step {:02d}'.format(i))
            for k in range(self.M):
              ax4.plot(self.X, dnij[k,:], '-', label='dn step {:02d}, species {:02d}'.format(i,k))

            ax1.legend(loc='best')
            ax2.legend(loc='best')
            ax3.legend(loc='best')
            ax4.legend(loc='best')

        i += 1

    if i == self.maxit and self.output:
        self.logger.warn("Maximum number of iterations reached")

    if self.output:
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

    # convergenceStepAbsolute[:i], convergenceStepRelative[:i], convergenceResidualAbsolute[:i]
    return xij

  def init(self):
    """Sets up discretization scheme and initial value"""
    # indices
    self.Ni = self.N+1
    I = np.arange(self.Ni)

    self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
      'discretization segments N', self.N, lwidth=self.label_width))
    self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
      'grid points N', self.Ni, lwidth=self.label_width))

    # discretization
    self.dx      = self.L_scaled / self.N # spatial step
    # maximum time step allowed
    # (irrelevant for our steady state case)
    # D * dt / dx^2 <= 1/2
    # dt <= dx^2 / ( 2*D )
    # dt_max = dx**2 / 2 # since D = 1
    # dt_max = dx**2 / (2*self.Dn_scaled)

    # dx2overtau = dx**2 / self.tau_scaled
    self.dx2overtau = 10.0

    self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
      'dx', self.dx, lwidth=self.label_width))
    self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
      'dx2overtau', self.dx2overtau, lwidth=self.label_width))

    # positions (scaled)
    self.X = I*self.dx

    # Bounary & initial values

    # internally:
    #   n: dimensionless concentrations
    #   u: dimensionless potential
    #   i: spatial index
    #   j: temporal index
    #   k: species index
    # initial concentrations equal to bulk concentrations

    # Kronecker product, M rows (ion species), Ni cols (grid points),
    self.ni0 = np.kron( self.c_scaled, np.ones((self.Ni,1)) ).T
    #self.ni0 = np.kron( self.c_scaled, np.zeros((self.Ni,1)) ).T
    self.zi0 = np.kron( self.z, np.ones((self.Ni,1)) ).T # does not change

    self.initial_values()

  def initial_values(self):
    """
    Solves decoupled linear system to get inital potential distribution.

    Returns:
    np.ndarray((Ni)): initial potential distribution ui0 from Poisson's equation
                      based on initial concentration distributions ni0 and
                      potential Dirichlet boubdary conditions
    """

    zini0 = self.zi0*self.ni0 # z*n
    #rhoi0 = np.zeros( self.Ni ) # initial charge distribution (dimensionless)
    #for k in range(0,self.M):
    #  rhoi0 += zini0[(k*self.Ni):((k+1)*self.Ni)]
    # shape: ion species (rows), grid points (cols), sum over ion species (along rows)
    rhoi0 = zini0.sum(axis=0)

    # system matrix of spatial poisson equation
    Au = np.zeros((self.Ni,self.Ni))
    bu = np.zeros(self.Ni)
    Au[0,0]   = 1
    Au[-1,-1] = 1
    for i in range(1,self.N):
        Au[i,[i-1,i,i+1]] = [1.0, -2.0, 1.0] # 1D Laplace operator, 2nd order

    bu = rhoi0*self.dx**2 # => Poisson equation
    bu[0]  = self.u0
    bu[-1] = self.u1

    # get initial potential distribution by solving Poisson equation
    self.ui0 = np.dot( np.linalg.inv(Au), bu) # A u - b = 0 <=> u = A^-1 b

    return self.ui0

  # evokes Newton solver
  def solve(self):
    """Evokes newton solver

    Returns
    uij:  np.ndarray((Ni), potential at Ni grid points
    nij:  np.ndarray((M,Nij)), concentrations of M species at Ni grid points
    lamj: np.ndarray((L)), value of L Lagrange multipliers
    """

    if len(self.g) > 0:
      self.xi0 = np.concatenate([self.ui0, self.ni0.flatten(), np.zeros(len(self.g))])
    else:
      self.xi0 = np.concatenate([self.ui0, self.ni0.flatten()])

    self.xij1 = self.newton(self.G,self.xi0.copy())

    # store results:
    self.uij  = self.xij1[:self.Ni] # potential
    self.nij  = self.xij1[self.Ni:(self.M+1)*self.Ni].reshape(self.M, self.Ni) # concentrations
    self.lamj = self.xij1[(self.M+1)*self.Ni:] # Lagrange multipliers

    return self.uij, self.nij, self.lamj

  # standard sets of boundary conditions:
  def useStandardInterfaceBC(self):
    """Interface at left hand side and open bulk at right hand side"""
    self.boundary_conditions = []

    # Potential Dirichlet BC
    self.u0 = self.delta_u_scaled
    self.u1 = 0
    self.logger.info('Left hand side Dirichlet boundary condition:                               u0 = {:> 8.4g}'.format(self.u0))
    self.logger.info('Right hand side Dirichlet boundary condition:                              u1 = {:> 8.4g}'.format(self.u1))

    self.boundary_conditions.extend([
      lambda x: self.leftPotentialDirichletBC(x,self.u0),
      lambda x: self.rightPotentialDirichletBC(x,self.u1) ])
    # self.rightPotentialBC = lambda x: self.rightPotentialDirichletBC(x,self.u1)

    #self.rightConcentrationBC = []
    for k in range(self.M):
      self.logger.info('Ion species {:02d} left hand side concentration Flux boundary condition:       j0 = {:> 8.4g}'.format(k,0))
      self.logger.info('Ion species {:02d} right hand side concentration Dirichlet boundary condition: c1 = {:> 8.4g}'.format(k,self.c_scaled[k]))
      self.boundary_conditions.extend( [
        lambda x, k=k: self.leftFluxBC(x,k),
        lambda x, k=k: self.rightDirichletBC(x,k,self.c_scaled[k]) ] )
      #self.rightConcentrationBC.append(
      #  lambda x, k=k: self.rightDirichletBC(x,k,self.c_scaled[k]) )
    # counter-intuitive behavior of lambda in loop:
    # https://stackoverflow.com/questions/2295290/what-do-lambda-function-closures-capture
    # workaround: default parameter k=k

  def useStandardCellBC(self):
    """Interfaces at left hand side and right hand side"""
    self.boundary_conditions = []

    # Potential Dirichlet BC
    self.u0 = self.delta_u_scaled / 2.0
    self.u1 = - self.delta_u_scaled / 2.0
    self.logger.info('Left hand side Dirichlet boundary condition:                               u0 = {:> 8.4g}'.format(self.u0))
    self.logger.info('Right hand side Dirichlet boundary condition:                              u1 = {:> 8.4g}'.format(self.u1))
    self.boundary_conditions.extend([
      lambda x: self.leftPotentialDirichletBC(x,self.u0),
      lambda x: self.rightPotentialDirichletBC(x,self.u1) ])
    #self.rightPotentialBC = lambda x: self.rightPotentialDirichletBC(x,self.u1)

    #self.leftConcentrationBC = []
    #self.rightConcentrationBC = []

    N0 = self.L_scaled*self.c_scaled # total amount of species in cell
    for k in range(self.M):
      self.logger.info('Ion species {:02d} left hand side concentration Flux boundary condition:       j0 = {:> 8.4g}'.format(k,0))
      #self.logger.info('Ion species {:02d} right hand side concentration Flux boundary condition:      c1 = {:> 8.4g}'.format(k,0))
      self.logger.info('Ion species {:02d} number conservation constraint:                             N0 = {:> 8.4g}'.format(k,N0[k]))
      self.boundary_conditions.extend(  [
        lambda x, k=k: self.leftFluxBC(x,k),
        lambda x, k=k, N0=N0[k]: self.numberConservationConstraint(x,k,N0) ] )
      #self.rightConcentrationBC.append( lambda x, k=k: self.rightFluxBC(x,k) )


    # constraints
    # N0 = self.L_scaled*self.c_scaled # total amount of species in cell
    # self.g = []
    # for k in range(self.M):
    #  self.logger.info('Ion species {:02d} number conservation constraint:                             N0 = {:> 8.4g}'.format(k,N0[k]))
    #  self.g.append( lambda x, k=k, N0=N0[k]: self.numberConservationConstraint(x,k,N0) )

  # TODO: meaningful test for Dirichlet BC
  def useStandardDirichletBC(self):
    self.boundary_conditions = []

    self.u0 = self.delta_u_scaled
    self.u1 = 0

    self.logger.info('Left hand side potential Dirichlet boundary condition:                     u0 = {:> 8.4g}'.format(self.u0))
    self.logger.info('Right hand side potential Dirichlet boundary condition:                    u1 = {:> 8.4g}'.format(self.u1))

    # set up boundary conditions
    self.boundary_conditions.extend( [
      lambda x: self.leftPotentialDirichletBC(x,self.u0),
      lambda x: self.rightPotentialDirichletBC(x,self.u1) ] )

    for k in range(self.M):
      self.logger.info('Ion species {:02d} left hand side concentration Dirichlet boundary condition:  c0 = {:> 8.4g}'.format(k,self.c_scaled[k]))
      self.logger.info('Ion species {:02d} right hand side concentration Dirichlet boundary condition: c1 = {:> 8.4g}'.format(k,self.c_scaled[k]))
      self.boundary_conditions.extend( [
        lambda x, k=k: self.leftDirichletBC(x,k,self.c_scaled[k]),
        lambda x, k=k: self.rightDirichletBC(x,k,self.c_scaled[k]) ] )

  # boundary conditions and constraints building blocks:
  def leftFluxBC(self,x,k,j0=0):
      """j0: flux, k: ion species"""
      uij = x[:self.Ni]
      nijk = x[(k+1)*self.Ni:(k+2)*self.Ni]
      # df0dx = 1 / (2*dx) * (-3 f0 + 4 f1 - f2 ) + O(dx^2)
      # - dndx - z n dudx = j0
      dndx = -3.0*nijk[0] + 4.0*nijk[1] - nijk[2]
      dudx = -3.0*uij[0]  + 4.0*uij[1]  - uij[2]
      bcval = - dndx - self.zi0[k,0]*nijk[0]*dudx - 2.0*self.dx*j0

      self.logger.debug(
        'Flux BC F[0]  = - dndx - z n dudx - 2*dx*j0 = {:> 8.4g}'.format(bcval))
      self.logger.debug(
        '   = - ({:.2f}) - ({:.0f})*{:.2f}*({:.2f}) - 2*{:.2f}*({:.2f})'.format(
            dndx, self.zi0[k,0], nijk[0], dudx, self.dx, j0))
      return bcval

  def rightFluxBC(self,x,k,j0=0):
      """j0: flux, k: ion species"""
      uij = x[:self.Ni]
      nijk = x[(k+1)*self.Ni:(k+2)*self.Ni]
      # df0dx = 1 / (2*dx) * (-3 f0 + 4 f1 - f2 ) + O(dx^2)
      # - dndx - z n dudx = j0
      dndx = 3.0*nijk[-1] - 4.0*nijk[-2] + nijk[-3]
      dudx = 3.0*uij[-1]  - 4.0*uij[-2]  + uij[-3]
      bcval = - dndx - self.zi0[k,-1]*nijk[-1]*dudx - 2.0*self.dx*j0

      self.logger.debug(
        'Flux BC F[-1]  = - dndx - z n dudx - 2*dx*j0 = {:> 8.4g}'.format(bcval))
      self.logger.debug(
        '  = - {:.2f} - {:.0f}*{:.2f}*{:.2f} - 2*{:.2f}*{:.2f}'.format(
            dndx, self.zi0[k,-1], nijk[-1], dudx, self.dx, j0))
      return bcval

  def leftPotentialDirichletBC(self,x,u0=0):
    return self.leftDirichletBC(x,-1,u0)

  def leftDirichletBC(self,x,k,x0=0):
    """Construct Dirichlet BC at left boundary"""
    nijk = x[(k+1)*self.Ni:(k+2)*self.Ni]
    return nijk[0] - x0

  def rightPotentialDirichletBC(self,x,x0=0):
    return self.rightDirichletBC(x,-1,x0)

  def rightDirichletBC(self,x,k,x0=0):
    nijk = x[(k+1)*self.Ni:(k+2)*self.Ni]
    return nijk[-1] - x0

  def numberConservationConstraint(self,x,k,N0):
      """N0: total amount of species, k: ion species"""
      nijk = x[(k+1)*self.Ni:(k+2)*self.Ni]

      # rescale to fit interval
      N = np.sum(nijk*self.dx) * self.N / self.Ni
      constraint_val = N - N0

      self.logger.debug(
        'Number conservation constraint F(x)  = N - N0 = {:.4g} - {:.4g} = {:.4g}'.format(
          N, N0, constraint_val ) )
      return constraint_val

  # TODO: remove or standardize
  # def leftNeumannBC(self,x,j0):
  #   """Construct finite difference Neumann BC (flux BC) at left boundary"""
  #   # right hand side first derivative of second order error
  #   # df0dx = 1 / (2*dx) * (-3 f0 + 4 f1 - f2 ) + O(dx^2) = j0
  #   bcval = -3.0*x[0] + 4.0*x[1] - x[2] - 2.0*self.dx*j0
  #   self.logger.debug(
  #     'Neumann BC F[0]  = -3*x[0]  + 4*x[1]  - x[2]  = {:> 8.4g}'.format(bcval))
  #   return bcval
  #
  # def rightNeumannBC(self,x,j0):
  #   """Construct finite difference Neumann BC (flux BC) at right boundray"""
  #   # left hand side first derivative of second order error
  #   # dfndx = 1 / (2*dx) * (+3 fn - 4 fn-1 + fn-2 ) + O(dx^2) = 0
  #   bcval = 3.0*x[-1] - 4.0*x[-2] + x[-3] - 2.0*self.dx*j0
  #   self.logger.debug(
  #     'Neumann BC F[-1] = -3*x[-1] + 4*x[-2] - nijk[-3] = {:> 8.4g}'.format(bcval))
  #   return bcval

  # non-linear system, "controlled volume" method
  # Selbherr, S. Analysis and Simulation of Semiconductor Devices, Spriger 1984
  def G(self, x):
    """Non-linear system

    Discretization of Poisson-Nernst-Planck system with M ion species.
    Implements "controlled volume" method as found in

      Selbherr, Analysis and Simulation of Semiconductor Devices, Spriger 1984

    Arguments
    x:  np.ndarray, system variables. 1D array of (M+1)*Ni values, wher M is number of ion
        sepcies, Ni number of spatial discretization points. First Ni entries
        are expected to contain potential, following M*Ni points contain
        ion concentrations.
    Returns
    np.ndarray: residual
    """

    uij1 = x[:self.Ni]
    self.logger.debug(
      'potential range [u_min, u_max] = [ {:>.4g}, {:>.4g} ]'.format(
        np.min(uij1),np.max(uij1)))

    nij1 = x[self.Ni:(self.M+1)*self.Ni]

    nijk1 = nij1.reshape( self.M, self.Ni )
    for k in range(self.M):
      self.logger.debug(
        'ion species {:02d} concentration range [c_min, c_max] = [ {:>.4g}, {:>.4g} ]'.format(
          k,np.min(nijk1[k,:]),np.max(nijk1[k,:]) ) )

    # M rows (ion species), N_i cols (grid points)
    zi0nijk1 = self.zi0*nijk1 # z_ik*n_ijk
    for k in range(self.M):
      self.logger.debug(
        'ion species {:02d} charge range [z*c_min, z*c_max] = [ {:>.4g}, {:>.4g} ]'.format(
          k,np.min(zi0nijk1[k,:]),np.max(zi0nijk1[k,:]) ) )

    # charge density sum_k=1^M (z_ik*n_ijk)
    rhoij1 = zi0nijk1.sum(axis=0)
    self.logger.debug(
      'charge density range [rho_min, rho_max] = [ {:>.4g}, {:>.4g} ]'.format(
        np.min(rhoij1),np.max(rhoij1) ) )

    # reduced Poisson equation: d2udx2 = rho
    Fu = - ( np.roll(uij1, -1) - 2*uij1 + np.roll(uij1, 1) ) - 0.5 * rhoij1*self.dx**2
    # TODO: double-check factor 0.5 here

    # potential boundary conditions:
    # Fu[0]  = uij1[0] - self.u0
    # Fu[-1] = uij1[-1] - self.u1
    #Fu[0] =  self.leftPotentialBC(x)
    #Fu[-1] = self.rightPotentialBC(x)
    Fu[0] = self.boundary_conditions[0](x)
    Fu[-1] = self.boundary_conditions[1](x)

    self.logger.debug('Potential BC residual Fu[0]  = {:> 8.4g}'.format(Fu[0]))
    self.logger.debug('Potential BC residual Fu[-1] = {:> 8.4g}'.format(Fu[-1]))

    Fn = np.zeros([self.M, self.Ni])
    # loop over k = 1..M reduced Nernst-Planck equations:
    # - d2nkdx2 - ddx (zk nk dudx ) = 0
    for k in range(self.M):
      Fn[k,:] = + B(np.roll(uij1, 1) - uij1)  * np.roll(zi0nijk1[k,:], 1) \
                - B(uij1 - np.roll(uij1, 1))  * (zi0nijk1[k,:]) \
                + B(np.roll(uij1, -1) - uij1) * np.roll(zi0nijk1[k,:], -1) \
                - B(uij1 - np.roll(uij1, -1)) * (zi0nijk1[k,:])

      Fn[k,0]  = self.boundary_conditions[2*k+2](x)
      Fn[k,-1] = self.boundary_conditions[2*k+3](x)

      self.logger.debug(
        'ion species {k:02d} BC residual Fn[{k:d},0]  = {:> 8.4g}'.format(
          Fn[k,0],k=k))
      self.logger.debug(
        'ion species {k:02d} BC residual Fn[{k:d},-1]  = {:> 8.4g}'.format(
          Fn[k,-1],k=k))
      # implement flux BC (Neumann BC) here
      #
      # right-hand side first derivative of second order error
      # df0dx = 1 / (2*dx) * (-3 f0 + 4 f1 - f2 ) + O(dx^2) = 0
      # f0 = (4 f1 - f2) / 3
      # right-hand side first derivative of second order error
      # dfndx = 1 / (2*dx) * (+3 fn - 4 fn-1 + fn-2 ) + O(dx^2) = 0
      #
      # Fn[k,0] = -3.0*nijk1[k,0] + 4.0*nijk1[k,1] - nijk1[k,2]
      # Fn[k,-1] = 3.0*nijk1[k,-1] - 4.0*nijk1[k,-2] + nijk1[k,-3]
      #Fn[int(self.Ni/2)] = nij1[int(self.Ni/2)] - self.c_scaled # BC
      # Fn[k,-1] = nij1[-1] - self.c_scaled # BC

      #self.logger.debug(
      #  'Neumann BC Fn[{k:d},0]  = -3*nijk[{k:d},0]  + 4*nijk[{k:d},1]  - nijk[{k:d},2]  = {:> 8.4g}'.format(Fn[k,0],k=k))
      #self.logger.debug(
      #  'Neumann BC Fn[{k:d},-1] = -3*nijk[{k:d},-1] + 4*nijk[{k:d},-2] - nijk[{k:d},-3] = {:> 8.4g}'.format(Fn[k,-1],k=k))
      # self.logger.debug('Dirichlet BC Fn[-1] = nij[-1] - n1 = {:> 8.4g}'.format(Fn[-1]))
      # self.logger.debug('Dirichlet BC Fn[-1] = -3*nij[-1] + 4*nij[-2] - nij[-3] = {:> 8.4g}'.format(Fn[-1]))

    # Apply constraints if set:
    if len(self.g) > 0:
      Flam = np.array([g(x) for g in self.g])
      F = np.concatenate([Fu,Fn.flatten(),Flam])
    else:
      F = np.concatenate([Fu,Fn.flatten()])

    return F

  def I(self): # ionic strength
    return 0.5*np.sum( np.square(self.z) * self.c )

  def lambda_D(self):
    return np.sqrt(
      self.relative_permittivity*self.vacuum_permittivity*self.R*self.T/(
        2.0*self.F**2*self.I() ) )

  # default 0.1 mM NaCl aqueous solution
  def __init__(self,
    c = np.array([0.1,0.1]),
    z = np.array([1,-1]),
    L = 100e-9, # 100 nm
    T = 298.15,
    delta_u = 0.05, # potential difference [V]
    relative_permittivity = 79,
    vacuum_permittivity   = sc.epsilon_0,
    R = sc.value('molar gas constant'),
    F = sc.value('Faraday constant') ):

    self.logger = logging.getLogger(__name__)

    assert len(c) == len(z), "Provide concentration AND charge for ALL ion species!"

    # default solver settings
    self.N      = 1000  # discretization segments
    self.e      = 1e-10 # Newton solver default tolerance
    self.maxit  = 200   # Newton solver maximum iterations

    # default output settings
    self.output = False   # let Newton solver output convergence plots...
    self.outfreq = 1      # ...at every nth iteration
    self.label_width = 40 # charcater width of quantity labels in log

    # empty BC

    self.boundary_conditions = []
    # empty constraints
    self.g = [] # list of constrain functions

    # system parameters
    self.M = len(c) # number of ion species

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

    for i, (c, z) in enumerate(zip(self.c,self.z)):
      self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
        "ion species {:02d} concentration c".format(i), c, lwidth=self.label_width))
      self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
        "ion species {:02d} number charge z".format(i), z, lwidth=self.label_width))

    self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
      'temperature T', self.T, lwidth=self.label_width))
    self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
      'domain size L', self.L, lwidth=self.label_width))
    self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
      'potential difference delta_u', self.delta_u, lwidth=self.label_width))
    self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
      'relative permittivity eps_R', self.relative_permittivity, lwidth=self.label_width))
    self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
      'vacuum permittivity eps_0', self.vacuum_permittivity, lwidth=self.label_width))
    self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
      'universal gas constant R', self.R, lwidth=self.label_width))
    self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
      'Faraday constant F', self.F, lwidth=self.label_width))
    self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
      'f = F / (RT)', self.f, lwidth=self.label_width))

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

    self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
      'spatial unit [l]', self.l_unit, lwidth=self.label_width))
    self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
      'concentration unit [c]', self.c_unit, lwidth=self.label_width))
    self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
      'potential unit [u]', self.u_unit, lwidth=self.label_width))

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
    self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
      'reduced domain size L*', self.L_scaled, lwidth=self.label_width))
    for i, c_scaled in enumerate(self.c_scaled):
      self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
        "ion species {:02d} reduced concentration c*".format(i),
        c_scaled, lwidth=self.label_width))
    self.logger.info('{:<{lwidth}s} {:> 8.4g}'.format(
      'reduced potential delta_u*', self.delta_u_scaled, lwidth=self.label_width))
