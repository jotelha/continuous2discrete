""" Calculate ionic densities consistent with the Poisson-Bolzmann equation.
Copyright 2019 Simulation Lab
University of Freiburg
Author: Lukas Elflein <elfleinl@cs.uni-freiburg.de>
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
import decimal

np.random.seed(74)

def debye(rho_bulk, charge, permittivity=79, temperature=298.15):
	"""Calculate the Debye length.
	The Dybe length indicates at which distance a charge will be screened off.

	Arguments:
	rho_bulk: dictionary of the bulk number densites for each ionic species [1/m^3]
	charge: dictionary of the charge of each ionic species [1]
	permittivity: capacitance of the ionic solution [1]
	temperature: Temperature of the solution [K]

	Returns:
	float: the Debye length [m], should be around 10^-19

	Example: the Debye length of 10^-4 M salt water is 30.4 nm.
	>>> density = sc.Avogadro * 1000 * 10**-4
	>>> rho = {'Na': density, 'Cl': density}
	>>> charge = {'Na': 1, 'Cl': -1}
	>>> deb = debye(rho_bulk=rho, charge=charge) * 10**9
	>>> deb - 30.4 < 0.5
	True
	"""
	# The numerator of the expression in the square root
	# e_r * e_0 * k_B * T
	numerator = permittivity * sc.epsilon_0 * sc.Boltzmann * temperature

	# The divisor of the expression in the square root
	# \sum_i rho_i e^2 z_i^2
	divisor = 0
	for key in rho_bulk.keys():
		divisor += rho_bulk[key] * sc.elementary_charge ** 2 * charge[key] ** 2

	# The complete square root
	return np.sqrt(numerator / divisor)


def gamma(surface_potential, temperature):
	"""Calculate term from Gouy-Chapmann theory.

	Arguments:
	surface_potential: Electrostatic potential at the metal/solution boundary in Volts, e.g. 0.05 [V]
	temperature: Temperature of the solution in Kelvin, e.g. 300 [K]

	Returns:
	float
	"""
	product = sc.elementary_charge * surface_potential / (4 * sc.Stefan_Boltzmann * temperature)
	return np.tanh(product)

def potential(location, rho_bulk, charge, surface_potential, temperature=300, permittivity=80):
	"""The potential near a charged surface in an ionic solution.

	A single value is returned, which specifies the value of the potential at this distance.

	The decimal package is used for increased precision.
	If only normal float precision is used, the potential is a step function.
	Steps in the potential result in unphysical particle concentrations.

	Arguments:
	location: z-distance from the surface [m]
	temperature: Temperature of the soultion [Kelvin]
	gamma: term from Gouy-Chapmann theory
	kappa: the inverse of the debye length
	charge: dictionary of the charge of each ionic species
	permittivity: capacitance of the ionic solution []
	temperature: Temperature of the solution [Kelvin]

	Returns:
	psi: Electrostatic potential [V]
	"""
	# Increase the precision of the calculation to 30 digits to ensure a smooth potential
	decimal.getcontext().prec = 30

	# Calculate the term in front of the log, containing a bunch of constants
	prefactor = decimal.Decimal(2 * sc.Stefan_Boltzmann * temperature / sc.elementary_charge)

	# For the later calculations we need the debye length
	debye_value =  debye(rho_bulk=rho_bulk, charge=charge, permittivity=permittivity, temperature=temperature)
	kappa = 1/debye_value

	# We also need to evaluate the gamma function
	gamma_value =  decimal.Decimal(gamma(surface_potential=surface_potential, temperature=temperature))

	# The e^{-kz} term
	exponential = decimal.Decimal(np.exp(-kappa * location))

	# The fraction inside the log
	numerator = decimal.Decimal(1) + gamma_value * exponential
	divisor =   decimal.Decimal(1) - gamma_value * exponential

	# This is the complete term for the potential
	psi = prefactor * (numerator / divisor).ln()

	# Convert to float again for better handling in plotting etc.
	psi = float(psi)

	return psi


def charge_density(location, rho_bulk, charge, surface_potential, temperature=300, permittivity=80, species='Na'):
	"""The density of ions near a charged surface.

	Calculates the number density of ions of a species, at a distance of "location"
	away from a metal/solution interface. The metal is charged, and thus the density
	is not homogeneous but higher (lower) than bulk concentration for opposite (equally)
	charged ions, converging to the bulk concentration at high distances.

	Arguments:
	location: z-distance from the surface [m]
	rho_bulk: dictionary of ionic concentrations far from the surface [m^-3]
	charge: dictionary of the charge of each ionic species [e]
	surface_potential: eletrostatic potential at the metal/liquid interface [V]
	temperature: Temperature of the solution [K]
	permittivity: relative permittivity of the ionic solution, 80 for water [1]
	species: the atom name the density shoul be calculated for.
		 Must be contained in `rho_bulk` and `charge`.

	Returns:
	rho: number density of ions of given species [1/m^3]
	"""

	# Evaluate the potential at the current location
	potential_value = potential(location, rho_bulk, charge, surface_potential, temperature, permittivity)

	# The density is an exponential function of the potential
	rho = np.exp(-1 * charge[species] * sc.elementary_charge * potential_value / (sc.Boltzmann * temperature))

	# The density is scaled relative to the bulk concentration
	rho *= rho_bulk[species]

	return rho


def test():
	"""Run docstring unittests"""
	import doctest
	doctest.testmod()


def main():
	"""Do stuff."""

	print('Done.')


if __name__ == '__main__':
	# Run doctests
	test()
	# Execute everything else
	main()
