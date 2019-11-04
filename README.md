# Introduction
Samples discrete coordinate sets from arbitrary continuous distributions

## Background
In order to investigate the electrochemical double layer at interfaces, this tool samples discrete coordinate sets from classical continuum solutions to Poisson-Nernst-Planck systems. This reduces to the Poisson-Boltzmann distribution as an analyitc solution for the special case of a binary electrolyte half space. Coordinate sets are stored as .xyz or LAMMPS data files.

![pic](https://i.ibb.co/Yh8DxVM/showpicture.png)

### Content
* `showcase_poisson_boltzmann.ipynb`: A working example
* `continuous2discrete`:
  * `continuous2discrete.py`: Sampling and plotting
  * `poisson_boltzmann_distribution.py`: generate potential and densities

### Usage
When installed with pip, i.e.

    pip install -e .

from within this directory, `pnp` (Poisson-Nernst-Planck) and `c2d` (continuous2discrete) 
offer simple command line interfaces to solve arbitrary (1D) Poisson-Nernst-Planck systems 
and to sample discrete coordinate sets from continuous distributions. Type `pnp --help` and
`c2d --help` for usage information.

A simple sample usage to generate a discrete coordinate set from
the continuous solution of Poisson-Nernst-Planck equations for
0.1 mM NaCl aqueous solution across a 0.05 V potential drop would look like this

    pnp -c 0.1 0.1 -u 0.05 -l 1.0e-7 -bc cell --verbose NaCl.txt
    c2d --verbose NaCl.txt NaCl.lammps

for PNP solution in plain text file and according coordinate samples LAMMPS
data file, or like this

    pnp -c 0.1 0.1 -u 0.05 -l 1.0e-7 -bc cell --verbose NaCl.npz
    c2d --verbose NaCl.npz NaCl.xyz

for PNP solution in binary numpy .npz file and coordinate samples in generic
xyz file, or as a pipeline

    pnp -c 0.1 0.1 -u 0.05 -l 1.0e-7 -bc cell --verbose | c2d --verbose > NaCl.xyz

for text and xyz format streams.

For more sophisticated usage examples have a look at the notebooks,
 `showcase_poisson_bolzman_distribution.ipynb` is a good candidate.

If you can't get the notebooks to run of the box, set up an environment
```bash
mkdir env
virtualenv env
source env/bin/activate
ipython kernel install --user --name=new_kernel
jupyter notebook showcase_poisson_bolzman_distribution.ipynb
```
and then choose `new_kernel` in the top right dropdown menu as a kernel.
