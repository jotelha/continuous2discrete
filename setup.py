#!/usr/bin/env python

from setuptools import setup, find_packages
import os, versioneer, subprocess, re

__author__     = "Lukas Elflein, Johannes Hörmann"
__copyright__  = "Copyright 2019, IMTEK Simulation, University of Freiburg"
__maintainer__ = "Johannes Hörmann"
__email__      = "johannes.hoermann@imtek.uni-freiburg.de"
__date__       = "Oct 25, 2019"

module_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    setup(
        name='continuous2discrete',
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        description='Sample continuous (concentration) distributions and generate xyz file format coordiante sets',
        long_description=open(os.path.join(module_dir, 'README.md')).read(),
        url='https://github.com/lukaselflein/generate_distributed_structure',
        author='Lukas Elflein, Johannes Hörmann',
        author_email='johannes.hoermann@imtek.uni-freiburg.de',
        license='MIT',
        packages=find_packages(),
        package_data={'': ['ChangeLog.md']},
        python_requires='>3.6.3',
        zip_safe=False,
        install_requires=[
            'ase>=3.19.0b1',
            'matplotlib>=3.0.3',
            'numpy >= 1.16.2',
            'pandas>=0.24.2',
            'pytest>=5.2.2',
            'pytest-datadir>=1.3.1',
            'scipy>=1.2.1',
            'six>=1.12.0'],
        entry_points={
            'console_scripts': [
                'c2d = continuous2discrete.continuous2discrete:main',
                'pnp = continuous2discrete.poisson_nernst_planck_distribution:main'
            ],
        }
    )
