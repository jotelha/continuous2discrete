"""
Generates atomic structure following a given distribution.

Copyright 2019 IMTEK Simulation
University of Freiburg

Authors:
  Lukas Elflein <elfleinl@cs.uni-freiburg.de>
  Johannes Hoermann <johannes.hoermann@imtek-uni-freiburg.de>
"""
import logging, os, sys
import os.path
from six.moves import builtins

import numpy as np
import matplotlib.pyplot as plt

import ase, ase.io
import scipy.constants as sc
from scipy import interpolate


logger = logging.getLogger(__name__)

def exponential(x, rate=0.1):
    """Exponential distribution."""
    return rate * np.exp(-1 * rate * x)


def uniform(x, *args, **kwargs):
    """Uniform distribution."""
    return np.ones(x.shape) / 2


def test():
    """Run docstring unittests"""
    import doctest
    doctest.testmod()


def pdf_to_cdf(pdf):
    """Transform partial distribution to cumulative distribution function

    >>> pdf_to_cdf(pdf=np.array([0.5, 0.3, 0.2]))
    array([ 0.5,  0.8,  1. ])
    """
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]

    return cdf


def get_centers(bins):
    """Return the center of the provided bins.

    Example:
    >>> get_centers(bins=np.array([0.0, 1.0, 2.0]))
    array([ 0.5,  1.5])
    """
    bins = bins.astype(float)
    return (bins[:-1] + bins[1:]) / 2


def get_nearest_pos(array, value):
    """Find the value of an array clostest to the second argument.

    Example:
    >>> get_nearest_pos(array=[0, 0.25, 0.5, 0.75, 1.0], value=0.55)
    2
    """
    array = np.asarray(array)
    pos = np.abs(array - value).argmin()
    return pos


def get_histogram(struc, box, n_bins=100):
    """Slice the list of atomic positions, aggregate positions into histogram."""
    # Extract x/y/z positions only
    x, y, z = struc[:, 0], struc[:, 1], struc[:, 2]

    histograms = []
    for dimension in (0, 1, 2):
        bins = np.linspace(0, box[dimension], n_bins)
        hist, bins = np.histogram(struc[:, dimension], bins=bins, density=True)
        # Normalize the histogram for all values to sum to 1
        hist /= sum(hist)

        histograms += [(hist, bins)]
    return histograms


def plot_dist(histogram, name, reference_distribution=None):
    """Plot histogram with an optional reference distribution."""
    hist, bins = histogram
    width = 1 * (bins[1] - bins[0])
    centers = get_centers(bins)

    fi, ax = plt.subplots()
    ax.bar( centers, hist, align='center', width=width, label='Empirical distribution',
            edgecolor="none")

    if reference_distribution is not None:
        ref = reference_distribution(centers)
        ref /= sum(ref)
        ax.plot(centers, ref, color='red', label='Target distribution')

    plt.title(name)
    plt.legend()
    plt.xlabel('Distance ' + name)
    plt.savefig(name + '.png')


def quartile_function(distribution, p, support=None):
    """Inverts a distribution x->p, and returns the x-value belonging to the provided p.

    Assumption: The distribution to be inverted must have a strictly increasing CDF!
    Also see 'https://en.wikipedia.org/wiki/Quantile_function'.

    Parameters
    ----------
    distribution: a function x -> p; x should be approximatable by a compact support
    p: an output of the distribution function, probablities in (0,1) are preferrable
    """
    if support is None:
        # Define the x-values to evaluate the function on
        support = np.arange(0,1,0.01)

    # Calculate the histogram of the distribution
    hist = distribution(support)

    # Sum the distribution to get the cumulatative distribution
    cdf =  pdf_to_cdf(hist)

    # If the p is not in the image of the support, get the nearest hit instead
    nearest_pos = get_nearest_pos(cdf, p)

    # Get the x-value belonging to the probablity value provided in the input
    x = support[nearest_pos]
    return x

def inversion_sampler(distribution, support):
    """Wrapper for quartile_function."""
    # z is distributed according to the given distribution
    # To approximate this, we insert an atom with probablity dis(z) at place z.
    # This we do by inverting the distribution, and sampling uniformely from distri^-1:
    p = np.random.uniform()
    sample = quartile_function(distribution, p, support=support)

    return sample


def rejection_sampler(distribution, support, max_tries=10000):
    """Sample distribution by drawing from support and keeping according to distribution.

    Draw a random sample from our support, and keep it if another random number is
    smaller than our target distribution at the support location.

    Parameters
    ----------
    distribution: The target distribution, as a histogram over the support
    support: locations in space where our distribution is defined
    max_tries: how often the sampler should attempt to draw before giving up.
       If the distribution is very sparse, increase this parameter to still get results.

    Returns
    -------
    sample: a location which is conistent (in expectation) with being drawn from the distribution.
    """

    # rejection sampling (https://en.wikipedia.org/wiki/Rejection_sampling):
    # Generates sampling values from a target distribution X with arbitrary
    # probability density function f(x) by using a proposal distribution Y
    # with probability density g(x).
    # Concept: Generates a sample value from X by instead sampling from Y and
    # accepting this sample with probability f(x) / ( M g(x) ), repeating the
    # draws from Y until a value is accepted. M here is a constant, finite bound
    # on the likelihood ratio f(x)/g(x), satisfying 1 < M < infty over the
    # support of X; in other words, M must satisfy f(x) <= Mg(x) for all values
    # of x. The support of Y must include the support of X.

    # Here, f(x) = distribution(x), g(x) is uniform density on [0,1)
    # X are f-distributed positions from support, Y are uniformly distributed
    # values from [0,1)]
    logger.debug("Rejection sampler on distribution f(x) ({}) with".format(
        distribution))
    logger.debug("discrete support X ({:d} points in interval [{},{}]".format(
        len(support), np.min(support), np.max(support)))
    # uniform probability density g(x) on support is
    g = 1.0 / len(support) # for discrete support
    # maximum probability on distributiom f(x) is
    fmax = np.max(distribution(support))
    # thus M must be at least
    M = fmax / g
    logger.debug("Uniform probability g(x) = {:g} and".format(g))
    logger.debug("maximum probability max(f(x)) = {:g} require".format(fmax))
    logger.debug("M >= g(x)/max(f(x)), i.e. M = {:g}.".format(M))

    for i in range(max_tries):
        # draw a sample from support
        sample = np.random.choice(support)

        # Generate random float in the half-open interval [0.0, 1.0) and .
        # keep sample with probablity of distribution
        if np.random.random() < distribution(sample) / (M*g):
            return sample

    raise RuntimeError('Maximum of attempts max_tries {} exceeded!'.format(max_tries))

def generate_structure(
    distribution=None, box=np.array([50, 50, 100]),
    atom_count=100, n_gridpoints=100,
    distribution_x=uniform, distribution_y=uniform, distribution_z=uniform):
    """Generate an atomic structure.

    Coordinates are distributed according to given distributions.
    To construct the positions, we insert an atom with probality
    distribution(z) at place z. This sampling is done by inverting
    the distribution, and sampling uniformely from distri^-1.

    Per default, X and Y coordinates are drawn uniformely.

    Parameters
    ----------
    distribution_x: func(x), optional
      distribution for sampling in x direction (default: uniform)
    distribution_y: func(x), optional (default: uniform)
    distribution_z: func(x), optional (default: uniform)
    distribution: func(x), optional (default: None)
      If none of the above is explicitly specified, but 'distribution' is, then
      uniform sampling appplies along x and y axes, while applying
      'distribution' along z axis.
    box: np.ndarray(3), optional (default: np.array([50, 50, 100]) )
      dimensions of volume to be filled with samples

    atom_count: int, optional (default: 100)
      number of samples to draw
    n_gridpoints: int or (int,int,int), optional (default: 100)
      samples are not placed arbitrarily, but on a evenly spaced grid of this
      many grid points along each axis. Specify

    Returns
    -------
    np.ndarray((sample_size,3)): sample coordinates
    """
    global logger

    if distribution is not None:
        logger.info("Using 'distribution' {} as 'distribution_z'".format(
            distribution))
        distribution_z = distribution

    assert callable(distribution_x), "distribution_x must be callable"
    assert callable(distribution_y), "distribution_y must be callable"
    assert callable(distribution_z), "distribution_z must be callable"

    assert np.array(box).shape == (3,), "wrong specification of 3d box dimensions"
    if isinstance(n_gridpoints,int):
        n_gridpoints = 3*[n_gridpoints] # to list

    n_gridpoints = np.array(n_gridpoints,dtype=int)

    logger.info("Using {} grid as sampling support.".format(
        n_gridpoints))

    assert n_gridpoints.shape == (3,), "n_gridpoints must be int or list of int"

    # We define which positions in space the atoms can be placed
    support = {}
    # Using the box parameter, we construct a grid inside the box
    # This results in a 100x100x100 grid:
    support_x = np.linspace(0, box[0], n_gridpoints[0])
    support_y = np.linspace(0, box[1], n_gridpoints[1])
    support_z = np.linspace(0, box[2], n_gridpoints[2])

    # Normalize distributions:
    Zx = np.sum(distribution_x(support_x))
    Zy = np.sum(distribution_y(support_y))
    Zz = np.sum(distribution_z(support_z))
    normalized_distribution_x = lambda x: distribution_x(x) / Zx
    normalized_distribution_y = lambda x: distribution_y(x) / Zy
    normalized_distribution_z = lambda x: distribution_z(x) / Zz

    logger.info("Normalizing 'distribution_x' {} by {}.".format(
        distribution_x,Zx))
    logger.info("Normalizing 'distribution_y' {} by {}.".format(
        distribution_y,Zy))
    logger.info("Normalizing 'distribution_z' {} by {}.".format(
        distribution_z,Zz))

    atom_positions = []
    # For every atom, draw random x, y and z coordinates
    for i in range(atom_count):
        x = rejection_sampler(normalized_distribution_x, support_x)
        y = rejection_sampler(normalized_distribution_y, support_y)
        z = rejection_sampler(normalized_distribution_z, support_z)
        atom_positions += [[x, y, z]]

    atom_positions = np.array(atom_positions)
    logger.info("Drew {} samples from distributions.".format(
        atom_positions.shape))
    return atom_positions


def export_xyz( struc, atom_name='Na', box=[50.0,50.0,100.0],
                outfile_name='distributed_atom_structure.xyz'):
    """Export atom structure to .xyz file format."""

    # We have as many atoms as coordinate positions
    n_atoms = struc.shape[0]

    with open(outfile_name, 'w') as outfile:
        outfile.write('{}\n'.format(n_atoms))
        # Add Lattice information
        comment = 'Lattice="'
        R_1 = '{:.1f} 0.0 0.0'.format(box[0])
        R_2 = ' 0.0 {:.1f} 0.0'.format(box[1])
        R_3 = ' 0.0 0.0 {:.1f}'.format(box[2])
        comment += R_1 + R_2 + R_3 + '"'
        comment += '\n'
        outfile.write(comment)

        for line in struc:
            x, y, z = line
            out_line = '{} {} {} {}\n'.format(atom_name, x, y, z)
            outfile.write(out_line)


def export_named_struc(struc):
    """Export atom structure to .xyz file format.

    Parameters
    ----------
    named_struc: list of atom names and positions, with lines like e.g. Na 1.2 5.1 4.2
    """

    # We have as many atoms as coordinate positions
    n_atoms = struc.shape[0]

    with open('distributed_atom_structure.xyz', 'w') as outfile:
        outfile.write('{}\n'.format(n_atoms))
        comment = '\n'
        outfile.write(comment)

        for line in struc:
            atom_name, x, y, z = line
            out_line = '{} {} {} {}\n'.format(atom_name, x, y, z)
            outfile.write(out_line)


def concat_names_structs(struc_list, name_list):
    """Append individual structure objects and name each line."""
    assert len(struc_list) == len(name_list)

    concated_list = []
    for i in range(len(struc_list)):
        atom_name = name_list[i]

    for line in struc_list[i]:
        x, y, z = line
        concated_list += [[atom_name, x, y, z]]

    return np.array(concated_list)

def main():
    """Generate and export an atomic structure, and plot its distribution"""
    import argparse

    # in order to have both:
    # * preformatted help text and ...
    # * automatic display of defaults
    class ArgumentDefaultsAndRawDescriptionHelpFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class = ArgumentDefaultsAndRawDescriptionHelpFormatter)

    parser.add_argument('infile', metavar='IN', nargs='?',
                        help='binary numpy .npz or plain text .txt input file')
    parser.add_argument('outfile', metavar='OUT', nargs='?',
                        help='.xyz format output file')

    parser.add_argument('--box','-b', default=[50.0e-9,50.0e-9,100.0e-9], nargs=3,
                        metavar=('X','Y','Z'), required=False, type=float,
                        dest="box", help='Box dimensions')

    #parser.add_argument('--distribution','-d',
    #                    default='continuous2discrete.exponential', type=str,
    #                    metavar='FUNC', required=False, dest="distribution",
    #                    help='Fully qualified distribution function name')

    parser.add_argument('--names', default=['Na','Cl'], type=str, nargs='+',
                        metavar=('NAME'), required=False, dest="names",
                        help='Atom names')

    parser.add_argument('--charges', default=[1,-1], type=float, nargs='+',
                        metavar=('NAME'), required=False, dest="charges",
                        help='Atom charges')

    # sampling
    parser.add_argument('--ngridpoints', default=1000, type=int, nargs='+',
                        metavar=('N'), required=False, dest="ngridpoints",
                        help='Number of grid points for discrete support')
    parser.add_argument('--sample-size', default=1000, type=int, nargs='+',
                        metavar=('N'), required=False, dest="sample_size",
                        help='Sample size')

    # output
    parser.add_argument('--nbins', default=100, type=int,
                        metavar=('N'), required=False, dest="nbins",
                        help='Number of bins for histogram plots')
    parser.add_argument('--hist-plot-file-name', default=None, nargs='+',
                        metavar=('IMAGE_FILE'), required=False, type=str,
                        dest="hist_plot_file_name",
                        help='File names for x,y,z histogram plots')

    parser.add_argument('--debug', default=False, required=False,
                        action='store_true', dest="debug", help='debug flag')
    parser.add_argument('--verbose', default=False, required=False,
                        action='store_true', dest="verbose", help='verbose flag')
    parser.add_argument('--log', required=False, nargs='?', dest="log",
                        default=None, const='c2d.log', metavar='LOG',
                        help='Write log file c2d.log, optionally specify log file name')

    try:
        import argcomplete
        argcomplete.autocomplete(parser)
        # This supports bash autocompletion. To enable this, pip install
        # argcomplete, activate global completion, or add
        #      eval "$(register-python-argcomplete lpad)"
        # into your .bash_profile or .bashrc
    except ImportError:
        pass

    args = parser.parse_args()


    if args.debug:
        loglevel = logging.DEBUG
    elif args.verbose:
        loglevel = logging.INFO
    else:
        loglevel = logging.WARNING

    # PoissonNernstPlanckSystem makes extensive use of Python's logging module

    # logformat  = ''.join(("%(asctime)s",
    #  "[ %(filename)s:%(lineno)s - %(funcName)s() ]: %(message)s"))
    logformat  = "[ %(filename)s:%(lineno)s - %(funcName)s() ]: %(message)s"

    logging.basicConfig(level=loglevel,
                        format=logformat)

    # explicitly modify the root logger (necessary?)
    logger = logging.getLogger()
    logger.setLevel(loglevel)

    # remove all handlers
    for h in logger.handlers: logger.removeHandler(h)

    # create and append custom handles
    ch = logging.StreamHandler()
    formatter = logging.Formatter(logformat)
    ch.setFormatter(formatter)
    ch.setLevel(loglevel)
    logger.addHandler(ch)

    if args.log:
        fh = logging.FileHandler(args.log)
        fh.setFormatter(formatter)
        fh.setLevel(loglevel)
        logger.addHandler(fh)

    logger.info('This is `{}` : `{}`.'.format(__file__,__name__))

    # input verification
    if args.hist_plot_file_name:
        if len(args.hist_plot_file_name) == 1:
            hist_plot_file_name_prefix, hist_plot_file_name_ext = os.path.splitext(
                args.hist_plot_file_name[0])
            hist_plot_file_name = [
                hist_plot_file_name_prefix + '_' + suffix + hist_plot_file_name_ext
                for suffix in ('x','y','z') ]
        elif len(args.hist_plot_file_name) == 3:
            hist_plot_file_name = args.hist_plot_file_name
        else:
            raise ValueError(
            """If specifying histogram plot file names, please give either one
            file name to be suffixed with '_x','_y','_z' or three specific file
            names.""")
    else:
        hist_plot_file_name = None

    box = np.array(args.box)

    # get Python function from function name string:
    # (from https://github.com/materialsproject/fireworks/blob/master/fireworks/user_objects/firetasks/script_task.py)
    # toks = args.distribution.rsplit('.', 1)
    # if len(toks) == 2:
    #    modname, funcname = toks
    #    mod = __import__(modname, globals(), locals(), [str(funcname)], 0)
    #    func = getattr(mod, funcname)
    # else:
    #    # Handle built in functions.
    #    func = getattr(builtins, toks[0])

    if not args.infile:
        infile = sys.stdin
        infile_format  = '.txt'
    else:
        infile = args.infile
        _, infile_format = os.path.splitext(infile)

    if infile_format == '.npz':
        file = np.load(infile)
        x = file['x']
        u = file['u']
        c = file['c']
    else: # elif infile_format == 'txt'
        data = np.loadtxt(infile, unpack=True)
        x = data[0,:]
        u = data[1,:]
        c = data[2:,:]

    if c.ndim > 1:
        C = [ c[k,:] for k in range(c.shape[0]) ]
    else:
        C = [c]

    del c

    logger.info('Read {:d} concentration distributions.'.format(len(C)))
    # dist = [ lambda X: interpolate.interp1d(x,c)(X) for c in C ]

    # TODO: sample size based on true concentratins and dimesnions
    # sample_size = int(np.round(box.prod()*pnp.c[0] * sc.Avogadro))
    # [V] = m^3, [c] = mol / m^3, [N_A] = 1 / mol

    sample_size = args.sample_size if isinstance(args.sample_size,list) else [args.sample_size]
    ngridpoints = args.ngridpoints if isinstance(args.ngridpoints,list) else [args.ngridpoints]
    sample_size = sample_size*len(C) if len(sample_size) == 1 else sample_size
    ngridpoints = ngridpoints*len(C) if len(ngridpoints) == 1 else ngridpoints

    logger.info('Generating structure from distribution ...')
    struc = [ generate_structure(
                distribution=interpolate.interp1d(x,c),
                box=box, atom_count=s)
                for c,s in zip(C,sample_size) ]

    logger.info('Generated {:d} coordinate sets.'.format(len(struc)))

    logger.info('Creating ase.Atom objects ...')
    system = ase.Atoms(
        cell=box/sc.angstrom,
        pbc=[1,1,0])

    for i, s in enumerate(struc):
        logger.info('{:d} samples in coordinate set {:d}.'.format(len(s),i))
        system += ase.Atoms(
            symbols=args.names[i]*sample_size[i],
            charges=[args.charges[i]]*sample_size[i],
            positions=s/sc.angstrom)

    logger.info('Writing output file ...')

    if not args.outfile:
        outfile = sys.stdout
        outfile_format  = 'xyz'
    else:
        outfile = args.outfile
        _, outfile_format = os.path.splitext(outfile)

    logger.info('Output format {} to {}.'.format(outfile_format,outfile))

    if outfile_format == '.lammps':
        # LAMMPS data format, units 'real', atom style 'full'
        # weird behavior of ase output
        if isinstance(outfile, str):
            with open(outfile,'wb') as f:
                ase.io.write(
                    f,system,format='lammps-data',units="real",atom_style='full')
        else:
            ase.io.write(
                outfile,system,format='lammps-data',units="real",atom_style='full')
    else: # elif outfile_format == '.xyz'
        ase.io.write(outfile,system,format='xyz')


    # export_xyz( struc, outfile_name=args.outfile,
    #            box=box, atom_name=args.atom_name)

    # only if requested
    if hist_plot_file_name:
        print('Plotting distribution histograms ...')
        histx, histy, histz = get_histogram(struc, box=box, n_bins=args.nbins)

        plot_dist(histx, hist_plot_file_name[0], reference_distribution=uniform)
        plot_dist(histy, hist_plot_file_name[1], reference_distribution=uniform)
        plot_dist(histz, hist_plot_file_name[2], reference_distribution=func)

    logger.info('Done.')

if __name__ == '__main__':
    # Run doctests1
    test()
    # Execute everything else
    main()
