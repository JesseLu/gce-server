import h5py
import numpy as np

def read(filename):
    """ Read in the global parameters from the hdf5 file. """

    # Open the file
    file = h5py.File(filename, 'r')

    # Read in all the datasets in the info group.
    print 'Reading in global parameters.'

    # The number of iterations to run the algorithm.
    iters = file['info']['t'][:].astype(np.float32)
    print 'iters:', iters

    # The stencil size of the simulation.
    stencil_size = file['info']['stencil'][:].astype(np.float32)
    print 'stencil_size:', stencil_size

    print ''

    # Close the hdf5 file.
    file.close()

    return iters, stencil_size

