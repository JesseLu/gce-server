import h5py
import numpy as np
import pycuda.driver as drv
import math

class MyField:
    """ Store data regarding fields. """

    def __init__(self, field_name, dset, stencil_size):
        """ Initialize the MyField instance. """
        self.name = field_name # What the field is called.
        self.data = dset[:].astype(np.float32) # Get the field values.
        self.ind = 0 # A counter for the variable dimension.

        # Determine whether this is a global field or not.
        if (self.data.ndim == 4): # Standard 3-D field.
            self.isglobal = False
            self.global_dims = dset.shape[1:] # Get the 3D shape of the dataset.
            # Get the dataset values.
            self.data = np.reshape(dset[:].astype(np.float32), self.global_dims) 
            self.global_offset = dset.attrs['offset'][:].astype(np.float32)
            self.spacing = dset.attrs['pitch'][:].astype(np.float32)
            self.d_data = None # We haven't loaded the field onto the GPU yet.

            # Describe what was read in.
            print self.name, 'field loading...'
            print '\tdimensions:', self.global_dims
            print '\toffset:', self.global_offset
            print '\tpitch:', self.spacing

            # Add the padding necessary for automatic parallelization.
            self.pad_size = np.ceil(stencil_size / self.spacing).astype(np.int)
            self.local_dims = self.global_dims + 2 * self.pad_size
            self.local_offset = self.pad_size
            temp_data = np.zeros(self.local_dims).astype(np.float32)
            temp_data[ \
                self.local_offset[0]:self.local_offset[0]+self.global_dims[0], \
                self.local_offset[1]:self.local_offset[1]+self.global_dims[1], \
                self.local_offset[2]:self.local_offset[2]+self.global_dims[2]] \
                = self.data 
            self.data = temp_data

            # Copy array over to the device.
            self.d_data = drv.mem_alloc(self.data.nbytes) 
            drv.memcpy_htod(self.d_data, self.data)

        else: # Global field.
            self.isglobal = True

            # Describe what was read in.
            print 'Loading global field', self.name

            # Copy array over to the device.
            data = dset[:].astype(np.float32)
            self.data = drv.pagelocked_zeros_like(data, \
                drv.host_alloc_flags.DEVICEMAP) 
            self.d_data = self.data.base.get_device_pointer()
            self.data[:] = data[:]

            # To enable writing to the hdf5 file at the very end only.
            self.data_hist = []

    def save_to_host(self):
        """ Transfer the field's values on the gpu back to cpu memory. """
        drv.memcpy_dtoh(self.data, self.d_data)

    def write_to_hdf5(self, file):
        """ Write the field values to the hdf5 file. """
        if (self.isglobal): # Global field.
            self.data_hist.append(self.data[0])

        else: # Standard field, cut off padding before saving.
            file['fields/' + self.name].resize((self.ind+1,) + \
                self.global_dims[:])
            file['fields/' + self.name][self.ind,...] = self.data[ \
                self.local_offset[0]:self.local_offset[0]+self.global_dims[0], \
                self.local_offset[1]:self.local_offset[1]+self.global_dims[1], \
                self.local_offset[2]:self.local_offset[2]+self.global_dims[2]]

        # Incrementing the counter after the write implements the convention
        # of overwriting the initial values.
        self.ind = self.ind + 1

    def dump_to_hdf5(self, file):
        """ If this is a global field, finally write to the hdf5 file. """
        # if (self.isglobal and (len(self.data_hist) > 0)): # Global field.
        if (self.isglobal): # Global field.
            file['fields/' + self.name].resize((len(self.data_hist),))
            file['fields/' + self.name][:] = np.array(self.data_hist).flatten();
        

def read(filename, stencil_size):
    """ Read in the fields from an hdf5 file. """
    # Open the file
    file = h5py.File(filename, 'r')

    # Read in all the datasets in the fields group.
    fields = dict()
    for k in list(file['fields']):
        fields[k] = MyField(k, file['fields'][k], stencil_size)
    print ''

    # Close the hdf5 file.
    file.close()
    
    return fields
