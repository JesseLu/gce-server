import h5py
import pycuda.driver as drv
import numpy as np

class MyOperation:
    """ Stores the information needed for an operation. """

    def __init__(self, operation_name, dset):
        """ Initialize the MyOperation instance. """
        # General parameters.
        self.name = operation_name # Name of the operation.
        self.source = str(dset[0]) # The source code for the operation.
        self.type = str(dset.attrs['type'][0]) # Type of operation.

        # Time parameters.
        self.step = dset.attrs['step'][0] # Determines order of execution.
        self.exec_ind, self.save_ind = 0, 0 # Indexing variables.
        self.exec_tt = dset.attrs['exec_tt'] # Timesteps in which to run.
        if ('save_tt' in list(dset.attrs)):
            self.save_tt = dset.attrs['save_tt'] # Timesteps in which to save.
        else:
            self.save_tt = np.array([-2]) # Never save anything.

        # Space parameters. Define cube over which to execute the operation.
        self.shape = dset.attrs['shape'].astype(np.int32) 
        self.offset = dset.attrs['offset'].astype(np.float32) 
        self.spacing = dset.attrs['pitch'].astype(np.float32) 

        # Write-field parameters. 
        # These are the fields which are altered by the operation. 
        # This is used to automate parallelization in the multi-GPU case.
        self.write_fields = dset.attrs['updated_fields']

        # Optional user-defined parameter.
        if ('params' in list(dset.attrs)): 
            # Immediately load the function parameters into device memory.
            self.params = dset.attrs['params'].astype(np.float32)
            self.d_params = (drv.mem_alloc(self.params.nbytes),)
            drv.memcpy_htod(self.d_params[0], self.params)
        else:
            # No parameters specificied.
            self.params = np.array([]).astype(np.float32)
            self.d_params = ()
        
        # Allocate space for counter variable as well, for sum operations.
        if (self.type == 'sum'):
            counter = np.zeros(1).astype(np.uint32)
            self.d_counter = drv.mem_alloc(counter.nbytes) 
            drv.memcpy_htod(self.d_counter, counter)

        # Stream for the operation.
        self.stream = drv.Stream()
        self.beg = drv.Event()
        self.end = drv.Event()

        # Tells us whether we have decided on a run shape or not.
        self.is_locked = False

        # Data logging parameters.
        self.log_dirs = []
        self.log_shapes = []
        self.log_times = []

        # Describe what was loaded.
        print 'Loaded operation', self.name
        print 'type:', self.type
        print 'step:', self.step
        print 'exec_tt:', self.exec_tt
        print 'save_tt:', self.save_tt
        print 'shape:', self.shape
        print 'offset:', self.offset
        print 'pitch:', self.spacing
        print 'updated_fields:', self.write_fields
        print 'params:', self.params, '\n'

    def set_shapes(self):
        """ Create a list of run_shapes over which each operation can 
        optimize over. A run_shape is a 4-tuple containing: 
        (step_dir, block_xx, block_yy, block_zz). """
        shapes = []
        # Design decision: step in y-direction, and block_xx = 1 always.
        for j in range(10):
            for k in range(10):
                shapes.append((1, 1, 2**j, 2**k))

        self.run_shapes = []
        for rs in shapes:
            num_threads = rs[1] * rs[2] * rs[3]
            # Max number of cuda threads is 512, and I want each block 
            # to have at least a warp (32) of threads.
            if ((num_threads <= 512) and (num_threads >= 32) and \
                (rs[1] <= 64) and (rs[2] <= 512) and (rs[3] <= 512)):
                self.run_shapes.append(rs)
        self.run_times = []
        self.run_ind = 0 

    def render(self, jinja_env, step_dir):
        """ Compile the operation to run on the GPU with a certain thread block
            shape and step direction. """

        # Describe what we are rendering.
        print 'Rendering operation', self.name, '|',
        print 'increment dir:', step_dir

        # Order the block ID's so that we step through the grid in the 
        # correct way.
        dirs = ['x', 'y']
        dirs.insert(step_dir, 'z')
        custom_blockID = (dirs[:])

        # Construct the string for the input parameters.
        if (self.params.size == 0):
            params = ''
        else:
            params = ', float p'
            for k in self.params.shape:
                params += '[' + str(k) + ']'
            params = ', float *p'

        # Get the jinja2 template containing the source code. 
        template = jinja_env.get_template(self.type + '.cu')

        # Render the template.
        self.cuda_source = template.render(params=params, \
            function_name=self.name + 'XYZ'[step_dir], \
            source=self.source, custom_blockID=custom_blockID, \
            offset=self.offset, \
            limit=self.offset + (self.shape - 0.5) * self.spacing, \
            spacing=self.spacing, step_spacing=self.spacing[step_dir], \
            ijk_step='ijk'[step_dir], zyx_step='zyx'[step_dir], \
            result_field=self.write_fields[0])

    def is_active(self, t):
        """ Check if the operation should be executed at this time step. """
        if ((self.exec_tt[self.exec_ind] == -1) | \
            (self.exec_tt[self.exec_ind] == t)):
            return True
        else:
            return False
 
    def prepare(self, t):
        """ Set everything in order so that the operation can be executed. """
        if (self.is_locked == False): 

            # Choose the next run shape.
            if (len(self.run_times) == len(self.run_shapes)): 
                # Choose best run shape.
                run_shape = self.run_shapes[np.argmin(self.run_times)]
                step_dir = run_shape[0]
                block_shape = run_shape[1:]
                self.is_locked = True # Lock the run parameters.
            else:
                # Set run_shape (that is, step_dir and block_shape).
                step_dir = self.run_shapes[self.run_ind][0]
                block_shape = self.run_shapes[self.run_ind][1:]
                self.run_ind += 1

                # Log the step direction and block shape.
                self.log_dirs.append(step_dir)
                self.log_shapes.append(block_shape)


            self.active_runtime = self.runtime[step_dir]

            # Determine the number of blocks in the grid.
            grid_shape = [int(np.ceil(float(self.shape[i]) / block_shape[i])) \
                for i in range(3)]

            # The grid and block sizes to be passed to the actual function.
            self.grid_params = grid_shape[0:step_dir] + \
                grid_shape[step_dir+1:]
            block_params = block_shape[::-1]

            # Set the block shape.
            self.runtime[step_dir].set_block_shape(*block_params)

            if (self.type == 'sum'): # Sum operation.
                # Allocate an intermediate buffer.
                block_sums = \
                    np.empty(np.prod(self.grid_params)).astype(np.float32)
                self.d_block_sums = drv.mem_alloc(block_sums.nbytes) 

                # Allocate shared memory for the intermediate (partial) sums.
                shared_size = int(4 * np.prod(block_params))
                self.runtime[step_dir].set_shared_size(shared_size)

        # Set parameters.
        if (self.type == 'update'): # Update operation.
            self.active_runtime.param_set(t, *self.d_params)

        elif (self.type == 'sum'): # Sum operation.
            self.active_runtime.param_set(t, *(self.d_params + \
                (self.d_block_sums, self.d_counter)))



    def run(self, t):
        """ Initiate execution of operation. """
        if (self.is_locked):
            # If we have determined optimal parameters, run asynchronously. 
            self.active_runtime.launch_grid_async(\
                self.grid_params[0], self.grid_params[1], self.stream)
        else:
            # Used to time execution, to find optimal run parameters.
            self.beg.record()
            self.active_runtime.launch_grid(*self.grid_params)
            self.end.record()

        # Increment the execution timestep counter if needed.
        if ((self.exec_tt[self.exec_ind] != -1) & \
            (self.exec_ind < np.size(self.exec_tt) - 1)):
            self.exec_ind += 1

    def is_done(self):
        """ Determine if execution has completed. """ 
        if (self.is_locked):
            # For determining whether asynchronous execution has completed.
            if self.stream.is_done():
                return True
            else:
                return False
        else:
            # Log the execution time
            self.end.synchronize()
            self.exec_time = self.beg.time_till(self.end)
            self.log_times.append(self.exec_time)
            self.run_times.append(self.exec_time)

            return True

    def save(self, t, fields, file):
        """ Save write fields if needed. """
        # Check if the write fields should be saved at this time step.
        if ((self.save_tt[self.save_ind] == -1) | \
            (self.save_tt[self.save_ind] == t)):

            # Write all the fields to the hdf5 file.
            for field_name in self.write_fields:
                if (self.type == 'update'):
                    fields[field_name].save_to_host()
                fields[field_name].write_to_hdf5(file)

            # Increment the save timestep counter if needed.
            if ((self.save_tt[self.save_ind] != -1) & \
                (self.save_ind < np.size(self.save_tt) - 1)):
                self.save_ind += 1

    def write_log(self, log):
        """ Write the data log to the hdf5 file. """
        # Write the logged data.
        print 'Writing logged data from operation:', self.name, '...'

        # Create sub-group.
        oplog = log.create_group(self.name)

        # Make sure the operation was actually run atleast once.
        if (np.size(self.log_dirs) > 0):
            # Write data.
            oplog.create_dataset('dirs', data=self.log_dirs)
            oplog.create_dataset('shapes', data=self.log_shapes)
            oplog.create_dataset('times', data=self.log_times)

def read(filename):
    """ Read in the operations from an hdf5 file. """
    # Open the file
    file = h5py.File(filename, 'r')

    # Read in all the datasets in the operations group.
    operations = dict()
    for k in list(file['operations']):
        operations[k] = MyOperation(k, file['operations'][k])

    # Close the hdf5 file.
    file.close()

    return operations

