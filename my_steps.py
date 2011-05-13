import my_operations
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule
import time

class MyStep:
    """ Manages and executes steps, which are groups of operations which are
        executed simultaneously. """

    def __init__(self, step_num, opnames, operations):
        """ Constructor. Copy relevant operations into class instance. """
        # List the operations from largest to smallest, so that we run the
        # largest first. 
        op_sizes = [(np.prod(operations[op].shape), op) for op in opnames]
        self.operations = [operations[k[1]] \
            for k in sorted(op_sizes, reverse=True)]
        self.num = step_num

        self.log_times = [] # Keep track of step duration.
        self.beg = drv.Event()
        self.end = drv.Event()

        # Set run shapes for all operations in this step.
        for op in self.operations:
            op.set_shapes()

        print 'Initialized step', self.num, ':', \
            [op.name for op in self.operations]

    def start_log(self, start_event):
        """ Load the reference timer that enables to track when the step
        starts and stops. """
        self.start_event = start_event

    def run(self, t, fields, file):
        """ Execute all operations within the step. """

        self.beg.record()

        # Assemble list of active operations.
        active_ops = [op for op in self.operations if op.is_active(t)]

        # Prepare for execution.
        for op in active_ops:
            op.prepare(t)

        # Start execution.
        for op in active_ops:
            op.run(t)

        # Finish up execution.
        while active_ops:
            for op in active_ops:
                if op.is_done():
                    op.save(t, fields, file)
                    active_ops.remove(op)

        self.end.record()

        # Log run time.
        self.end.synchronize()
        self.log_times.append((self.beg.time_since(self.start_event), \
            self.end.time_since(self.start_event)))

    def write_log(self, log):
        """ Write the data log to the hdf5 file. """
        # Write the logged data.
        name = 'step' + str(self.num)
        print 'Writing logged data from', name, '...'

        # Create sub-group.
        oplog = log.create_group(name)

        # Make sure the step was actually run atleast once.
        if (np.size(self.log_times) > 0):
            # Write data.
            oplog.create_dataset('times', data=self.log_times)

        # Allow operations to log their data.
        for op in self.operations:
            op.write_log(log)

def read(filename):
    """ Read in the operations and organize them into steps. """
    # Read in the operations.
    operations = my_operations.read(filename)

    # Find all the step numbers used.
    step_nums = [operations[k].step for k in list(operations)]

    # Assemble the operations into individual steps.
    steps = []
    for k in range(min(step_nums), max(step_nums)+1):
        # Find the operations for step k.
        step_ops = []
        for l in list(operations):
            if (operations[l].step == k):
                step_ops.append(l)

        if (len(step_ops) > 0):
            # Make the MyStep class instances.
            steps.append(MyStep(k, step_ops, operations))

    print ''
    return steps

def compile(steps, jinja_env, fields):
    """ Combine all the operation source codes into one large string, compile
    it, load the constant values, and store runtime functions into each 
    operation instance. """

    print 'Compiling all operations...'

    # Get the jinja2 template containing the source code. 
    template = jinja_env.get_template('update.cu')

    # Assemble all the source into one long string.
    source = jinja_env.get_template('field_access_macros.cu').render( \
        field_names=sorted(fields), fields=fields)
    for step in steps:
        for op in step.operations:
            for step_dir in range(3):
                op.render(jinja_env, step_dir)
                source += str(op.cuda_source)
    
    # Write out source code, for debugging purposes.
    f = open('source_code.debug', 'w')
    f.write(source)
    f.close()

    # Compile the cuda source code.
    mod = SourceModule(source)

    # Load the location (on the GPU) of all the fields.
    dest, size = mod.get_global("M_FIELD")
    field_locations = \
        np.array([int(fields[fname].d_data) for fname in sorted(fields)])
    drv.memcpy_htod(dest, field_locations)

    # Store ready-to-run functions back into each My_Operation class instance.
    for step in steps:
        for op in step.operations:
            # Create the list to store runtimes for all 3 step directions.
            op.runtime = [] 
            for step_dir in range(3):
                op.runtime.append(mod.get_function(op.name + 'XYZ'[step_dir]))
                op.runtime[step_dir].set_cache_config(drv.func_cache.PREFER_L1)

    # Compiling finished.
    print '... compiling complete.', '\n'

