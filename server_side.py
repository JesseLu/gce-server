#! /usr/bin/env python

import sys
import my_info
import my_fields 
import my_steps
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import h5py
from jinja2 import Environment, PackageLoader
import time
 
# device = pycuda.autoinit.device
# print device.get_attributes()
# asdas[asd]
# Create a jinja2 environment which will be used to load templates.
jinja_env = Environment(loader=PackageLoader(__name__, 'templates'))

# Get the hdf5 filename.
filename = sys.argv[1]
file = h5py.File(filename, 'r+')

# Load the global parameters.
iters, stencil_size = my_info.read(filename)

# Load the fields onto the GPU.
print 'Loading fields... (STATUS)'
fields = my_fields.read(filename, stencil_size)

# Organize the operations into steps.
print 'Loading operations... (STATUS)'
steps = my_steps.read(filename)

# Compile all operations. 
print 'Compiling... (STATUS)'
start_time = time.clock()
my_steps.compile(steps, jinja_env, fields)
stop_time = time.clock()
# print 'Compilation took', stop_time - start_time, 'seconds.'

# Run the simulation!
print 'Executing... (STATUS)'
start_event = drv.Event()
start_event.record()
for s in steps:
    s.start_log(start_event)

start_time = time.clock()
for t in range(len(iters)):
    print t*100/len(iters), '% | t =', iters[t]
    for s in steps:
        s.run(iters[t], fields, file)
stop_time = time.clock()

# Save datalog
log = file.create_group('datalog')
for s in steps:
    s.write_log(log)

# Make sure all fields are written to the hdf5 file
for fname in list(fields):
    fields[fname].dump_to_hdf5(file)

file.close() # Close hdf5 file.

# Print our how we did in terms of speed.
print '\nTime needed to execute', iters.size, 'iterations:', \
    stop_time - start_time, 'seconds.'
print 'Speed:', iters.size/(stop_time-start_time), 'iterations per second.'

print int(stop_time - start_time), 'seconds to execute ', \
    iters.size, 'iterations. (RESULT)'

