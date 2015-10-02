__author__ = 'Abuenameh'

import pyalps
import numpy as np
import matplotlib.pyplot as plt
import pyalps.plot

#prepare the input parameters
parms = [ {
        'optimization'              : 'singlesite',
        'LATTICE'                   : 'open chain lattice',
        'L'                         : 20,
        'MODEL'                     : 'spin',
        'local_S0'                  : '0.5',
        'local_S1'                  : '1',
        'CONSERVED_QUANTUMNUMBERS'  : 'N,Sz',
        'Sz_total'                  : 9,
        'J'                         : 1,
        'SWEEPS'                    : 4,
        'NUMBER_EIGENVALUES'        : 1,
        'MAXSTATES'                 : 50,
        'MEASURE_LOCAL[Spin]'       : 'Sz',
        # 'init_state'                : 'local_quantumnumbers',
        # 'initial_local_Sz'          : ','.join(['0.5']*10+['-0.5']*1+['0.5']*9),#'0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,-0.5',#'1,0,0,0,0,0,0,0,0,0',
        # 'initial_local_S'           : ','.join(['0.5']*20+['-0.5']*0),#'0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,-0.5',#'1,0,0,0,0,0,0,0,0,0',
       } ]

#write the input file and run the simulation
input_file = pyalps.writeInputFiles('SingleSite3/parm_spin_one',parms)
res = pyalps.runApplication('mps_optim',input_file,writexml=True)

#load all measurements for all states
data = pyalps.loadEigenstateMeasurements(pyalps.getResultFiles(prefix='SingleSite3/parm_spin_one'))

# print properties of the eigenvector:
for s in data[0]:
    print s.props['observable'], ' : ', s.y[0]

