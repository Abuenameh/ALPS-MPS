__author__ = 'Abuenameh'

import pyalps
import numpy as np
import matplotlib.pyplot as plt
import pyalps.plot
import shutil
import os

try:
    shutil.rmtree('SingleSite2')
    os.mkdir('SingleSite2')
except:
    pass

#prepare the input parameters
parms = [ {
        'LATTICE'                   : "open chain lattice",
        'MODEL'                     : "spin",
        'CONSERVED_QUANTUMNUMBERS'  : 'Sz',
        'Sz_total'                  : 1,
        'J'                         : 1,
        'SWEEPS'                    : 12,
        'NUMBER_EIGENVALUES'        : 1,
        'L'                         : 30,
        'MAXSTATES'                 : 100,
        'MEASURE_LOCAL[Local spin]' : 'Sz',
        # 'initfile'                  : 'SingleSite2/parm_spin_one_half.task1.out.chkp',
        # 'init_state'                : 'local_quantumnumbers',
        # 'initial_local_Sz'          : ','.join(['0.5']*6+['-0.5']*4),#'0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,-0.5',#'1,0,0,0,0,0,0,0,0,0',
        # 'initial_local_S'           : ','.join(['0.5']*10+['-0.5']*0),#'0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,-0.5',#'1,0,0,0,0,0,0,0,0,0',
        'optimization'              : 'singlesite',
        # 'ngrowsweeps'               : 4,
        # 'nmainsweeps'               : 4,
       } ]

#write the input file and run the simulation
input_file = pyalps.writeInputFiles('SingleSite2/parm_spin_one_half',parms)
res = pyalps.runApplication('mps_optim',input_file,writexml=True)

#load all measurements for all states
data = pyalps.loadEigenstateMeasurements(pyalps.getResultFiles(prefix='SingleSite2/parm_spin_one_half'))

# print properties of the eigenvector:
for s in data[0]:
    print s.props['observable'], ' : ', s.y[0]


