__author__ = 'Abuenameh'

import pyalps
import numpy as np
from collections import OrderedDict
import time

basename = 'Tasks/bh'+str(time.time())

#prepare the input parameters
# parms = [ {
#         'LATTICE_LIBRARY'           : 'lattice.xml',
#         'LATTICE'                   : 'inhomogeneous open chain lattice',
#         # 'LATTICE'                   : 'open chain lattice',
#         'MODEL'                     : 'boson Hubbard',
#         'L'                         : 5,
#         'CONSERVED_QUANTUMNUMBERS'  : 'N',
#         'N_total'                   : 5,
#         'Nmax'                      : 5,
#         # 't'                         : 0.01,
#         't0'                         : 0.01,
#         't1'                         : 0.02,
#         't2'                         : 0.03,
#         't3'                         : 0.02,
#         # 'U'                         : 1,
#         'U0'                         : 1,
#         'U1'                         : 2,
#         'U2'                         : 3,
#         'U3'                         : 1,
#         'U4'                         : 2,
#         'SWEEPS'                    : 4,
#         'NUMBER_EIGENVALUES'        : 1,
#         'MAXSTATES'                 : 200
#        } ]

L = 50

parms = OrderedDict()
parms['LATTICE_LIBRARY'] = 'lattice.xml'
parms['LATTICE'] = 'inhomogeneous open chain lattice'
# parms['LATTICE'] = 'open chain lattice'
parms['MODEL'] = 'boson Hubbard'
parms['L'] = L
parms['CONSERVED_QUANTUMNUMBERS'] = 'N'
parms['Nmax'] = 5
parms['SWEEPS'] = 80
parms['NUMBER_EIGENVALUES'] = 1
parms['MAXSTATES'] = 200
parms['MEASURE_LOCAL[Local density]'] = 'n'

np.random.seed(0)
t = (1 + 0.5 * np.random.uniform(-1, 1, L-1)) * 0.01 #[0.01, 0.02, 0.03, 0.02]
U = (1 + 0.5 * np.random.uniform(-1, 1, L)) * 1 #[1, 2, 3, 1, 2]
parms['ts'] = ["{:.20f}".format(ti) for ti in t]
parms['Us'] = ["{:.20f}".format(Ui) for Ui in U]
for i in range(L-1):
    parms['t'+str(i)] = t[i]
for i in range(L):
    parms['U'+str(i)] = U[i]

parms['N_total'] = L

#write the input file and run the simulation
input_file = pyalps.writeInputFiles(basename,[parms])
res = pyalps.runApplication('mps_optim',input_file,writexml=True)

#load all measurements for all states
data = pyalps.loadEigenstateMeasurements(pyalps.getResultFiles(prefix=basename))

print(data)