"""
This file produces analyzes the markovianness in state transition on the population level
"""

import scipy.stats
import numpy as np
from numpy.linalg import pinv

__author__ = 'Tee Udomlumleart'
__maintainer__ = 'Tee Udomlumleart'
__email__ = ['teeu@mit.edu', 'salilg@mit.edu']
__status__ = 'Production'

# T1 contains the proportions of S1, S2, S3 (each col) on D0, D6, D12, D18 (each row)
# data acquired from the FACS plot
T1 = np.array([
    [89.5, 2.51, 1.38],
    [87.8, 2.88, 3.44],
    [87.6, 3.40, 2.62],
    [90.3, 3.06, 2.18]
])

# T2 contains the proportions of S1, S2, S3 (each col) on D6, D12, D18, D24 (each row)
# data acquired from the FACS plot
T2 = np.array([
    [87.8, 2.88, 3.44],
    [87.6, 3.40, 2.62],
    [90.3, 3.06, 2.18],
    [80.9, 2.83, 6.69]
])

# normalize each matrix
sum_T1 = T1.sum(axis=1)
normalized_T1 = T1 / sum_T1[:, np.newaxis]

sum_T2 = T2.sum(axis=1)
normalized_T2 = T2 / sum_T2[:, np.newaxis]

# calculate matrix M using least square estimation
M = pinv(normalized_T1.T@normalized_T1)@normalized_T1.T@normalized_T2

# use chi square to check if estimated M can predict the real proportion of cells on the later timepoint
for i in range(4):
    result = scipy.stats.chisquare((T1@M)[i, :], T2[i, :])
    print(result[1])
    print(result[1] > 0.05)