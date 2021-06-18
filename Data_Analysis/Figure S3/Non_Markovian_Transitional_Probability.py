"""
Non_Markovian_Transitional_Probability.py finds non-Markovian lineages and calculate their transitional probability
matrices
"""

import scipy.optimize
import pickle
import math
import scipy
import statsmodels
import statsmodels.api as sm
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv
from scipy import stats


__author__ = 'Tee Udomlumleart'
__maintainer__ = 'Tee Udomlumleart'
__email__ = ['teeu@mit.edu', 'salilg@mit.edu']
__status__ = 'Production'


def euclidean_distance(coor_1, coor_2):
    return math.sqrt(sum((i - j) ** 2 for i, j in zip(coor_1, coor_2)))


def vector_size(x_displacement, y_displacement):
    return math.sqrt(x_displacement ** 2 + y_displacement ** 2)


# initialize and normalize data
all_barcode_set = set()

total_cell_number = 10 ** 6

state_1_ratio = 0.90
state_2_ratio = 0.05
state_3_ratio = 0.05

state_1_number = state_1_ratio * total_cell_number
state_2_number = state_2_ratio * total_cell_number
state_3_number = state_3_ratio * total_cell_number

normalizing_factor = [total_cell_number, state_1_number, state_2_number, state_3_number] * 10

table = pickle.load(open('191012_finished_table.pickle', 'rb'))

sum_table = table.sum(axis=0)
normalized_table = table.div(sum_table)
true_number_table = (normalized_table * normalizing_factor).round()

states = ['s1', 's2', 's3']
states_coords = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

timepoints = ['d0', 'd6', 'd12', 'd18', 'd24']

all_size_set = set()
all_barcode_list = []
all_transition_size_list = []

top_right_coord = (10, 10)
top_left_coord = (0, 10)
bottom_left_coord = (0, 0)

triangle_vertices = np.array([top_right_coord, top_left_coord, bottom_left_coord])

for barcode, row in true_number_table.iterrows():
    barcode_dict = {}

    barcode_dict['d0_all'], barcode_dict['d0_s1'], barcode_dict['d0_s2'], barcode_dict['d0_s3'] = row[0:4]
    barcode_dict['d6_all'], barcode_dict['d6_s1'], barcode_dict['d6_s2'], barcode_dict['d6_s3'] = row[4:8]
    barcode_dict['d12_all'], barcode_dict['d12_s1'], barcode_dict['d12_s2'], barcode_dict['d12_s3'] = row[20:24]
    barcode_dict['d18_all'], barcode_dict['d18_s1'], barcode_dict['d18_s2'], barcode_dict['d18_s3'] = row[32:36]
    barcode_dict['d24_all'], barcode_dict['d24_s1'], barcode_dict['d24_s2'], barcode_dict['d24_s3'] = row[36:40]

    barcode_summary = {'ternary_coord': [], 'cartesian_coord': [], 'vector': [], 'size': [], 'assigned_state': [],
                       'timepoint_size': [], 'total_transition_amount': 0, 'id': None}

    barcode_size = [sum(row[1:4]), sum(row[5:8]), sum(row[21:24]), sum(row[33:36]), sum(row[37:40])]

    for timepoint in timepoints:
        timepoint_all_present = all(barcode_size)
        timepoint_total = sum([barcode_dict[timepoint + '_' + state] for state in states])
        if timepoint_total:
            ternary_coord = []
            timepoint_size = []
            dist = []
            for state in states:
                ternary_coord.append(barcode_dict[timepoint + '_' + state] / timepoint_total)
                timepoint_size.append(barcode_dict[timepoint + '_' + state])
            barcode_summary['ternary_coord'].append(ternary_coord)
            barcode_summary['timepoint_size'].append(timepoint_size)

            cartesian_coord = np.dot(np.array(ternary_coord), triangle_vertices)
            barcode_summary['cartesian_coord'].append(list(cartesian_coord))

            for state_coord in triangle_vertices:
                dist.append(euclidean_distance(cartesian_coord, state_coord))
            barcode_summary['assigned_state'].append(dist.index(min(dist)))

            barcode_summary['size'].append(timepoint_total)

    if len(barcode_summary['cartesian_coord']) == 5:
        remove_flag = False
        for i in range(4):
            vector = (barcode_summary['cartesian_coord'][i + 1][0] - barcode_summary['cartesian_coord'][i][0],
                      barcode_summary['cartesian_coord'][i + 1][1] - barcode_summary['cartesian_coord'][i][1])
            barcode_summary['vector'].append(vector)
            barcode_summary['total_transition_amount'] += vector_size(vector[0], vector[1])
        all_transition_size_list.append(barcode_summary['total_transition_amount'])
        for size in barcode_summary['size']:
            all_size_set.add(round(size, 3))
            if size < 1:
                remove_flag = True
        if not remove_flag:
            all_barcode_list.append(barcode_summary)

# sort the barcode based on their total transition amount
all_barcode_list.sort(reverse=True, key=lambda barcode: barcode['total_transition_amount'])


# function estimating lineage-specific transitional probability matrix
def least_square_estimation_all_lineage(all_barcode_list):
    probability_lineage_list = []
    for barcode in all_barcode_list:
        ternary_coord = barcode['ternary_coord']
        T_0 = np.zeros((4, 3))
        T_1 = np.zeros((4, 3))
        for timepoint in range(4):
            T_0[timepoint] = np.array(ternary_coord[timepoint])
            T_1[timepoint] = np.array(ternary_coord[timepoint + 1])
        T_0_t = np.transpose(T_0)
        probability_lineage_list.append(np.matmul(pinv(np.matmul(T_0_t, T_0)), np.matmul(T_0_t, T_1)))
    return probability_lineage_list


LSE_lineage_specific = least_square_estimation_all_lineage(all_barcode_list)
p_value_list = [[] for i in range(4)]  # a list of p-values for each timepoint
lineage_number_list = [[] for i in range(4)]  # a list of lineage that gets chi-square tested
bool_list = []

# iterate across all lineages
for index, barcode in enumerate(all_barcode_list):
    # retrive lineage-specific transitional probability matrix
    transitional_probability = LSE_lineage_specific[index]
    # iterate through each timepoint
    for timepoint in range(4):
        probability_lineage_list = []
        ternary_coord = barcode['ternary_coord']
        T_0 = np.zeros((1, 3))
        T_1 = np.zeros((1, 3))
        T_0[0] = np.array(ternary_coord[timepoint])
        T_1[0] = np.array(ternary_coord[timepoint + 1])
        # find the expected state distribution on the next timepoint
        expected_distribution = np.matmul(T_0, transitional_probability)
        # find the expected distribution of number of cells on the next timepoint
        expected_T_1 = (expected_distribution * barcode['size'][timepoint + 1]).round(0)[0]
        # the actual observed number of cells
        actual_T_1 = np.array(barcode['timepoint_size'][timepoint + 1]).round(0)
        # put them into a matrix
        matrix = pd.DataFrame([actual_T_1, expected_T_1])

        # remove columns with all zeros
        matrix[matrix < 0] = 0
        matrix = matrix.loc[:, (matrix != 0).any(axis=0)]

        # only perform chi-square test when there are more than 1 columns
        if matrix.iloc[1, :].all() and matrix.shape[1] > 1:
            res = scipy.stats.chisquare(matrix.iloc[0, :], matrix.iloc[1, :])
            p_value_list[timepoint].append(res[1])  # keep p-value in the p-value list
            lineage_number_list[timepoint].append(index)  # keep the lineage number

# iterate through each timepoint
for timepoint in range(4):
    # set the default lineage state to be uninformative unless they are chi-square tested
    p_value_lineage_list = ['Uninformative' for lineage in all_barcode_list]

    # perform Benjamini-Hochberg p-value correction
    l = statsmodels.stats.multitest.multipletests(p_value_list[timepoint], method='fdr_bh')[0]

    # iterate across all lineages that are tested
    for lineage, p_value in zip(lineage_number_list[timepoint], l):
        if p_value:  # p_value = True when the null hypothesis is rejected
            p_value_lineage_list[lineage] = 'Non-Markovian'
        else:  # else, null hypothesis is accepted
            p_value_lineage_list[lineage] = 'Markovian'

    bool_list.append(p_value_lineage_list)

# turn a bool_list into a dataframe
df = pd.DataFrame(bool_list)
df = df.T
df = df.rename(columns={i: 'D{} to D{}'.format(i*6, (i+1)*6) for i in range(5)})

all_non_markov_list = []

# find lineages with all non-Markovian transition
for barcode, row in df.iterrows():
    non_markov_num = list(row).count('Non-Markovian')
    markov_num = list(row).count('Markovian')
    none_num = list(row).count('Uninformative')
    if none_num >= 0 and markov_num == 0 and non_markov_num > 0:
        all_non_markov_list.append(barcode)  # keep the id of lineage that have all non_Markovian transitions

# initialize a list of lists where each inner list contains a data for each matrix entry (there are 9 inner lists)
data_list = [[] for i in range(9)]
for index, barcode in enumerate(all_barcode_list):
    if index in all_non_markov_list:  # only perform calculation on all non_markovian lineages

        # let Ax = B where
        # A is a pre-transition matrix
        # B is a post-transition matrix
        # x is a transitional probability matrix
        A = np.zeros((15, 9))
        B = np.zeros((15, 1))
        ternary_coord = barcode['ternary_coord']
        for timepoint in range(5):
            if timepoint < 4:
                for state in range(3):
                    # put these values so that it satisfies the original equation
                    A[timepoint * 3 + state, :] = [0, 0, 0] * state + ternary_coord[timepoint] + [0, 0, 0] * (2 - state)
                    B[timepoint * 3 + state, :] = ternary_coord[timepoint + 1][state]
            else:
                # make sure that all rows sum to 1
                for state in range(3):
                    A[timepoint * 3 + state, :] = [0, 0, 0] * state + [1, 1, 1] + [0, 0, 0] * (2 - state)
                    B[timepoint * 3 + state, :] = 1
        # estimate the parameters using bounded least square estimation
        x = scipy.optimize.lsq_linear(A, B[:, 0], bounds=(0, 1))['x']
        # put each matrix value in a matrix
        for ientry, entry in enumerate(x):
            data_list[ientry].append(entry)

# plot the data
fig, axs = plt.subplots(3, 3, figsize=(6*2,4*2), sharex=True, sharey=True)
for irow in range(3):
    for icol in range(3):
        entry = 3*irow + icol
        ax = axs[irow, icol]
        sns.distplot(x=data_list[entry], ax=ax, kde=True, rug=True, hist=False, color='black')
        ax.axvline(x=np.mean(data_list[entry]), color='#686868')
        print(np.round(np.mean(data_list[entry]), 2))
        ax.set_xlim(-0.5, 1.5)
plt.savefig('Non_Markov_RightStochastic.tiff', format='tiff', dpi=720)
