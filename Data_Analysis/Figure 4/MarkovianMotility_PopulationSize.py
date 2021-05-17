"""
This software calculates the Markovianness of each lineage based on their population size:

If the estimated transition matrix cannot precisely predict the state distribution in later timepoints in some of the
lineages, those lineages are non-markovian.

Otherwise, those lineages are markovian in that timepoint
"""

import pickle
import math
import scipy
import statsmodels
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib import cm
from numpy.linalg import pinv
from scipy import stats

__author__ = 'Tee Udomlumleart'
__maintainer__ = 'Tee Udomlumleart'
__email__ = ['teeu@mit.edu', 'salilg@mit.edu']
__status__ = 'Production'

# turn off irrelevent warnings
pd.options.mode.chained_assignment = None  # default='warn'


def euclidean_distance(coor_1, coor_2):
    return math.sqrt(sum((i - j) ** 2 for i, j in zip(coor_1, coor_2)))


def vector_size(x_displacement, y_displacement):
    return math.sqrt(x_displacement ** 2 + y_displacement ** 2)


# normalize reads
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

all_barcode_list.sort(reverse=True, key=lambda barcode: barcode['total_transition_amount'])

# make a list that contains 9 lists inside
matrix_data_list = [np.empty(0) for i in range(9)]  # store each entry of matrix in a list
bootstrap_sample_number = round(0.8*len(all_barcode_list))  # sample 80% of the total lineages
log_count = 0  # keep track of bootstrap iterations

# estimate lineage-specific transition matrix
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
p_value_list = [[] for i in range(4)]  # list that contains 4 lists -> each will contain p-values for each timepoint
# p-value from chi-square test to check if the empirical distribution of cells is significantly different from the
# simulation
lineage_number_list = [[] for i in range(4)]  # list that contains 4 lists ->
# each will contain the index of lineage that can be analyzed (some lineages are not analyzable)
bool_list = []  # this list will contain a list which represents whether a lineage is non-markovian (TRUE)
# or markovian (FALSE)

# iterate through each barcode
for index, barcode in enumerate(all_barcode_list):
    p_value_lineage_specific = []
    transitional_probability = LSE_lineage_specific[index]  # lineage specific transition matrix for that timepoint
    for timepoint in range(4):
        probability_lineage_list = []
        ternary_coord = barcode['ternary_coord']
        T_0 = np.zeros((1, 3))
        T_1 = np.zeros((1, 3))
        T_0[0] = np.array(ternary_coord[timepoint])
        T_1[0] = np.array(ternary_coord[timepoint + 1])
        # expected proportion = earlier proportion * transition matrix
        expected_distribution = np.matmul(T_0, transitional_probability)
        # expected cell distribution = total number of cells * expected proportion
        expected_T_1 = (expected_distribution * barcode['size'][timepoint + 1]).round(0)[0]
        actual_T_1 = np.array(barcode['timepoint_size'][timepoint + 1]).round(0)
        # put expected cell distributions and empirical cell distributions in a matrix
        matrix = pd.DataFrame([actual_T_1, expected_T_1])
        # negative number for cell number does not make biological sense -> turn it into 0
        matrix[matrix < 0] = 0
        # remove columns that are all 0s
        matrix = matrix.loc[:, (matrix != 0).any(axis=0)]

        # if a matrix shape >= 2x2 and not all 0s, perform chi-square test
        if matrix.iloc[1, :].all() and matrix.shape[1] > 1:
            res = scipy.stats.chisquare(matrix.iloc[0, :], matrix.iloc[1, :])
            p_value_list[timepoint].append(res[1])  # append p-value
            lineage_number_list[timepoint].append(index)  # record lineage id

# iterate through each timepoint
for timepoint in range(4):
    # initialize a data structure, 'uninformative' is a default unless there is an information that it is not
    # from the analysis above
    p_value_lineage_list = ['Uninformative' for lineage in all_barcode_list]
    non_markovian_percent_list = []

    # perform Benjamini-Hochberg correction
    l = statsmodels.stats.multitest.multipletests(p_value_list[timepoint], method='fdr_bh')[0]

    # If the test reveals statistical significance, that lineage is Non-Markovian; otherwise, Markovian
    for lineage, p_value in zip(lineage_number_list[timepoint], l):
        if p_value:
            p_value_lineage_list[lineage] = 'Non-Markovian'
        else:
            p_value_lineage_list[lineage] = 'Markovian'

    bool_list.append(p_value_lineage_list)

# turn the data structure into a dataframe
df = pd.DataFrame(bool_list)
df = df.T
df = df.rename(columns={i: 'D{} to D{}'.format(i*6, (i+1)*6) for i in range(5)})

# assign different types of lineage (Non-Markovian, Markovian, Uninformative) to some integers
# for making heatmap
value_to_int = {value: i for i, value in enumerate(sorted(pd.unique(df.values.ravel())))}
n = len(value_to_int)

color = cm.get_cmap('Set1', 9)
color_list = color(range(9))

cmap = matplotlib.colors.ListedColormap([color_list[i] for i in range(3)])

# make clustermap
clustermap = sns.clustermap(df.replace(value_to_int), cmap=cmap, cbar_pos=None, col_cluster=False, yticklabels=False,
                            metric='hamming')

for a in clustermap.ax_row_dendrogram.collections:
    a.set_linewidth(2.5)

for a in clustermap.ax_col_dendrogram.collections:
    a.set_linewidth(2.5)

colorbar = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap))
r = colorbar.vmax - colorbar.vmin
colorbar.set_ticks([colorbar.vmin + r / n * (0.5 + i) for i in range(n)])
colorbar.set_ticklabels(list(value_to_int.keys()))

plt.savefig('Markovian_Heatmap_Expected.tiff', format='tiff', dpi=720, bbox_inches='tight')