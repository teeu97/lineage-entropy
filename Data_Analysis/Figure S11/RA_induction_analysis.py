"""
RA_induction_analysis.py analyzes the reads from RA experiement
"""

import pickle
import math
import matplotlib
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

__author__ = 'Tee Udomlumleart'
__maintainer__ = 'Tee Udomlumleart'
__email__ = ['teeu@mit.edu', 'salilg@mit.edu']
__status__ = 'Production'


def euclidean_distance(coor_1, coor_2):
    return math.sqrt(sum((i - j) ** 2 for i, j in zip(coor_1, coor_2)))


def vector_size(x_displacement, y_displacement):
    return math.sqrt(x_displacement ** 2 + y_displacement ** 2)

# normalize reads
total_cell_number = 10 ** 8

state_1_ratio = 0.90
state_2_ratio = 0.05
state_3_ratio = 0.05

state_1_number = state_1_ratio * total_cell_number
state_2_number = state_2_ratio * total_cell_number
state_3_number = state_3_ratio * total_cell_number

normalizing_factor = [total_cell_number, state_1_number, state_2_number, state_3_number] * 3 + \
                     [10 ** 6, 3 * 10 ** 6] + [0] * 26   # normalize APC- samples to 1 million cells and APC+ to 3 million cells
                    # and ignore the rest of the samples

table = pickle.load(open('210416_finished_table.pickle', 'rb'))

sum_table = table.sum(axis=0) + 1
normalized_table = np.divide(table, sum_table)
true_number_table = (normalized_table * normalizing_factor).round()

states = ['s1', 's2', 's3']
states_coords = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

timepoints = ['d0', 'd6', 'd12']

all_size_set = set()
all_barcode_dict = {}
all_barcode_set = set()
all_barcode_list = []
all_transition_size_list = []

right_extreme_barcode_list = []
left_extreme_barcode_list = []

top_right_coord = (1, 1)
top_left_coord = (0, 1)
bottom_left_coord = (0, 0)

triangle_vertices = np.array([top_right_coord, top_left_coord, bottom_left_coord])

for barcode, row in true_number_table.iterrows():
    barcode_dict = {}

    barcode_dict['d0_all'], barcode_dict['d0_s1'], barcode_dict['d0_s2'], barcode_dict['d0_s3'] = row[0:4]
    barcode_dict['d6_all'], barcode_dict['d6_s1'], barcode_dict['d6_s2'], barcode_dict['d6_s3'] = row[4:8]
    barcode_dict['d12_all'], barcode_dict['d12_s1'], barcode_dict['d12_s2'], barcode_dict['d12_s3'] = row[8:12]
    barcode_dict['APC-'], barcode_dict['APC+'] = row[12:14]

    barcode_summary = {'ternary_coord': [], 'cartesian_coord': [], 'vector': [], 'size': [], 'assigned_state': [],
                       'total_transition_amount': 0, 'transition_amount': [], 'APC+': 0, 'APC-': 0}

    barcode_size = [sum(row[1:4]), sum(row[5:8]), sum(row[9:12])]

    for timepoint in timepoints:
        timepoint_all_present = all(barcode_size)
        timepoint_total = sum([barcode_dict[timepoint + '_' + state] for state in states])
        if timepoint_total:
            ternary_coord = []
            dist = []
            for state in states:
                ternary_coord.append(barcode_dict[timepoint + '_' + state] / timepoint_total)
            barcode_summary['ternary_coord'].append(ternary_coord)

            cartesian_coord = np.dot(np.array(ternary_coord), triangle_vertices)
            barcode_summary['cartesian_coord'].append(list(cartesian_coord))

            for state_coord in triangle_vertices:
                dist.append(euclidean_distance(cartesian_coord, state_coord))
            barcode_summary['assigned_state'].append(dist.index(min(dist)))

            barcode_summary['size'].append(timepoint_total)

    if len(barcode_summary['cartesian_coord']) == 3:
        for i in range(2):
            vector = (barcode_summary['cartesian_coord'][i + 1][0] - barcode_summary['cartesian_coord'][i][0],
                      barcode_summary['cartesian_coord'][i + 1][1] - barcode_summary['cartesian_coord'][i][1])
            barcode_summary['vector'].append(vector)
            barcode_summary['total_transition_amount'] += vector_size(vector[0], vector[1])
            barcode_summary['transition_amount'].append(vector_size(vector[0], vector[1]))
        barcode_summary['APC-'], barcode_summary['APC+'] = barcode_dict['APC-'], barcode_dict['APC+']
        # only consider barcodes that have reads in both APC+ and APC- samples
        if barcode_summary['APC-'] != 0 and barcode_summary['APC+'] != 0:
            all_transition_size_list.append(barcode_summary['total_transition_amount'])
            for size in barcode_summary['size']:
                all_size_set.add(round(size, 3))
            all_barcode_list.append(barcode_summary)
            all_barcode_dict[barcode] = barcode_summary
            all_barcode_set.add(barcode)

# sirt barcodes based on their transition amount
all_barcode_list.sort(reverse=True, key=lambda barcode: barcode['total_transition_amount'])

barcode_number = len(all_barcode_list)

timepoint_list = [[[] for i in range(3)] for j in range(3)]

# separate barcodes based on their assigned states in all timepoints
for timepoint in range(3):
    for barcode in all_barcode_list:
        ternary_coord = barcode['ternary_coord'][timepoint]
        if barcode['APC-'] != 0:
            # add log(APC ratio) to the list
            timepoint_list[timepoint][np.argmax(ternary_coord)].append(np.log10(barcode['APC+'] / barcode['APC-']))

# function that removes the vertical line at the end of CDF
def fix_hist_step_vertical_line_at_end(ax):
    axpolygons = [poly for poly in ax.get_children() if isinstance(poly, matplotlib.patches.Polygon)]
    for poly in axpolygons:
        poly.set_xy(poly.get_xy()[:-1])

# plot CDF of total transition amount
fig, ax = plt.subplots()

rainbow = cm.get_cmap('rainbow_r', 10)
rainbow_list = rainbow(range(10))

# add APC+/APC- ratio to the list
APC_ratio_list = []
for barcode in all_barcode_list:
    if barcode['APC-'] != 0:
        APC_ratio_list.append(barcode['APC+'] / barcode['APC-'])

# find log(APC_ratio)
APC_ratio_list = np.array(APC_ratio_list)
APC_ratio_list = APC_ratio_list[APC_ratio_list > 0]
APC_ratio_list = np.log10(APC_ratio_list)

# plot CDF of APC ratio from all lineages
plt.hist(APC_ratio_list, bins=1000, density=True, histtype='step', cumulative=True, color='black', label='All lineages')

all_var = np.var(APC_ratio_list, ddof=1)

quantile_var = []
len_list = []
# plot CDF of APC ratio from different quantiles
for quantile, i in enumerate(np.arange(0, 1, 0.1)):
    quantile_population = all_barcode_list[round(i * barcode_number):round((i + 0.1) * barcode_number)]
    APC_ratio_list = []
    for barcode in quantile_population:
        if barcode['APC-'] != 0:
            APC_ratio_list.append(barcode['APC+'] / barcode['APC-'])

    APC_ratio_list = np.array(APC_ratio_list)
    APC_ratio_list = APC_ratio_list[APC_ratio_list > 0]
    APC_ratio_list = np.log10(APC_ratio_list)

    plt.hist(APC_ratio_list, bins=10000, density=True, histtype='step', cumulative=True, color=rainbow_list[quantile],
             label='Decile {}'.format(quantile + 1))

    len_list.append(len(APC_ratio_list))
    quantile_var.append(np.var(APC_ratio_list, ddof=1))

# use F-test to calculate variance differences across different group
for index, q_var in enumerate(quantile_var):
    alpha = 0.05
    F = q_var / all_var
    df1 = len_list[index] - 1
    df2 = len(all_barcode_list) - 1
    prob = scipy.stats.f.sf(F, df1, df2)
    print(index, F, prob)
    if prob < alpha / 2 or prob > 1 - (alpha / 2):  # if prob < alpha/2 or prob > 1 - (alpha/2) -> significant
        print('Significantly different: {}'.format(index))

fix_hist_step_vertical_line_at_end(ax)
plt.legend(loc='upper left')
plt.title('Quantile: Total Transition Amount')
plt.xlabel(r'$\log_{10}\frac{APC+}{APC-}$')
plt.ylabel('Cumulative Proportion')
plt.savefig('RAinduction_TransitionAmount.tiff', format='tiff', dpi=720)

# plot the distribution of APC ratio across different states on different timepoints
for timepoint in range(3):
    fig, ax = plt.subplots()
    for state in range(3):
        # plot the CDF across different states
        plt.hist(timepoint_list[timepoint][state], bins=100, density=True, histtype='step', cumulative=True,
                 label='State {}'.format(state + 1))

    # plot CDF from all lineages
    APC_ratio_list = []
    for barcode in all_barcode_list:
        if barcode['APC-'] != 0:
            APC_ratio_list.append(barcode['APC+'] / barcode['APC-'])

    APC_ratio_list = np.array(APC_ratio_list)
    APC_ratio_list = APC_ratio_list[APC_ratio_list > 0]
    APC_ratio_list = np.log10(APC_ratio_list)

    plt.hist(APC_ratio_list, bins=1000, density=True, histtype='step', cumulative=True, color='black',
             label='All lineages')

    plt.title('Day {}'.format(timepoint * 6))
    plt.legend(loc='upper left')
    plt.xlabel(r'$\log_{10}\frac{APC+}{APC-}$')
    plt.ylabel('Cumulative Proportion')
    plt.savefig('RAinduction_State_{}.tiff'.format(timepoint), format='tiff', dpi=720)
    plt.show()


