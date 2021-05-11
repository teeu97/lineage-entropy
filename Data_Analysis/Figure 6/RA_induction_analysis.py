import pickle
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from matplotlib import cm


def euclidean_distance(coor_1, coor_2):
    return math.sqrt(sum((i - j) ** 2 for i, j in zip(coor_1, coor_2)))


def vector_size(x_displacement, y_displacement):
    return math.sqrt(x_displacement ** 2 + y_displacement ** 2)


total_cell_number = 10 ** 8

state_1_ratio = 0.90
state_2_ratio = 0.05
state_3_ratio = 0.05

state_1_number = state_1_ratio * total_cell_number
state_2_number = state_2_ratio * total_cell_number
state_3_number = state_3_ratio * total_cell_number

normalizing_factor = [total_cell_number, state_1_number, state_2_number, state_3_number] * 3 + [10 ** 6,
                                                                                                3 * 10 ** 6] + [0] * 26

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
        if barcode_summary['APC-'] != 0 and barcode_summary['APC+'] != 0:
            APC_mean, APC_std = 0.4875082474644084, 1.013916978302975
            if barcode_summary['APC+'] / barcode_summary['APC-'] > 0:
                log_ratio = np.log10(barcode_summary['APC+'] / barcode_summary['APC-'])
                if log_ratio < APC_mean - 2 * APC_std:
                    left_extreme_barcode_list.append(barcode_summary)
                elif log_ratio > APC_mean + 2 * APC_std:
                    right_extreme_barcode_list.append(barcode_summary)
            all_transition_size_list.append(barcode_summary['total_transition_amount'])
            for size in barcode_summary['size']:
                all_size_set.add(round(size, 3))
            all_barcode_list.append(barcode_summary)
            all_barcode_dict[barcode] = barcode_summary
            all_barcode_set.add(barcode)

all_barcode_list.sort(reverse=True, key=lambda barcode: barcode['total_transition_amount'])

barcode_number = len(all_barcode_list)

from matplotlib import cm
from matplotlib import rc
import scipy
import scipy.stats

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

all_barcode_list.sort(reverse=True, key=lambda barcode: barcode['total_transition_amount'])


def fix_hist_step_vertical_line_at_end(ax):
    axpolygons = [poly for poly in ax.get_children() if isinstance(poly, matplotlib.patches.Polygon)]
    for poly in axpolygons:
        poly.set_xy(poly.get_xy()[:-1])


fig, ax = plt.subplots()

rainbow = cm.get_cmap('rainbow_r', 10)
rainbow_list = rainbow(range(10))

APC_ratio_list = []
for barcode in all_barcode_list:
    if barcode['APC-'] != 0:
        APC_ratio_list.append(barcode['APC+'] / barcode['APC-'])

APC_ratio_list = np.array(APC_ratio_list)
APC_ratio_list = APC_ratio_list[APC_ratio_list > 0]
APC_ratio_list = np.log10(APC_ratio_list)

plt.hist(APC_ratio_list, bins=1000, density=True, histtype='step', cumulative=True, color='black', label='All lineages')

all_var = np.var(APC_ratio_list, ddof=1)

quantile_var = []
len_list = []
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

for index, q_var in enumerate(quantile_var):
    alpha = 0.05
    F = q_var / all_var
    df1 = len_list[index] - 1
    df2 = len(all_barcode_list) - 1
    p_value = scipy.stats.f.sf(F, df1, df2)
    print(index, F, p_value)
    if p_value < alpha / 2 or p_value > 1 - (alpha / 2):
        print('Significantly different: {}'.format(index))

fix_hist_step_vertical_line_at_end(ax)
plt.legend(loc='upper left')
plt.title('Quantile: Total Transition Amount')
plt.xlabel(r'$\log_{10}\frac{APC+}{APC-}$')
plt.ylabel('Cumulative Proportion')
plt.savefig('RAinduction_TransitionAmount.tiff', format='tiff', dpi=720)

all_APC_ratio_list = []

for quantile, i in enumerate(np.arange(0, 1, 0.1)):
    if quantile == 0 or quantile == 9:
        quantile_population = all_barcode_list[round(i * barcode_number):round((i + 0.1) * barcode_number)]
        APC_ratio_list = []
        for barcode in quantile_population:
            if barcode['APC-'] != 0:
                APC_ratio_list.append(barcode['APC+'] / barcode['APC-'])

        APC_ratio_list = np.array(APC_ratio_list)
        APC_ratio_list = APC_ratio_list[APC_ratio_list > 0]
        APC_ratio_list = np.log10(APC_ratio_list)

        all_APC_ratio_list.append(APC_ratio_list)

        #         plt.hist(APC_ratio_list, bins=100, density=True, color=rainbow_list[quantile], label='Decile {}'.format(quantile+1), alpha=0.25)

        len_list.append(len(APC_ratio_list))
        quantile_var.append(np.var(APC_ratio_list, ddof=1))

APC_ratio_list = []
for barcode in all_barcode_list:
    if barcode['APC-'] != 0:
        APC_ratio_list.append(barcode['APC+'] / barcode['APC-'])

APC_ratio_list = np.array(APC_ratio_list)
APC_ratio_list = APC_ratio_list[APC_ratio_list > 0]
APC_ratio_list = np.log10(APC_ratio_list)

all_APC_ratio_list.append(APC_ratio_list)

plt.hist(all_APC_ratio_list, bins=25, density=True, color=[rainbow_list[0], rainbow_list[9], 'black'],
         label=['Decile 1', 'Decile 10', 'All Lineages'])
plt.legend(loc='upper left')

plt.savefig('RAinduction_TransitionAmount_Histogram.tiff', format='tiff', dpi=720)

from matplotlib import cm
from matplotlib import rc
import scipy
import scipy.stats

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

all_barcode_list.sort(reverse=True, key=lambda barcode: barcode['total_transition_amount'])


def fix_hist_step_vertical_line_at_end(ax):
    axpolygons = [poly for poly in ax.get_children() if isinstance(poly, matplotlib.patches.Polygon)]
    for poly in axpolygons:
        poly.set_xy(poly.get_xy()[:-1])


fig, ax = plt.subplots()

rainbow = cm.get_cmap('rainbow_r', 10)
rainbow_list = rainbow(range(10))

APC_ratio_list = []
for barcode in all_barcode_list:
    if barcode['APC-'] != 0:
        APC_ratio_list.append(barcode['APC+'] / barcode['APC-'])

APC_ratio_list = np.array(APC_ratio_list)
APC_ratio_list = APC_ratio_list[APC_ratio_list > 0]
APC_ratio_list = np.log10(APC_ratio_list)

plt.hist(APC_ratio_list, bins=1000, density=True, histtype='step', cumulative=True, color='black', label='All lineages')

all_var = np.var(APC_ratio_list, ddof=1)

for quantile, i in enumerate(np.arange(0, 1, 0.1)):
    if quantile == 0 or quantile == 9:
        quantile_population = all_barcode_list[round(i * barcode_number):round((i + 0.1) * barcode_number)]
        APC_ratio_list = []
        for barcode in quantile_population:
            if barcode['APC-'] != 0:
                APC_ratio_list.append(barcode['APC+'] / barcode['APC-'])

        APC_ratio_list = np.array(APC_ratio_list)
        APC_ratio_list = APC_ratio_list[APC_ratio_list > 0]
        APC_ratio_list = np.log10(APC_ratio_list)

        plt.hist(APC_ratio_list, bins=10000, density=True, histtype='step', cumulative=True,
                 color=rainbow_list[quantile], label='Decile {}'.format(quantile + 1))

fix_hist_step_vertical_line_at_end(ax)
plt.savefig('RAinduction_TransitionAmount_CDF_inset.tiff', format='tiff', dpi=720)