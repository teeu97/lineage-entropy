import pickle
import math
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.lines import Line2D
from matplotlib import cm
from scipy import stats


def euclidean_distance(coor_1, coor_2):
    return math.sqrt(sum((i - j) ** 2 for i, j in zip(coor_1, coor_2)))


def vector_size(x_displacement, y_displacement):
    return math.sqrt(x_displacement ** 2 + y_displacement ** 2)

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)
matplotlib.rcParams['font.sans-serif'] = "Helvetica"
matplotlib.rcParams['font.family'] = "sans-serif"

conditions = ['WT1', 'WT2', 'J1', 'J2']

total_cell_number = 10 ** 8

state_1_ratio = 0.90
state_2_ratio = 0.05
state_3_ratio = 0.05

state_1_number = state_1_ratio * total_cell_number
state_2_number = state_2_ratio * total_cell_number
state_3_number = state_3_ratio * total_cell_number

normalizing_factor = [total_cell_number, state_1_number, state_2_number, state_3_number] * 20

table = pickle.load(open('20200628_finished_table.pickle', 'rb'))
table_2 = pickle.load(open('20200713_finished_table.pickle', 'rb'))

combined_table = pd.concat([table, table_2], axis=1, sort=False)
combined_table.fillna(0, inplace=True)

sum_table = combined_table.sum(axis=0)
normalized_table = combined_table.div(sum_table)
true_number_table = (normalized_table * normalizing_factor).round()

states = ['s1', 's2', 's3']
states_coords = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

top_right_coord = (1, 1)
top_left_coord = (0, 1)
bottom_left_coord = (0, 0)

all_barcode_list_dict = {}
all_barcode_set_dict = {}

triangle_vertices = np.array([top_right_coord, top_left_coord, bottom_left_coord])

timepoints = ['d0', 'd6', 'd12', 'd18', 'd24']

for condition in conditions:
    all_size_set = set()
    all_barcode_set = set()
    all_barcode_list = []
    all_transition_size_list = []

    for barcode, row in true_number_table.iterrows():
        barcode_dict = {}

        barcode_dict['d0_WT1_all'], barcode_dict['d0_WT1_s1'], barcode_dict['d0_WT1_s2'], barcode_dict[
            'd0_WT1_s3'] = row[0:4]
        barcode_dict['d0_WT2_all'], barcode_dict['d0_WT2_s1'], barcode_dict['d0_WT2_s2'], barcode_dict[
            'd0_WT2_s3'] = row[4:8]
        barcode_dict['d0_J1_all'], barcode_dict['d0_J1_s1'], barcode_dict['d0_J1_s2'], barcode_dict['d0_J1_s3'] = row[
                                                                                                                  8:12]
        barcode_dict['d0_J2_all'], barcode_dict['d0_J2_s1'], barcode_dict['d0_J2_s2'], barcode_dict['d0_J2_s3'] = row[
                                                                                                                  12:16]
        barcode_dict['d6_WT1_all'], barcode_dict['d6_WT1_s1'], barcode_dict['d6_WT1_s2'], barcode_dict[
            'd6_WT1_s3'] = row[16:20]
        barcode_dict['d6_WT2_all'], barcode_dict['d6_WT2_s1'], barcode_dict['d6_WT2_s2'], barcode_dict[
            'd6_WT2_s3'] = row[20:24]
        barcode_dict['d6_J1_all'], barcode_dict['d6_J1_s1'], barcode_dict['d6_J1_s2'], barcode_dict['d6_J1_s3'] = row[
                                                                                                                  24:28]
        barcode_dict['d6_J2_all'], barcode_dict['d6_J2_s1'], barcode_dict['d6_J2_s2'], barcode_dict['d6_J2_s3'] = row[
                                                                                                                  28:32]
        barcode_dict['d24_WT1_all'], barcode_dict['d24_WT1_s1'], barcode_dict['d24_WT1_s2'], barcode_dict[
            'd24_WT1_s3'] = row[32:36]
        barcode_dict['d24_J1_all'], barcode_dict['d24_J1_s1'], barcode_dict['d24_J1_s2'], barcode_dict[
            'd24_J1_s3'] = row[36:40]
        barcode_dict['d12_WT1_all'], barcode_dict['d12_WT1_s1'], barcode_dict['d12_WT1_s2'], barcode_dict[
            'd12_WT1_s3'] = row[40:44]
        barcode_dict['d12_WT2_all'], barcode_dict['d12_WT2_s1'], barcode_dict['d12_WT2_s2'], barcode_dict[
            'd12_WT2_s3'] = row[44:48]
        barcode_dict['d12_J1_all'], barcode_dict['d12_J1_s1'], barcode_dict['d12_J1_s2'], barcode_dict[
            'd12_J1_s3'] = row[48:52]
        barcode_dict['d12_J2_all'], barcode_dict['d12_J2_s1'], barcode_dict['d12_J2_s2'], barcode_dict[
            'd12_J2_s3'] = row[52:56]
        barcode_dict['d18_WT1_all'], barcode_dict['d18_WT1_s1'], barcode_dict['d18_WT1_s2'], barcode_dict[
            'd18_WT1_s3'] = row[56:60]
        barcode_dict['d18_WT2_all'], barcode_dict['d18_WT2_s1'], barcode_dict['d18_WT2_s2'], barcode_dict[
            'd18_WT2_s3'] = row[60:64]
        barcode_dict['d18_J1_all'], barcode_dict['d18_J1_s1'], barcode_dict['d18_J1_s2'], barcode_dict[
            'd18_J1_s3'] = row[64:68]
        barcode_dict['d18_J2_all'], barcode_dict['d18_J2_s1'], barcode_dict['d18_J2_s2'], barcode_dict[
            'd18_J2_s3'] = row[68:72]
        barcode_dict['d24_WT2_all'], barcode_dict['d24_WT2_s1'], barcode_dict['d24_WT2_s2'], barcode_dict[
            'd24_WT2_s3'] = row[72:76]
        barcode_dict['d24_J2_all'], barcode_dict['d24_J2_s1'], barcode_dict['d24_J2_s2'], barcode_dict[
            'd24_J2_s3'] = row[76:80]

        barcode_summary = {'id': barcode, 'ternary_coord': [], 'cartesian_coord': [], 'vector': [], 'size': [],
                           'assigned_state': [], 'transition_amount': [], 'total_transition_amount': 0}

        barcode_size = [sum([barcode_dict[timepoint + '_{}_'.format(condition) + state] for state in states]) for
                        timepoint in timepoints]
        for timepoint in timepoints:
            timepoint_all_present = all(barcode_size)
            timepoint_total = sum([barcode_dict[timepoint + '_{}_'.format(condition) + state] for state in states])
            if timepoint_total > 10:
                ternary_coord = []
                dist = []
                for state in states:
                    ternary_coord.append(barcode_dict[timepoint + '_{}_'.format(condition) + state] / timepoint_total)
                barcode_summary['ternary_coord'].append(ternary_coord)

                cartesian_coord = np.dot(np.array(ternary_coord), triangle_vertices)
                barcode_summary['cartesian_coord'].append(list(cartesian_coord))

                for state_coord in triangle_vertices:
                    dist.append(euclidean_distance(cartesian_coord, state_coord))
                barcode_summary['assigned_state'].append(dist.index(min(dist)))

                barcode_summary['size'].append(timepoint_total)

        if len(barcode_summary['cartesian_coord']) == len(barcode_size):
            for i in range(len(barcode_size) - 1):
                vector = (barcode_summary['cartesian_coord'][i + 1][0] - barcode_summary['cartesian_coord'][i][0],
                          barcode_summary['cartesian_coord'][i + 1][1] - barcode_summary['cartesian_coord'][i][1])
                barcode_summary['vector'].append(vector)
                barcode_summary['transition_amount'].append(vector_size(vector[0], vector[1]))
                barcode_summary['total_transition_amount'] += vector_size(vector[0], vector[1])
            all_transition_size_list.append(barcode_summary['total_transition_amount'])
            for size in barcode_summary['size']:
                all_size_set.add(round(size, 3))
            all_barcode_set.add(barcode)
            all_barcode_list.append(barcode_summary)
    all_barcode_list.sort(reverse=True, key=lambda barcode: barcode['total_transition_amount'])
    all_barcode_list_dict[condition] = all_barcode_list
    all_barcode_set_dict[condition] = all_barcode_set

def correlation_WT():
    import scipy

    shared_barcode = all_barcode_set_dict['WT1'] & all_barcode_set_dict['WT2']
    motility_dict = {}
    for barcode in all_barcode_list_dict['WT1']:
        if barcode['id'] in shared_barcode:
            motility_dict[barcode['id']] = [barcode['transition_amount']]
    for barcode in all_barcode_list_dict['WT2']:
        if barcode['id'] in shared_barcode:
            motility_dict[barcode['id']] += [barcode['transition_amount']]

    color_scalarMap = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=100*math.sqrt(2)), cmap='rainbow')
    rainbow = cm.get_cmap('rainbow', len(shared_barcode))
    rainbow_list = rainbow(range(len(shared_barcode)))[::-1]

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=[8 * 1.5, 6 * 1.5])
    for a in range(2):
        for b in range(2):
            transition = 2 * a + b
            ax = axs[a, b]

            x = np.array([motility_dict[barcode][0][transition] for barcode in motility_dict])
            x /= math.sqrt(2)
            x *= 100
            y = np.array([motility_dict[barcode][1][transition] for barcode in motility_dict])
            y /= math.sqrt(2)
            y *= 100

            c_ = [color_scalarMap.to_rgba(round(math.sqrt(i**2 + j**2), 3)) for i, j in zip(x, y)]
            slope, intercept, r, p, stderr = scipy.stats.linregress(x, y)
            rho, p_ = scipy.stats.spearmanr(x, y)

            ax.plot(np.arange(0, max(x), 0.0001), intercept + slope * np.arange(0, max(x), 0.0001), color='red')
            ax.scatter(x, y, marker='.', color=c_)
            ax.set_ylim(0, max(y))
            ax.set_xlim(0, max(x))
            ax.set_title('Transition: Day {} to {}'.format(transition * 6, (transition + 1) * 6))
            if transition == 3:
                ax.set_xlabel('Percent Transition in WT1 sample')
                ax.set_ylabel('Percent Transition in WT2 sample')
    plt.savefig('RepeatedExperiment_Correlation_WT.svg', format='svg', dpi=720)

correlation_WT()