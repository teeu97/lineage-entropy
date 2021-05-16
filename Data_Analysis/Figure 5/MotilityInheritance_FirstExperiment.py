"""
MotilityInheritance_FirstExperiment.py analyzes the inheritance of motility in the first experiment
"""

import pickle
import math
import matplotlib
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


# normalize barcodes
total_cell_number = 10 ** 8

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

top_right_coord = (1, 1)
top_left_coord = (0, 1)
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
                       'total_transition_amount': 0, 'transition_amount': []}

    barcode_size = [sum(row[1:4]), sum(row[5:8]), sum(row[21:24]), sum(row[33:36]), sum(row[37:40])]

    for timepoint in timepoints:
        timepoint_all_present = all(barcode_size)
        timepoint_total = sum([barcode_dict[timepoint + '_' + state] for state in states])
        if timepoint_total > 10:
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

    if len(barcode_summary['cartesian_coord']) == 5:
        for i in range(4):
            vector = (barcode_summary['cartesian_coord'][i + 1][0] - barcode_summary['cartesian_coord'][i][0],
                      barcode_summary['cartesian_coord'][i + 1][1] - barcode_summary['cartesian_coord'][i][1])
            barcode_summary['vector'].append(vector)
            barcode_summary['transition_amount'].append(vector_size(vector[0], vector[1]))
            barcode_summary['total_transition_amount'] += vector_size(vector[0], vector[1])
        all_transition_size_list.append(barcode_summary['total_transition_amount'])
        for size in barcode_summary['size']:
            all_size_set.add(round(size, 3))
        all_barcode_list.append(barcode_summary)
all_barcode_list.sort(reverse=True, key=lambda barcode: barcode['total_transition_amount'])

barcode_number = len(all_barcode_list)

def motility_super_switcher_histogram_all(all_barcode_list):
    rainbow = cm.get_cmap('rainbow_r', 10)
    rainbow_list = rainbow(range(10))

    # sort barcodes based on its total transition amount
    all_barcode_list.sort(reverse=True, key=lambda barcode: barcode['total_transition_amount'])
    for transition in range(4):
        fig, ax = plt.subplots()
        if transition == 0:
            ax.set_ylabel('Number of Lineages')
            ax.set_xlabel('Percent Transition')
        ax.set_title('Day {} to {}'.format(transition * 6, (transition + 1) * 6))
        data = [[] for i in range(10)]
        for index, barcode in enumerate(all_barcode_list):
            motility = barcode['transition_amount'][transition]
            data[int(index / len(all_barcode_list) * 10)].append(motility * 100 / math.sqrt(2))

        # plot stacked bar plot
        ax.hist(data, stacked=True, log=True, histtype='barstacked', bins=np.arange(0, 105, 5), color=rainbow_list)

        plt.savefig("MotilityInheritance_FirstExperiment_Log_T{}.svg".format(transition), format='svg', bbox_inches='tight', dpi=720)

def fix_hist_step_vertical_line_at_end(ax):
    # this function removes the vertical line at the end of CDF
    axpolygons = [poly for poly in ax.get_children() if isinstance(poly, matplotlib.patches.Polygon)]
    for poly in axpolygons:
        poly.set_xy(poly.get_xy()[:-1])

def motility_super_switcher_histogram_all(all_barcode_list):
    rainbow = cm.get_cmap('rainbow_r', 10)
    rainbow_list = rainbow(range(10))

    all_barcode_list.sort(reverse=True, key=lambda barcode: barcode['total_transition_amount'])
    for transition in range(4):
        fig, ax = plt.subplots()
        plt.axis('off')
        data = [[] for i in range(10)]
        for index, barcode in enumerate(all_barcode_list):
            motility = barcode['transition_amount'][transition]
            data[int(index / len(all_barcode_list) * 10)].append(motility * 100 / math.sqrt(2))

        # plot CDF
        ax.hist(data, density=True, histtype='step', cumulative=True, bins=np.arange(0, 105, 1), color=rainbow_list)
        fix_hist_step_vertical_line_at_end(ax)
        plt.savefig("MotilityInheritance_FirstExperiment_CDF_T{}.svg".format(transition), format='svg', bbox_inches='tight', dpi=720)

motility_super_switcher_histogram_all(all_barcode_list)
motility_super_switcher_histogram_all(all_barcode_list)