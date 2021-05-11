import pickle
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from matplotlib import cm
from tqdm.auto import tqdm


def euclidean_distance(coor_1, coor_2):
    return math.sqrt(sum((i - j) ** 2 for i, j in zip(coor_1, coor_2)))


def vector_size(x_displacement, y_displacement):
    return math.sqrt(x_displacement ** 2 + y_displacement ** 2)


matplotlib.rcParams['font.sans-serif'] = "Helvetica"
matplotlib.rcParams['font.family'] = "sans-serif"

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
all_vector_size_set = set()
all_barcode_list = []

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
                       'vector_size': [], 'monte_ternary_coord': [], 'monte_cartesian_coord': [], 'monte_vector': [],
                       'timepoint_size': [], 'total_transition_amount': 0, 'total_size': 0}

    barcode_size = [sum(row[1:4]), sum(row[5:8]), sum(row[21:24]), sum(row[33:36]), sum(row[37:40])]

    for timepoint in timepoints:
        timepoint_all_present = all(barcode_size)
        timepoint_total = sum([barcode_dict[timepoint + '_' + state] for state in states])
        if timepoint_total:
            timepoint_size = []
            ternary_coord = []
            cell_number = []
            dist = []
            for state in states:
                timepoint_size.append(barcode_dict[timepoint + '_' + state])
                ternary_coord.append(barcode_dict[timepoint + '_' + state] / timepoint_total)
            barcode_summary['timepoint_size'].append(timepoint_size)
            barcode_summary['ternary_coord'].append(ternary_coord)

            cartesian_coord = np.dot(np.array(ternary_coord), triangle_vertices)
            barcode_summary['cartesian_coord'].append(list(cartesian_coord))

            for state_coord in triangle_vertices:
                dist.append(euclidean_distance(cartesian_coord, state_coord))
            barcode_summary['assigned_state'].append(dist.index(min(dist)))

            barcode_summary['size'].append(timepoint_total)

    if len(barcode_summary['cartesian_coord']) == 5:
        for i in range(4):
            barcode_summary['vector'].append((barcode_summary['cartesian_coord'][i + 1][0] -
                                              barcode_summary['cartesian_coord'][i][0],
                                              barcode_summary['cartesian_coord'][i + 1][1] -
                                              barcode_summary['cartesian_coord'][i][1]))
            barcode_summary['vector_size'].append(
                vector_size(barcode_summary['vector'][i][0], barcode_summary['vector'][i][1]))
        for size in barcode_summary['size']:
            all_size_set.add(round(size, 3))
            barcode_summary['total_size'] += size
        for size_ in barcode_summary['vector_size']:
            all_vector_size_set.add(round(size_, 3))
        all_barcode_list.append(barcode_summary)

transitions = ['t{}'.format(i) for i in range(4)]

def creating_dict(i, states):
    if i == 5:
        return {'barcode': []}
    else:
        updated_dict = {'t{}'.format(i):{state: creating_dict(i+1, states) for state in states}}
        updated_dict['t{}'.format(i)].update({'barcode': []})
        return updated_dict

test_dict = creating_dict(0, states)

def fill_dict(test_dict, i, barcode_list):
    if i == 5:
        for barcode in barcode_list:
            test_dict['barcode'].append(barcode)
        return test_dict
    else:
        barcode_allocation_list = [[] for i in range(3)]
        for barcode in barcode_list:
            barcode_allocation_list[barcode['assigned_state'][i]].append(barcode)
            test_dict['t{}'.format(i)]['barcode'].append(barcode)
        for index, state in enumerate(states):
            test_dict['t{}'.format(i)][state] = fill_dict(test_dict['t{}'.format(i)][state], i+1, barcode_allocation_list[index])
        return test_dict

test_dict = fill_dict(test_dict, 0, all_barcode_list)

def following_bundle(test_dict, state_list=[], i=1):
    color_scalarMap = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.LogNorm(vmin = 1, vmax = max(all_size_set)), cmap='YlOrRd')
    if i == 4:
        fig, ax = plt.subplots()
        barcode_list = test_dict['t{}'.format(i)]['barcode']
        for barcode in barcode_list:
            vector = barcode['vector']
            size = barcode['size']
            barcode_color = color_scalarMap.to_rgba(round(size[i], 3))
            ax.axis([-1.2, 1.2, -1.2, 1.2])
            ax.axis('off')
            ax.arrow(0, 0, vector[i-1][0], vector[i-1][1], shape='full', head_width=0.01, color=barcode_color)
        plt.savefig('FollowingBundle_{}_{}.svg'.format(str(state_list), i), format='svg', dpi=720, bbox_inches='tight')
        plt.close()
    else:
        fig, ax = plt.subplots()
        barcode_list = test_dict['t{}'.format(i)]['barcode']
        for barcode in barcode_list:
            vector = barcode['vector']
            size = barcode['size']
            barcode_color = color_scalarMap.to_rgba(round(size[i], 3))
            ax.axis([-1.2, 1.2, -1.2, 1.2])
            ax.axis('off')
            ax.arrow(0, 0, vector[i-1][0], vector[i-1][1], shape='full', head_width=0.01, color=barcode_color)
        plt.savefig('FollowingBundle_{}_{}.svg'.format(str(state_list), i), format='svg', dpi=720, bbox_inches='tight')
        plt.close()
        for state in states:
            following_bundle(test_dict['t{}'.format(i)][state], state_list + [state], i+1)

for state in states:
    following_bundle(test_dict['t0'][state], [state])

