import pickle
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from numpy.linalg import pinv
from matplotlib import cm
from tqdm.auto import tqdm


def euclidean_distance(coor_1, coor_2):
    return math.sqrt(sum((i - j) ** 2 for i, j in zip(coor_1, coor_2)))


def vector_size(x_displacement, y_displacement):
    return math.sqrt(x_displacement ** 2 + y_displacement ** 2)


matplotlib.rcParams['figure.dpi'] = 1200
matplotlib.rcParams['figure.figsize'] = (6, 6)
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
                       'vector_size': [], 'monte_ternary_coord': [], 'monte_cartesian_coord': [], 'monte_vector': [],
                       'timepoint_size': [], 'total_transition_amount': 0, 'total_size': 0}

    barcode_size = [sum(row[1:4]), sum(row[5:8]), sum(row[21:24]), sum(row[33:36]), sum(row[37:40])]

    for timepoint in timepoints:
        timepoint_all_present = all(barcode_size)
        timepoint_total = sum([barcode_dict[timepoint + '_' + state] for state in states])
        if timepoint_total > 10:
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


def least_square_estimation_all_separate_timepoint(all_barcode_list):
    probability_timepoint_list = []
    for timepoint in range(4):
        T_0 = np.zeros((len(all_barcode_list), 3))
        T_1 = np.zeros((len(all_barcode_list), 3))
        for index, barcode in enumerate(all_barcode_list):
            ternary_coord = barcode['ternary_coord']
            T_0[index] = np.array(ternary_coord[timepoint])
            T_1[index] = np.array(ternary_coord[timepoint + 1])
            T_0_t = np.transpose(T_0)
        probability_timepoint_list.append(np.matmul(pinv(np.matmul(T_0_t, T_0)), np.matmul(T_0_t, T_1)))
    return probability_timepoint_list


transitional_prob_list = least_square_estimation_all_separate_timepoint(all_barcode_list)

possible_states = [0, 1, 2]
for index, barcode in enumerate(all_barcode_list):
    ternary_coord = barcode['ternary_coord']
    cartesian_coord = barcode['cartesian_coord']
    timepoint_size = barcode['timepoint_size']
    monte_ternary_coord = barcode['monte_ternary_coord']
    assigned_state = barcode['assigned_state']
    monte_cartesian_coord = barcode['monte_cartesian_coord']
    current_timepoint_size = timepoint_size[0]
    monte_cartesian_coord.append(cartesian_coord[0])
    monte_ternary_coord.append(ternary_coord[0])
    for timepoint in range(4):
        current_transitional_prob = transitional_prob_list[timepoint]
        new_distribution = [0, 0, 0]
        for j, state in enumerate(current_timepoint_size):
            for cell in range(int(state)):
                new_distribution[random.choices(possible_states, current_transitional_prob[j])[0]] += 1
        current_distribution = np.array(new_distribution) / sum(new_distribution)
        monte_ternary_coord.append(current_distribution)
        monte_cartesian_coord.append(list(np.dot(current_distribution, triangle_vertices)))
        current_timepoint_size = new_distribution
    for i in range(4):
        vector = (monte_cartesian_coord[i + 1][0] - monte_cartesian_coord[i][0],
                  monte_cartesian_coord[i + 1][1] - monte_cartesian_coord[i][1])
        barcode['monte_vector'].append(vector)


def vector_field_size_weight_shifted_size(all_barcode_list):
    default_size = 3
    color_scalarMap = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.LogNorm(vmin=1, vmax=max(all_size_set)),
                                                   cmap='YlOrRd')
    all_barcode_list.sort(key=lambda barcode: barcode['total_size'])
    for timepoint in tqdm(range(4)):
        fig = plt.figure()
        plt.plot([0, 0], [0, 10], color='black', alpha=0.25)
        plt.plot([0, 10], [10, 10], color='black', alpha=0.25)
        plt.plot([10, 0], [10, 0], color='black', alpha=0.25)
        for barcode in all_barcode_list:
            if random.random() <= 0.2:
                cartesian_coord = barcode['monte_cartesian_coord']
                vector = barcode['monte_vector']
                size = barcode['size'][timepoint + 1]
                x_coord = cartesian_coord[timepoint][0]
                y_coord = cartesian_coord[timepoint][1]
                dx = vector[timepoint][0]
                dy = vector[timepoint][1]

                arrow_color = color_scalarMap.to_rgba(size)
                vector_magnitude = vector_size(4 * dx, 4 * dy)

                plt.arrow(x_coord, y_coord, dx / 15 + dx / vector_magnitude,
                          dy / 15 + dy / vector_magnitude, shape='full',
                          head_width=0.1, color=arrow_color, length_includes_head=True)
        plt.axis('off')
        plt.savefig('VectorField_MonteCarloSimulation_Right_T{}.svg'.format(timepoint), bbox_inches='tight', format='svg',
                    dpi=720)

vector_field_size_weight_shifted_size(all_barcode_list)

