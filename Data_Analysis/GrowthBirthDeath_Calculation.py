"""
This file calculates the estimated growth birth death rate
"""

import pickle
import math
import numpy as np
from numpy.linalg import pinv

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
                       'timepoint_size': [], 'total_transition_amount': 0}

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
        for i in range(4):
            vector = (barcode_summary['cartesian_coord'][i + 1][0] - barcode_summary['cartesian_coord'][i][0],
                      barcode_summary['cartesian_coord'][i + 1][1] - barcode_summary['cartesian_coord'][i][1])
            barcode_summary['vector'].append(vector)
            barcode_summary['total_transition_amount'] += vector_size(vector[0], vector[1])
        all_transition_size_list.append(barcode_summary['total_transition_amount'])
        for size in barcode_summary['size']:
            all_size_set.add(round(size, 3))
        all_barcode_list.append(barcode_summary)
all_barcode_list.sort(reverse=True, key=lambda barcode: barcode['total_transition_amount'])


def least_square_estimation_all_separate_timepoint(all_barcode_list):
    probability_timepoint_list = []
    for timepoint in range(4):
        P = np.zeros((3, 3))
        T_0 = np.zeros((len(all_barcode_list), 3))
        T_1 = np.zeros((len(all_barcode_list), 3))
        for index, barcode in enumerate(all_barcode_list):
            ternary_coord = barcode['ternary_coord']
            size = barcode['size']
            T_0[index] = np.array(ternary_coord[timepoint])
            T_1[index] = np.array(ternary_coord[timepoint + 1])
            T_0_t = np.transpose(T_0)
        probability_timepoint_list.append(np.matmul(pinv(np.matmul(T_0_t, T_0)), np.matmul(T_0_t, T_1)))
    return probability_timepoint_list

def least_square_estimation_all_timepoint(all_barcode_list):
    P = np.zeros((3, 3))
    T_0 = np.zeros((len(all_barcode_list) * 4, 3))
    T_1 = np.zeros((len(all_barcode_list) * 4, 3))
    length = 0
    for timepoint in range(4):
        for barcode in all_barcode_list:
            ternary_coord = barcode['ternary_coord']
            T_0[length] = np.array(ternary_coord[timepoint])
            T_1[length] = np.array(ternary_coord[timepoint + 1])
            length += 1
    T_0_t = np.transpose(T_0)
    P = np.matmul(pinv(np.matmul(T_0_t, T_0)), np.matmul(T_0_t, T_1))
    return P

def least_square_estimation_gbd():
    gbd_timepoint_list = []
    for timepoint in range(4):
        N_0 = np.zeros((len(all_barcode_list), 3))
        N_1 = np.zeros((len(all_barcode_list), 3))
        for index, barcode in enumerate(all_barcode_list):
            timepoint_size = barcode['timepoint_size']
            N_0[index] = np.array(timepoint_size[timepoint])
            N_1[index] = np.array(timepoint_size[timepoint + 1])
        M = transitional_prob_list[timepoint]

        c_N_0 = np.matmul(N_0, M)
        c_N_0_t = np.transpose(c_N_0)

        gbd = np.matmul(pinv(np.matmul(c_N_0_t, c_N_0)), np.matmul(c_N_0_t, N_1))

        gbd_timepoint_list.append(gbd)

    return gbd_timepoint_list


def gbd_rate():
    state_list = [
        [100, 0, 0],
        [0, 100, 0],
        [0, 0, 100],
    ]

    all_rate_list = []
    for timepoint in range(4):
        rate_list = []
        for X in state_list:
            M = transitional_prob_list[timepoint]
            G = gbd_list_new[timepoint]
            X_hat = np.matmul(X, M)
            Y_hat = np.matmul(X_hat, G)
            rate = Y_hat / X_hat
            rate_list.append(rate)
        all_rate_list.append(rate_list)

    return all_rate_list


def least_square_estimation_gbd_all_new():
    gbd_timepoint_list = []

    N_0 = np.zeros((len(all_barcode_list) * 4, 3))
    N_1 = np.zeros((len(all_barcode_list) * 4, 3))

    index = 0
    for barcode in all_barcode_list:
        for timepoint in range(4):
            timepoint_size = barcode['timepoint_size']
            N_0[index] = np.array(timepoint_size[timepoint])
            N_1[index] = np.array(timepoint_size[timepoint + 1])
            index += 1
    M = transitional_prob

    c_N_0 = np.matmul(N_0, M)
    c_N_0_t = np.transpose(c_N_0)

    gbd = np.matmul(pinv(np.matmul(c_N_0_t, c_N_0)), np.matmul(c_N_0_t, N_1))
    gbd_timepoint_list.append(gbd)

    return gbd_timepoint_list

def gbd_rate_all_new():
    state_list = [
        [1000, 0, 0],
        [0, 100, 0],
        [0, 0, 100],
    ]

    all_rate_list = []
    for X in state_list:
        M = transitional_prob
        G = gbd_all_new
        X_hat = np.matmul(X, M)
        Y_hat = np.matmul(X_hat, G)
        rate = Y_hat / X_hat
        all_rate_list.append(np.round(rate, 2))

    return all_rate_list


transitional_prob = least_square_estimation_all_timepoint(all_barcode_list)
transitional_prob_list = least_square_estimation_all_separate_timepoint(all_barcode_list)
gbd_list_new = least_square_estimation_gbd()
all_rate_list = gbd_rate()
gbd_all_new = least_square_estimation_gbd_all_new()
all_rate_list = gbd_rate_all_new()