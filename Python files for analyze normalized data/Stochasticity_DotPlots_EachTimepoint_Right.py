import pickle
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from tqdm.auto import tqdm
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy.linalg import pinv

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
                       'vector_size': [], 'cell_number': [], 'observed_bulk_size': []}

    barcode_size = [sum(row[1:4]), sum(row[5:8]), sum(row[21:24]), sum(row[33:36]), sum(row[37:40])]

    for timepoint in timepoints:
        timepoint_all_present = all(barcode_size)
        timepoint_total = sum([barcode_dict[timepoint + '_' + state] for state in states])
        if timepoint_total:
            ternary_coord = []
            cell_number = []
            dist = []
            for state in states:
                ternary_coord.append(barcode_dict[timepoint + '_' + state] / timepoint_total)
                cell_number.append(barcode_dict[timepoint + '_' + state])

            barcode_summary['cell_number'].append(cell_number)
            barcode_summary['ternary_coord'].append(ternary_coord)

            cartesian_coord = np.dot(np.array(ternary_coord), triangle_vertices)
            barcode_summary['cartesian_coord'].append(list(cartesian_coord))

            for state_coord in triangle_vertices:
                dist.append(euclidean_distance(cartesian_coord, state_coord))
            barcode_summary['assigned_state'].append(dist.index(min(dist)))

            barcode_summary['size'].append(timepoint_total)
            barcode_summary['observed_bulk_size'].append(barcode_dict[timepoint + '_all'])

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
            T_1[index] = np.array(ternary_coord[timepoint+1])
            T_0_t = np.transpose(T_0)
        probability_timepoint_list.append(np.matmul(pinv(np.matmul(T_0_t, T_0)), np.matmul(T_0_t, T_1)))
    return probability_timepoint_list

def steady_state_simulation():
    for timepoint in range(4):
        T_0 = np.zeros((len(all_barcode_list), 3))
        for index, barcode in enumerate(all_barcode_list):
            ternary_coord = barcode['ternary_coord']
            T_0[index] = np.array(ternary_coord[timepoint])
        T_0 = pd.DataFrame(np.matmul(T_0, transitional_prob_list[timepoint]))
    return T_0


def entropy(ternary_coord):
    sum_ = 0
    for index, p in enumerate(ternary_coord):
        if p == 0:
            sum_ += 0
        else:
            if p <= steady_state_population[index]:
                p_ = p/steady_state_population[index] * 1/3
                sum_ += p_ * np.log10(p_)/np.log10(3)
            else:
                p_ = 1 + (2/3 * (p-1)/(1-steady_state_population[index]))
                sum_ += p_ * np.log10(p_)/np.log10(3)
    return -sum_

transitional_prob_list = least_square_estimation_all_separate_timepoint(all_barcode_list)
steady_state_population = steady_state_simulation().mean(axis = 0)

for barcode in all_barcode_list:
    barcode['entropy'] = []
    for timepoint in range(5):
        barcode['entropy'].append(entropy(barcode['ternary_coord'][timepoint]))

def scatter_plot_right_each(all_barcode_list):
    color_scalarMap = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=1),
                                                   cmap='rainbow')
    for i in range(5):
        fig, ax = plt.subplots()
        x_all = []
        y_all = []
        c_all = []
        all_barcode_list.sort(key=lambda barcode: barcode['entropy'][i])
        for barcode in tqdm(all_barcode_list):
            cartesian_coord = barcode['cartesian_coord']
            ternary_coord = barcode['ternary_coord']
            x_all.append(cartesian_coord[i][0])
            y_all.append(cartesian_coord[i][1])
            c_all.append(color_scalarMap.to_rgba(round(entropy(ternary_coord[i]), 3)))

        ax.axis('off')
        ax.scatter(x_all, y_all, c=c_all, marker='.')

        plt.savefig("Stochasticity_DotPlots_EachTimepoint_Right_{}.svg".format(i), bbox_inches='tight', format='svg', dpi=720)

scatter_plot_right_each(all_barcode_list)