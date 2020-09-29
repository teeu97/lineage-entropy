import pickle
import math
import random
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from numpy.linalg import pinv
from matplotlib import cm
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)
matplotlib.rcParams['font.sans-serif'] = "Helvetica"
matplotlib.rcParams['font.family'] = "sans-serif"

def euclidean_distance(coor_1, coor_2):
    return math.sqrt(sum((i - j) ** 2 for i, j in zip(coor_1, coor_2)))


def vector_size(x_displacement, y_displacement):
    return math.sqrt(x_displacement ** 2 + y_displacement ** 2)


top_right_coord = (10, 10)
top_left_coord = (0, 10)
bottom_left_coord = (0, 0)

triangle_vertices = np.array([top_right_coord, top_left_coord, bottom_left_coord])

empirical_stochasticity = {i: [] for i in range(5)}

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
                       'timepoint_size': [], 'total_transition_amount': 0, 'proportion_diff': []}

    barcode_size = [sum(row[1:4]), sum(row[5:8]), sum(row[21:24]), sum(row[33:36]), sum(row[37:40])]

    for timepoint in timepoints:
        timepoint_all_present = all(barcode_size)
        timepoint_total = sum([barcode_dict[timepoint + '_' + state] for state in states])
        if timepoint_all_present:
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
            barcode_summary['proportion_diff'].append(tuple(
                np.array(barcode_summary['ternary_coord'][i + 1]) - np.array(barcode_summary['ternary_coord'][i])))
            barcode_summary['vector_size'].append(
                vector_size(barcode_summary['vector'][i][0], barcode_summary['vector'][i][1]))
        for size in barcode_summary['size']:
            all_size_set.add(round(size, 3))
        for size_ in barcode_summary['vector_size']:
            all_vector_size_set.add(round(size_, 3))
        all_barcode_list.append(barcode_summary)
all_barcode_list.sort(key=lambda barcode: barcode['total_transition_amount'])

timepoint_vector_dict = {transition: np.array([barcode['vector'][transition] for barcode in all_barcode_list]) for
                         transition in range(4)}
simulated_timepoint_vector_dict = {transition: [] for transition in range(4)}
timepoint_position_dict = {timepoint: [] for timepoint in range(5)}
for index, barcode in enumerate(all_barcode_list):
    timepoint_position_dict[0].append(np.array(barcode['cartesian_coord'][0]))

for transition in range(4):
    # find clustering
    distance_vector = np.array(timepoint_position_dict[transition])
    db = DBSCAN(eps=0.5, min_samples=100).fit(distance_vector)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    for index, barcode in enumerate(all_barcode_list):
        x = timepoint_position_dict[transition][index][0]
        y = timepoint_position_dict[transition][index][1]
        if labels[index] == 0:
            if random.random() <= 0.1:  # this means they are changing the states
                dx = random.expovariate(1 / 6) - x
                if dx >= 0:
                    dx = 0
                y_new = random.uniform(x + dx, 10)
                dy = y_new - y
                simulated_timepoint_vector_dict[transition].append([dx, dy])
                timepoint_position_dict[transition + 1].append(
                    timepoint_position_dict[transition][index] + np.array([dx, dy]))
            else:
                simulated_timepoint_vector_dict[transition].append([0, 0])
                timepoint_position_dict[transition + 1].append(timepoint_position_dict[transition][index])
        else:
            x_new = random.gauss(9.5, 0.75)
            y_new = random.gauss(9.5, 0.75)
            if x_new > 10:
                x_new = 10
            if y_new < x_new:
                y_new = x_new
            if y_new > 10:
                y_new = 10
            dx = x_new - x
            dy = y_new - y
            simulated_timepoint_vector_dict[transition].append([dx, dy])
            timepoint_position_dict[transition + 1].append(
                timepoint_position_dict[transition][index] + np.array([dx, dy]))

timepoint_position_dict = {timepoint: [] for timepoint in range(5)}

for timepoint in range(5):
    for index, barcode in enumerate(all_barcode_list):
        if timepoint == 0:
            timepoint_position_dict[timepoint].append(np.array(barcode['cartesian_coord'][0]))
        else:
            timepoint_position_dict[timepoint].append(np.array(simulated_timepoint_vector_dict[timepoint - 1][index]) +
                                                      timepoint_position_dict[timepoint - 1][index])

top_right_coord = (10, 10)
top_left_coord = (0, 10)
bottom_left_coord = (0, 0)

triangle_vertices = np.array([top_right_coord, top_left_coord, bottom_left_coord])

gss = matplotlib.gridspec.GridSpec(3, 3, height_ratios=[1, 1, 0.1])


def steady_state_simulation():
    for timepoint in range(4):
        T_0 = np.zeros((len(all_barcode_list), 3))
        for index, barcode in enumerate(all_barcode_list):
            ternary_coord = barcode['ternary_coord']
            size = barcode['size']
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


def least_square_estimation_all_separate_timepoint(all_barcode_list):
    probability_timepoint_list = []
    for timepoint in range(4):
        P = np.zeros((3, 3))
        T_0 = np.zeros((len(all_barcode_list), 3))
        T_1 = np.zeros((len(all_barcode_list), 3))
        total_size = 0
        length = 0
        for index, barcode in enumerate(all_barcode_list):
            ternary_coord = barcode['ternary_coord']
            size = barcode['size']
            T_0[index] = np.array(ternary_coord[timepoint])
            T_1[index] = np.array(ternary_coord[timepoint + 1])
            T_0_t = np.transpose(T_0)
        probability_timepoint_list.append(np.matmul(pinv(np.matmul(T_0_t, T_0)), np.matmul(T_0_t, T_1)))
    return probability_timepoint_list


transitional_prob_list = least_square_estimation_all_separate_timepoint(all_barcode_list)

steady_state_population = steady_state_simulation().mean(axis=0)


def ternary_conversion(cartesian_coord):
    x = cartesian_coord[0] / 10
    y = (cartesian_coord[1] - cartesian_coord[0]) / 10
    z = 1 - x - y
    return x, y, z


color_scalarMap = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=1), cmap='rainbow')

for timepoint in range(5):
    fig, ax = plt.subplots()
    ax.axis('off')
    if timepoint == 5:
        break
    ax.plot([0, 0], [0, 10], color='black', lw=0.5)
    ax.plot([0, 10], [10, 10], color='black', lw=0.5)
    ax.plot([10, 0], [10, 0], color='black', lw=0.5)
    for barcode in timepoint_position_dict[timepoint]:
        ax.scatter(barcode[0], barcode[1], marker='.',
                   color=color_scalarMap.to_rgba(entropy(ternary_conversion(barcode))))
    ax.set_title('Day {}'.format(timepoint * 6))
    plt.savefig("VelocityModel_DotPlot_{}.svg".format(timepoint), bbox_inches='tight', format='svg', dpi=720)
    plt.close()

from scipy import stats

empirical_stochasticity = {i: [] for i in range(5)}
simulated_stochasticity = {i: [] for i in range(5)}

for barcode in all_barcode_list:
    for timepoint in range(5):
        empirical_stochasticity[timepoint].append(entropy(barcode['ternary_coord'][timepoint]))

for timepoint in range(5):
    for barcode in timepoint_position_dict[timepoint]:
        simulated_stochasticity[timepoint].append(entropy(ternary_conversion(barcode)))

bins_ = [i * 1 / 200 for i in range(201)]

for timepoint in range(5):
    fig, ax = plt.subplots()
    ax.hist(empirical_stochasticity[timepoint], bins=bins_, density=True, histtype='step', cumulative=True,
            label='Empirical')
    ax.hist(simulated_stochasticity[timepoint], bins=bins_, density=True, histtype='step', cumulative=True,
            label='Simulation')
    statistic = tuple(stats.ks_2samp(empirical_stochasticity[timepoint], simulated_stochasticity[timepoint],
                                     alternative='two-sided', mode='auto'))
    print(timepoint, statistic[0], statistic[1])
    plt.savefig("VelocotyModel_CDF_{}.svg".format(timepoint), bbox_inches='tight', format='svg', dpi=720)
    plt.close()
