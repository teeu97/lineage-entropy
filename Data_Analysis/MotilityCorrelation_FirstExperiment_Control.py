import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import random

from tqdm.auto import tqdm

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


def correlation_randomized_WT_p():
    rho_list, r_list = [[] for i in range(3)], [[] for i in range(3)]
    for attempt in tqdm(range(100000)):
        all_possible_motility = [[] for transition in range(3)]
        all_motility = [[] for transition in range(4)]

        for barcode in all_barcode_list:
            for timepoint in range(4):
                if timepoint > 0:
                    all_possible_motility[timepoint - 1].append(barcode['transition_amount'][timepoint])
                all_motility[timepoint].append(barcode['transition_amount'][timepoint])

        all_randomized_motility = []
        for timepoint in range(3):
            timepoint_list = []
            for barcode in all_barcode_list:
                random_ = random.choice(all_possible_motility[timepoint])
                timepoint_list.append(random_)
                all_possible_motility[timepoint].remove(random_)
            all_randomized_motility.append(timepoint_list)

        for i in range(3):
            x = all_motility[i]
            y = all_randomized_motility[i]
            slope, intercept, r, p, stderr = scipy.stats.linregress(x, y)
            rho, p_ = scipy.stats.spearmanr(x, y)

            rho_list[i].append(rho)
            r_list[i].append(r)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=[8 * 2, 6])
    ax1.violinplot(r_list, vert=True, showmeans=True, showmedians=True)
    ax1.set_title('r values')
    ax2.violinplot(rho_list, vert=True, showmeans=True, showmedians=True)
    ax2.set_title('rho values')

    def set_axis_style(ax, labels):
        ax.get_xaxis().set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlabel('Transition')

    labels = ['{} to {} vs {} to {}'.format(i*6, (i+1)*6, (i+1)*6, (i+2)*6) for i in range(2)]
    for ax in [ax1, ax2]:
        set_axis_style(ax, labels)
    plt.savefig("MotilityCorrelation_FirstExperiment_Control.svg", bbox_inches='tight', format='svg', dpi=720)

correlation_randomized_WT_p()