import pickle
import math
import random
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats

from matplotlib import cm
from tqdm.auto import tqdm


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
    shared_barcode = all_barcode_set_dict['WT1'] & all_barcode_set_dict['WT2']
    motility_dict = {}
    for barcode in all_barcode_list_dict['WT1']:
        if barcode['id'] in shared_barcode:
            motility_dict[barcode['id']] = [barcode['transition_amount']]
    for barcode in all_barcode_list_dict['WT2']:
        if barcode['id'] in shared_barcode:
            motility_dict[barcode['id']] += [barcode['transition_amount']]

    color_scalarMap = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=math.log10(101)), cmap='rainbow')
    rainbow = cm.get_cmap('rainbow', len(shared_barcode))
    rainbow_list = rainbow(range(len(shared_barcode)))[::-1]

    print('WT')
    for transition in range(4):
        fig, ax = plt.subplots()

        x = np.array([motility_dict[barcode][0][transition] for barcode in motility_dict])
        x /= math.sqrt(2)
        x *= 100
        x += 1
        x = np.log10(x)

        y = np.array([motility_dict[barcode][1][transition] for barcode in motility_dict])
        y /= math.sqrt(2)
        y *= 100
        y += 1
        y = np.log10(y)

        slope, intercept, r, p, stderr = scipy.stats.linregress(x, y)
        rho, p_ = scipy.stats.spearmanr(x, y)

        ax.plot(np.arange(0, max(x), 0.0001), intercept + slope * np.arange(0, max(x), 0.0001), color='#202020')
        ax.scatter(x, y, marker='.', color='#696969')
        print('transition', round(r, 3), round(r**2, 3))
        ax.set_ylim(0, max(y) + 0.05)
        ax.set_xlim(0, max(x) + 0.05)

        plt.savefig("MotilityCorrelation_RepeatedExperiment_TwoSamples_{}.tiff".format(transition), bbox_inches='tight', format='tiff', dpi=720)
        plt.close()

def correlation_randomized_WT():
    shared_barcode = all_barcode_set_dict['WT1'] & all_barcode_set_dict['WT2']
    all_possible_motility_1 = {timepoint: [] for timepoint in range(5)}
    all_possible_motility_2 = {timepoint: [] for timepoint in range(5)}

    motility_dict = {}

    for barcode in all_barcode_list_dict['WT1']:
        if barcode['id'] in shared_barcode:
            for timepoint in range(4):
                all_possible_motility_1[timepoint].append(barcode['transition_amount'][timepoint])
    for barcode in all_barcode_list_dict['WT2']:
        if barcode['id'] in shared_barcode:
            for timepoint in range(4):
                all_possible_motility_2[timepoint].append(barcode['transition_amount'][timepoint])

    for barcode_id in shared_barcode:
        all_timepoint_list = []
        for timepoint in range(4):
            timepoint_list = []

            random_1 = random.choice(all_possible_motility_1[timepoint])
            timepoint_list.append(random_1)
            all_possible_motility_1[timepoint].remove(random_1)

            random_2 = random.choice(all_possible_motility_2[timepoint])
            timepoint_list.append(random_2)
            all_possible_motility_2[timepoint].remove(random_2)

            all_timepoint_list.append(timepoint_list)
        motility_dict[barcode_id] = all_timepoint_list

    color_scalarMap = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.LogNorm(vmin=10e-1, vmax=100), cmap='rainbow')
    rainbow = cm.get_cmap('rainbow', len(shared_barcode))
    rainbow_list = rainbow(range(len(shared_barcode)))[::-1]

    print('control')
    for transition in range(4):
        fig, ax = plt.subplots()

        x = np.array([motility_dict[barcode][transition][0] for barcode in motility_dict])
        x /= math.sqrt(2)
        x *= 100
        x += 1
        x = np.log10(x)
        y = np.array([motility_dict[barcode][transition][1] for barcode in motility_dict])
        y /= math.sqrt(2)
        y *= 100
        y += 1
        y = np.log10(y)

        slope, intercept, r, p, stderr = scipy.stats.linregress(x, y)
        rho, p_ = scipy.stats.spearmanr(x, y)

        ax.plot(np.arange(0, max(x), 0.0001), intercept + slope * np.arange(0, max(x), 0.0001), color='#202020')
        ax.scatter(x, y, marker='.', color='#696969')
        print(transition, round(r, 3), round(r**2, 3))
        # ax.text(0, 1.5, 'r={}, rho={}'.format(round(r, 3), round(rho, 3)))
        ax.set_ylim(0, max(y) + 0.05)
        ax.set_xlim(0, max(x) + 0.05)
        plt.savefig("MotilityCorrelation_RepeatedExperiment_Control_{}.tiff".format(transition), bbox_inches='tight', format='tiff', dpi=720)
        plt.close()

def correlation_randomized_WT_p():
    rho_list, r_list = [[] for i in range(4)], [[] for i in range(4)]
    for attempt in tqdm(range(100000)):
        shared_barcode = all_barcode_set_dict['WT1'] & all_barcode_set_dict['WT2']
        all_possible_motility_1 = {timepoint: [] for timepoint in range(5)}
        all_possible_motility_2 = {timepoint: [] for timepoint in range(5)}

        motility_dict = {}

        for barcode in all_barcode_list_dict['WT1']:
            if barcode['id'] in shared_barcode:
                for timepoint in range(4):
                    all_possible_motility_1[timepoint].append(barcode['transition_amount'][timepoint])
        for barcode in all_barcode_list_dict['WT2']:
            if barcode['id'] in shared_barcode:
                for timepoint in range(4):
                    all_possible_motility_2[timepoint].append(barcode['transition_amount'][timepoint])

        for barcode_id in shared_barcode:
            all_timepoint_list = []
            for timepoint in range(4):
                timepoint_list = []

                random_1 = random.choice(all_possible_motility_1[timepoint])
                timepoint_list.append(random_1)
                all_possible_motility_1[timepoint].remove(random_1)

                random_2 = random.choice(all_possible_motility_2[timepoint])
                timepoint_list.append(random_2)
                all_possible_motility_2[timepoint].remove(random_2)

                all_timepoint_list.append(timepoint_list)
            motility_dict[barcode_id] = all_timepoint_list

        for transition in range(4):
            x = np.array([motility_dict[barcode][transition][0] for barcode in motility_dict])
            y = np.array([motility_dict[barcode][transition][1] for barcode in motility_dict])
            slope, intercept, r, p, stderr = scipy.stats.linregress(x, y)
            rho, p_ = scipy.stats.spearmanr(x, y)

            rho_list[transition].append(rho)
            r_list[transition].append(r)

    for x in r_list:
        print('mean', np.mean(x))
        print('std', np.std(x))

correlation_randomized_WT()
correlation_WT()
# correlation_randomized_WT_p()