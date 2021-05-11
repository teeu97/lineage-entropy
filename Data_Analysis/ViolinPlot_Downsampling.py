import pickle
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from matplotlib import ticker as mticker


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


percentages = ['0.01', '0.1', '1', '10', '25', '50', '75', '90', '100']

y_data_percentages = []
y_data_sizes = []

for percentage in percentages:

    total_cell_number = 10 ** 8

    state_1_ratio = 0.90
    state_2_ratio = 0.05
    state_3_ratio = 0.05

    state_1_number = state_1_ratio * total_cell_number
    state_2_number = state_2_ratio * total_cell_number
    state_3_number = state_3_ratio * total_cell_number

    normalizing_factor = [total_cell_number, state_1_number, state_2_number, state_3_number] * 10

    table = pickle.load(open('20200501_finished_table_{}_AnalyzedReady.pickle'.format(percentage), 'rb'))

    sum_table = table.sum(axis=0)
    normalized_table = table.div(sum_table)
    true_number_table = (normalized_table * normalizing_factor).round()
    total_number_reads = table.sum().sum()

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
                           'total_transition_amount': 0}

        barcode_size = [sum(row[1:4]), sum(row[5:8]), sum(row[21:24]), sum(row[33:36]), sum(row[37:40])]

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

    barcode_number = len(all_barcode_list)

    all_transition_amounts = []
    all_population_sizes = []

    for barcode in all_barcode_list:
        average_size = sum(barcode['size'][1:]) / len(barcode['size'][1:])
        all_transition_amounts.append(barcode['total_transition_amount'])
        all_population_sizes.append(np.log10(average_size))

    y_data_percentages.append(np.array(all_transition_amounts) * 100 / (math.sqrt(2)*4))
    y_data_sizes.append(all_population_sizes)

fig = plt.figure()
ax = plt.axes()
ax.set_title('Distribution of Percent Transition')
ax.set_ylabel('Percent Transition')
ax.set_xlabel('Percent Sampling')
ax.violinplot(y_data_percentages, showmedians=True)
ax.get_xaxis().set_tick_params(direction='out')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(np.arange(1, 10))
ax.set_xticklabels(percentages)
ax.set_xlim(0.25, len(percentages) + 0.75)
plt.savefig('ViolinPlot_Downsampling_TransitionPercent.tiff', bbox_inches='tight', format='tiff', dpi=720)

fig = plt.figure()
ax = plt.axes()
ax.set_ylim(bottom=0, top=7)
ax.set_title('Distribution of Lineage Size')
ax.set_ylabel('Lineage Size')
ax.set_xlabel('Percent Sampling')
ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
ax.yaxis.set_ticks([np.log10(x) for p in range(0,6) for x in np.linspace(10**p, 10**(p+1), 10)], minor=True)
ax.violinplot(y_data_sizes, showmedians=True)
ax.get_xaxis().set_tick_params(direction='out')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(np.arange(1, 10))
ax.set_xticklabels(percentages)
ax.set_xlim(0.25, len(percentages) + 0.75)
plt.savefig("ViolinPlot_Downsampling_LineageSize.tiff", bbox_inches='tight', format='tiff', dpi=720)