import pickle
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm

def euclidean_distance(coor_1, coor_2):
    return math.sqrt(sum((i - j)**2 for i, j in zip(coor_1, coor_2)))

matplotlib.rcParams['figure.dpi'] = 500
matplotlib.rcParams['figure.figsize'] = (4, 4)
matplotlib.rcParams['font.family'] = "sans-serif"

total_cell_number = 10**8

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

    barcode_summary = {'ternary_coord': [], 'cartesian_coord': [], 'vector': [], 'size': [], 'assigned_state': []}

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
            barcode_summary['vector'].append((barcode_summary['cartesian_coord'][i + 1][0] -
                                              barcode_summary['cartesian_coord'][i][0],
                                              barcode_summary['cartesian_coord'][i + 1][1] -
                                              barcode_summary['cartesian_coord'][i][1]))
        for size in barcode_summary['size']:
            all_size_set.add(round(size, 3))
        all_barcode_list.append(barcode_summary)

state_length = [0, 0, 0]
color_scalarMap = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.LogNorm(vmin = 1, vmax = max(all_size_set)), cmap='YlOrRd')
for i in range(4):
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 4))
    fig.suptitle('Day {} to {}'.format(i*6, (i+1)*6))
    for j in range(3):
        length = 0
        axs[j].axis([-1.5, 1.5, -2, 2])
        for barcode in all_barcode_list:
            vector = barcode['vector']
            size = barcode['size']
            assigned_state_list = barcode['assigned_state']
            if assigned_state_list[i] == j:
                length += 1
                barcode_color = color_scalarMap.to_rgba(round(size[i + 1], 3))
                axs[j].arrow(0, 0, vector[i][0], vector[i][1], shape='full', head_width=0.01, color=barcode_color)

        axs[j].set_title('State ' + str(j + 1) + '(' + str(length) + ')')
        axs[j].axis('off')

    axins = inset_axes(axs[2],
                       width="5%",
                       height="100%",
                       loc='lower left',
                       bbox_to_anchor=(1.05, 0., 1, 1),
                       bbox_transform=axs[2].transAxes,
                       borderpad=0,
                       )

    cbar = plt.colorbar(color_scalarMap, cax=axins)
    cbar.set_label('Lineage Size', rotation=270, labelpad=10)
    fig.savefig('RecenteredPlots_EachTimepoint_T{}.svg'.format(i), bbox_inches='tight',
                format='svg', dpi=720)


