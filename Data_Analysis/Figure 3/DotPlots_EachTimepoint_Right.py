"""
This file produces dot plots showing proportion of cells in different states in all timepoint
"""

import pickle
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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


def scatter_plot_right_each(all_barcode_list):
    color_scalarMap = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.LogNorm(vmin=1, vmax=max(all_size_set)),
                                                   cmap='YlOrRd')
    for i in range(5):
        fig, ax = plt.subplots(1, 1)
        x_all = []
        y_all = []
        c_all = []

        ax.axis('off')

        all_barcode_list.sort(key=lambda barcode: barcode['size'][i])
        # iterate through each barcode
        for barcode in all_barcode_list:
            cartesian_coord = barcode['cartesian_coord']
            size = barcode['size']
            # record x and y coordinates
            x_all.append(cartesian_coord[i][0])
            y_all.append(cartesian_coord[i][1])
            # assign color based on their size
            c_all.append(color_scalarMap.to_rgba(round(size[i], 3)))

        # plot
        ax.scatter(x_all, y_all, c=c_all, marker='.')
        # ax.set_title('Day {}'.format(6*i))

        if i == 5:
            axins = inset_axes(ax,
                           width="5%",
                           height="100%",
                           loc='lower left',
                           bbox_to_anchor=(1.05, 0., 1, 1),
                           bbox_transform=ax.transAxes,
                           borderpad=0,
                           )

            cbar = plt.colorbar(color_scalarMap, cax=axins)

            # cbar.set_label('$\log_{10}$ Lineage Size', labelpad=10)
        plt.savefig("DotPlots_EachTimepoint_Right_{}.svg".format(i), bbox_inches='tight', format='svg', dpi=720)

scatter_plot_right_each(all_barcode_list)