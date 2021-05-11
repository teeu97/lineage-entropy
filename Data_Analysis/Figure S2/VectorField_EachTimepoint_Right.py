import pickle
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from matplotlib.lines import Line2D
from matplotlib import cm


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

def vector_field_size_weight_shifted_size(all_barcode_list):
    default_size = 3
    color_scalarMap = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.LogNorm(vmin=1, vmax=max(all_size_set)),
                                                   cmap='YlOrRd')
    custom_lines = [Line2D([0], [0], color='black', lw=math.sqrt(default_size ** (i))) for i in range(4)]
    for timepoint in tqdm(range(4)):
        fig = plt.figure()
        triangle_coord = []
        vector_dict = {}
        vector_size_list = []
        for i in range(11):
            for j in range(i + 1):
                triangle_coord.append((j, i))
                vector_dict[(j, i)] = [0, 0, 0, 0]
        for barcode in all_barcode_list:
            cartesian_coord = barcode['cartesian_coord']
            vector = barcode['vector']
            size = barcode['size']
            x_coord = cartesian_coord[timepoint][0]
            y_coord = cartesian_coord[timepoint][1]
            local_vector = vector_dict[(round(x_coord), round(y_coord))]
            local_vector[0] += (vector[timepoint][0]) * size[timepoint + 1]
            local_vector[1] += (vector[timepoint][1]) * size[timepoint + 1]
            local_vector[2] += size[timepoint + 1]
            local_vector[3] += 1
        for coord in vector_dict:
            if vector_dict[coord][2] == 0:
                vector_dict[coord][2] = 1
        for coord in vector_dict:
            vector_dict[coord][0] /= vector_dict[coord][2]
            vector_dict[coord][1] /= vector_dict[coord][2]
        for coord in vector_dict:
            if vector_dict[coord][2] > 1:
                arrow_color = color_scalarMap.to_rgba(vector_dict[coord][2])
                vector_magnitude = vector_size(4 * vector_dict[coord][0], 4 * vector_dict[coord][1])
                dot_size = default_size ** (math.floor(math.log10(vector_dict[coord][3])))
                line_size = math.sqrt(dot_size)
                plt.scatter(coord[0], coord[1], marker='.', color=arrow_color, s=dot_size)
                plt.arrow(coord[0], coord[1], vector_dict[coord][0] / 15 + vector_dict[coord][0] / vector_magnitude,
                          vector_dict[coord][1] / 15 + vector_dict[coord][1] / vector_magnitude, shape='full',
                          head_width=0.1, color=arrow_color, linewidth=line_size, length_includes_head=True)
        # plt.title('Day ' + str(timepoint * 6) + ' to ' + str((timepoint + 1) * 6))
        # plt.arrow(6.5, 3.5, 17 / 30, 0, shape='full', head_width=0.1, color='black',
        #           linewidth=math.sqrt(default_size ** (0)), length_includes_head=True)
        # plt.arrow(6.5, 3, 5 / 6, 0, shape='full', head_width=0.1, color='black',
        #           linewidth=math.sqrt(default_size ** (0)), length_includes_head=True)
        # plt.arrow(6.5, 2.5, 7 / 6, 0, shape='full', head_width=0.1, color='black',
        #           linewidth=math.sqrt(default_size ** (0)), length_includes_head=True)
        # plt.text(6.25 + 47 / 30, 3.4, '10% of Transition', size='small')
        # plt.text(6.25 + 47 / 30, 2.9, '50% of Transition', size='small')
        # plt.text(6.25 + 47 / 30, 2.4, '100% of Transition', size='small')
        # plt.text(11, 9.85, 'State 1')
        # plt.text(-1.5, 9.85, 'State 2')
        # plt.text(-1.5, -0.15, 'State 3')
        plt.axis('off')
        # plt.legend(custom_lines, ['[0, 10)', '[10, 100)', '[100, 1000)', '[1000, 10000)]'],
        #            title='Number of Distinct Barcodes', loc='lower right', fontsize='small', framealpha=1,
        #            edgecolor='white')
        # cbaxes = fig.add_axes([0.95, 3 / 22.5, 0.05, 0.65])
        # cbar = plt.colorbar(color_scalarMap, pad=0.05, shrink=0.8, cax=cbaxes)
        # cbar.set_label('Lineage Size', rotation=270, labelpad=10)
        plt.savefig('VectorField_EachTimepoint_Right_D{}.svg'.format(timepoint*6), bbox_inches='tight', format='svg', dpi=720)

vector_field_size_weight_shifted_size(all_barcode_list)