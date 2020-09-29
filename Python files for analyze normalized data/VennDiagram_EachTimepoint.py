import pickle
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib_venn import venn3_unweighted

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

timepoints = ['Day 0', 'Day 6', 'Day 12', 'Day 18', 'Day 24']
timepoints_ = ['d{}'.format(i*6) for i in range(5)]

states = ['s1', 's2', 's3']
states_coords = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

all_size_set = set()
all_vector_size_set = set()
all_barcode_list = []

top_right_coord = (10, 10)
top_left_coord = (0, 10)
bottom_left_coord = (0, 0)

triangle_vertices = np.array([top_right_coord, top_left_coord, bottom_left_coord])

all_barcode_number_dict = {timepoint: [0 for i in range(7)] for timepoint in timepoints}
barcode_number = 0
intersect_number = 0

for barcode, row in true_number_table.iterrows():
    barcode_dict = {}

    barcode_dict['d0_all'], barcode_dict['d0_s1'], barcode_dict['d0_s2'], barcode_dict['d0_s3'] = row[0:4]
    barcode_dict['d6_all'], barcode_dict['d6_s1'], barcode_dict['d6_s2'], barcode_dict['d6_s3'] = row[4:8]
    barcode_dict['d12_all'], barcode_dict['d12_s1'], barcode_dict['d12_s2'], barcode_dict['d12_s3'] = row[20:24]
    barcode_dict['d18_all'], barcode_dict['d18_s1'], barcode_dict['d18_s2'], barcode_dict['d18_s3'] = row[32:36]
    barcode_dict['d24_all'], barcode_dict['d24_s1'], barcode_dict['d24_s2'], barcode_dict['d24_s3'] = row[36:40]

    barcode_summary = {'ternary_coord': [], 'cartesian_coord': [], 'vector': [], 'size': [], 'assigned_state': [],
                       'vector_size': [], 'monte_ternary_coord': [], 'monte_cartesian_coord': [], 'monte_vector': [],
                       'timepoint_size': [], 'total_transition_amount': 0}

    barcode_size = [sum(row[1:4]), sum(row[5:8]), sum(row[21:24]), sum(row[33:36]), sum(row[37:40])]

    for timepoint in timepoints_:
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

    if len(barcode_summary['cartesian_coord']) == 5:
        barcode_number += 1
        if barcode_dict['d0_s1'] and not barcode_dict['d0_s2'] and not barcode_dict['d0_s3']:
            all_barcode_number_dict['Day 0'][1] += 1
        if barcode_dict['d0_s2'] and not barcode_dict['d0_s1'] and not barcode_dict['d0_s3']:
            all_barcode_number_dict['Day 0'][0] += 1
        if barcode_dict['d0_s3'] and not barcode_dict['d0_s1'] and not barcode_dict['d0_s2']:
            all_barcode_number_dict['Day 0'][3] += 1
        if barcode_dict['d0_s1'] and barcode_dict['d0_s2'] and not barcode_dict['d0_s3']:
            all_barcode_number_dict['Day 0'][2] += 1
        if barcode_dict['d0_s1'] and barcode_dict['d0_s3'] and not barcode_dict['d0_s2']:
            all_barcode_number_dict['Day 0'][5] += 1
        if barcode_dict['d0_s2'] and barcode_dict['d0_s3'] and not barcode_dict['d0_s1']:
            all_barcode_number_dict['Day 0'][4] += 1
        if barcode_dict['d0_s1'] and barcode_dict['d0_s2'] and barcode_dict['d0_s3']:
            all_barcode_number_dict['Day 0'][6] += 1

        if barcode_dict['d6_s1'] and not barcode_dict['d6_s2'] and not barcode_dict['d6_s3']:
            all_barcode_number_dict['Day 6'][1] += 1
        if barcode_dict['d6_s2'] and not barcode_dict['d6_s1'] and not barcode_dict['d6_s3']:
            all_barcode_number_dict['Day 6'][0] += 1
        if barcode_dict['d6_s3'] and not barcode_dict['d6_s1'] and not barcode_dict['d6_s2']:
            all_barcode_number_dict['Day 6'][3] += 1
        if barcode_dict['d6_s1'] and barcode_dict['d6_s2'] and not barcode_dict['d6_s3']:
            all_barcode_number_dict['Day 6'][2] += 1
        if barcode_dict['d6_s1'] and barcode_dict['d6_s3'] and not barcode_dict['d6_s2']:
            all_barcode_number_dict['Day 6'][5] += 1
        if barcode_dict['d6_s2'] and barcode_dict['d6_s3'] and not barcode_dict['d6_s1']:
            all_barcode_number_dict['Day 6'][4] += 1
        if barcode_dict['d6_s1'] and barcode_dict['d6_s2'] and barcode_dict['d6_s3']:
            all_barcode_number_dict['Day 6'][6] += 1

        if barcode_dict['d12_s1'] and not barcode_dict['d12_s2'] and not barcode_dict['d12_s3']:
            all_barcode_number_dict['Day 12'][1] += 1
        if barcode_dict['d12_s2'] and not barcode_dict['d12_s1'] and not barcode_dict['d12_s3']:
            all_barcode_number_dict['Day 12'][0] += 1
        if barcode_dict['d12_s3'] and not barcode_dict['d12_s1'] and not barcode_dict['d12_s2']:
            all_barcode_number_dict['Day 12'][3] += 1
        if barcode_dict['d12_s1'] and barcode_dict['d12_s2'] and not barcode_dict['d12_s3']:
            all_barcode_number_dict['Day 12'][2] += 1
        if barcode_dict['d12_s1'] and barcode_dict['d12_s3'] and not barcode_dict['d12_s2']:
            all_barcode_number_dict['Day 12'][5] += 1
        if barcode_dict['d12_s2'] and barcode_dict['d12_s3'] and not barcode_dict['d12_s1']:
            all_barcode_number_dict['Day 12'][4] += 1
        if barcode_dict['d12_s1'] and barcode_dict['d12_s2'] and barcode_dict['d12_s3']:
            all_barcode_number_dict['Day 12'][6] += 1

        if barcode_dict['d18_s1'] and not barcode_dict['d18_s2'] and not barcode_dict['d18_s3']:
            all_barcode_number_dict['Day 18'][1] += 1
        if barcode_dict['d18_s2'] and not barcode_dict['d18_s1'] and not barcode_dict['d18_s3']:
            all_barcode_number_dict['Day 18'][0] += 1
        if barcode_dict['d18_s3'] and not barcode_dict['d18_s1'] and not barcode_dict['d18_s2']:
            all_barcode_number_dict['Day 18'][3] += 1
        if barcode_dict['d18_s1'] and barcode_dict['d18_s2'] and not barcode_dict['d18_s3']:
            all_barcode_number_dict['Day 18'][2] += 1
        if barcode_dict['d18_s1'] and barcode_dict['d18_s3'] and not barcode_dict['d18_s2']:
            all_barcode_number_dict['Day 18'][5] += 1
        if barcode_dict['d18_s2'] and barcode_dict['d18_s3'] and not barcode_dict['d18_s1']:
            all_barcode_number_dict['Day 18'][4] += 1
        if barcode_dict['d18_s1'] and barcode_dict['d18_s2'] and barcode_dict['d18_s3']:
            all_barcode_number_dict['Day 18'][6] += 1

        if barcode_dict['d24_s1'] and not barcode_dict['d24_s2'] and not barcode_dict['d24_s3']:
            all_barcode_number_dict['Day 24'][1] += 1
        if barcode_dict['d24_s2'] and not barcode_dict['d24_s1'] and not barcode_dict['d24_s3']:
            all_barcode_number_dict['Day 24'][0] += 1
        if barcode_dict['d24_s3'] and not barcode_dict['d24_s1'] and not barcode_dict['d24_s2']:
            all_barcode_number_dict['Day 24'][3] += 1
        if barcode_dict['d24_s1'] and barcode_dict['d24_s2'] and not barcode_dict['d24_s3']:
            all_barcode_number_dict['Day 24'][2] += 1
        if barcode_dict['d24_s1'] and barcode_dict['d24_s3'] and not barcode_dict['d24_s2']:
            all_barcode_number_dict['Day 24'][5] += 1
        if barcode_dict['d24_s2'] and barcode_dict['d24_s3'] and not barcode_dict['d24_s1']:
            all_barcode_number_dict['Day 24'][4] += 1
        if barcode_dict['d24_s1'] and barcode_dict['d24_s2'] and barcode_dict['d24_s3']:
                all_barcode_number_dict['Day 24'][6] += 1

        if barcode_dict['d0_s1'] and barcode_dict['d0_s2'] and barcode_dict['d0_s3'] and\
            barcode_dict['d6_s1'] and barcode_dict['d6_s2'] and barcode_dict['d6_s3'] and\
            barcode_dict['d12_s1'] and barcode_dict['d12_s2'] and barcode_dict['d12_s3'] and\
            barcode_dict['d18_s1'] and barcode_dict['d18_s2'] and barcode_dict['d18_s3'] and\
            barcode_dict['d24_s1'] and barcode_dict['d24_s2'] and barcode_dict['d24_s3']:
            intersect_number += 1

print(barcode_number)
print(intersect_number)

def venn_diagram(all_barcode_number_dict):
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=[12, 8])
    plt.suptitle("Distribution of Lineages")
    for iy in range(2):
        for ix in range(3):
            i = 3 * iy + ix
            ax = axes[iy, ix]
            ax.axis('off')
            if i == 5:
                break
            ax.set_title('Day {}'.format(6*i))
            timepoint_list = all_barcode_number_dict['Day {}'.format(6*i)]
            v = venn3_unweighted(subsets=timepoint_list, set_labels = ('S2', 'S1', 'S3'),
                                 set_colors=('#70A1D7', '#F47C7C','#A1DE93'), ax=ax)
    plt.savefig('VennDiagram_EachTimepoint.svg', format='svg', dpi=720)
    plt.show()

venn_diagram(all_barcode_number_dict)