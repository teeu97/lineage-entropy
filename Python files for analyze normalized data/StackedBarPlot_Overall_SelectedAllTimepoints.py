import pickle
import math
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from tqdm.auto import tqdm

def euclidean_distance(coor_1, coor_2):
    return math.sqrt(sum((i - j) ** 2 for i, j in zip(coor_1, coor_2)))


def vector_size(x_displacement, y_displacement):
    return math.sqrt(x_displacement ** 2 + y_displacement ** 2)

matplotlib.rcParams['font.sans-serif'] = "Helvetica"
matplotlib.rcParams['font.family'] = "sans-serif"

timepoints = ['d0', 'd6', 'd12', 'd18', 'd24']
states = ['s1', 's2', 's3']

top_right_coord = (1, 1)
top_left_coord = (0, 1)
bottom_left_coord = (0, 0)

triangle_vertices = np.array([top_right_coord, top_left_coord, bottom_left_coord])

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

all_barcode_number_list = []
for barcode, row in true_number_table.iterrows():
    barcode_dict = {}

    barcode_dict['d0_all'], barcode_dict['d0_s1'], barcode_dict['d0_s2'], barcode_dict['d0_s3'] = row[0:4]
    barcode_dict['d6_all'], barcode_dict['d6_s1'], barcode_dict['d6_s2'], barcode_dict['d6_s3'] = row[4:8]
    barcode_dict['d12_all'], barcode_dict['d12_s1'], barcode_dict['d12_s2'], barcode_dict['d12_s3'] = row[20:24]
    barcode_dict['d18_all'], barcode_dict['d18_s1'], barcode_dict['d18_s2'], barcode_dict['d18_s3'] = row[32:36]
    barcode_dict['d24_all'], barcode_dict['d24_s1'], barcode_dict['d24_s2'], barcode_dict['d24_s3'] = row[36:40]

    new_list = [1, sum(row[1:4]), sum(row[5:8]), sum(row[21:24]), sum(row[33:36]), sum(row[37:40])]
    barcode_size = [sum(row[1:4]), sum(row[5:8]), sum(row[21:24]), sum(row[33:36]), sum(row[37:40])]

    barcode_summary = {'ternary_coord': [], 'cartesian_coord': []}

    for timepoint in timepoints:
        timepoint_all_present = all(barcode_size)
        timepoint_total = sum([barcode_dict[timepoint + '_' + state] for state in states])
        if timepoint_all_present:
            ternary_coord = []
            dist = []
            for state in states:
                ternary_coord.append(barcode_dict[timepoint + '_' + state] / timepoint_total)
            barcode_summary['ternary_coord'].append(ternary_coord)

            cartesian_coord = np.dot(np.array(ternary_coord), triangle_vertices)
            barcode_summary['cartesian_coord'].append(list(cartesian_coord))

    if len(barcode_summary['cartesian_coord']) == 5:
        all_barcode_number_list.append(new_list)

rainbow = cm.get_cmap('rainbow', len(all_barcode_number_list))
rainbow_list = rainbow(range(len(all_barcode_number_list)))

df_all_barcode_number = pd.DataFrame(all_barcode_number_list)
df_total = df_all_barcode_number.sum(axis=0)
df_percentage = (df_all_barcode_number/df_total).sort_values(by=0, ascending=False)
df_percentage_sum = df_percentage.sum(axis=0)

complexity = df_all_barcode_number[df_all_barcode_number > 0.0].count()
timepoint = np.arange(6)
previous_barcode = None
index = 0
width = 0.5
for i_1, i_2 in tqdm(df_percentage.iterrows(), total=df_percentage.shape[0]):
    barcode = i_2
    if previous_barcode is None:
        for i in range(5):
            plt.fill_between([i+width/2, i+1-width/2], [0, 0], [barcode[i], barcode[i+1]], color=rainbow_list[index], alpha=0.5)
        plt.bar(timepoint, barcode, color=rainbow_list[index], edgecolor = 'black', linewidth=0.03, width=width)
        previous_barcode = barcode
        for j in range(5):
            plt.plot([j+width/2, j+1-width/2], [previous_barcode[j], previous_barcode[j+1]], color='black', linewidth=0.05)
    else:
        for k in range(5):
            plt.fill_between([k+width/2, k+1-width/2], [previous_barcode[k], previous_barcode[k+1]], [previous_barcode[k]+barcode[k], previous_barcode[k+1]+barcode[k+1]], color=rainbow_list[index], alpha=0.75)
        plt.bar(timepoint, barcode, bottom=previous_barcode, color=rainbow_list[index], edgecolor = 'black', linewidth=0.03, width=width)
        previous_barcode += barcode
        for l in range(5):
            plt.plot([l+width/2, l+1-width/2], [previous_barcode[l], previous_barcode[l+1]], color='black', linewidth=0.05)
    index += 1
plt.xticks(timepoint, ['Timepoint\n\n# Distinct\nLineages', 'Day 0 \n\n' + str(complexity[1]), 'Day 6 \n\n' + str(complexity[2]), 'Day 12 \n\n' + str(complexity[3]), 'Day 18 \n\n' + str(complexity[4]), 'Day 24 \n\n' + str(complexity[5])])
plt.ylabel('Proportion of Total Cells')
plt.savefig('StackedBarplot_Overall_SelectedAllTimepoints.png', bbox_inches='tight', format='png', dpi=720)
plt.show()
