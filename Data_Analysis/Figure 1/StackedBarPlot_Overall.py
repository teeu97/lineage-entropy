"""
This file produces a stacked barplot that shows the number of lineages over time
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

__author__ = 'Tee Udomlumleart'
__maintainer__ = 'Tee Udomlumleart'
__email__ = ['teeu@mit.edu', 'salilg@mit.edu']
__status__ = 'Production'

# normalize the reads
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
    d0_all, d0_s1, d0_s2, d0_s3 = row[0:4]
    d6_all, d6_s1, d6_s2, d6_s3 = row[4:8]
    d9_all, d9_s1, d9_s2, d9_s3 = row[16:20]
    d12_all, d12_s1, d12_s2, d12_s3 = row[20:24]
    d18_all, d18_s1, d18_s2, d18_s3 = row[32:36]
    d24_all, d24_s1, d24_s2, d24_s3 = row[36:40]
    new_list = [1, sum(row[1:4]), sum(row[5:8]), sum(row[21:24]), sum(row[33:36]), sum(row[37:40])]
    all_barcode_number_list.append(new_list)

# each color represents each lineage
rainbow = cm.get_cmap('rainbow', len(all_barcode_number_list))
rainbow_list = rainbow(range(len(all_barcode_number_list)))

# convert the size of each barcode on each timepoint into percentages
df_all_barcode_number = pd.DataFrame(all_barcode_number_list)
df_total = df_all_barcode_number.sum(axis=0)
df_percentage = (df_all_barcode_number/df_total).sort_values(by=0, ascending=False)
df_percentage_sum = df_percentage.sum(axis=0)

# complexity is the number barcodes that have more than 0 reads in that timepoint
complexity = df_all_barcode_number[df_all_barcode_number > 0.0].count()
timepoint = np.arange(6)
previous_barcode = None
index = 0
width = 0.5
# produce stacked bar plots
for i_1, i_2 in df_percentage.iterrows():
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
plt.savefig('StackedBarplot_Overall.png', bbox_inches='tight', format='png', dpi=720)
plt.show()
