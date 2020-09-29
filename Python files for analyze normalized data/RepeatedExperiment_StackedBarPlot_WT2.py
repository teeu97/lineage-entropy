import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from tqdm.auto import tqdm

total_cell_number = 10**8

state_1_ratio = 0.90
state_2_ratio = 0.05
state_3_ratio = 0.05

state_1_number = state_1_ratio * total_cell_number
state_2_number = state_2_ratio * total_cell_number
state_3_number = state_3_ratio * total_cell_number

print('WT2')

normalizing_factor = [total_cell_number, state_1_number, state_2_number, state_3_number] * 20

table = pickle.load(open('20200628_finished_table.pickle', 'rb'))
table_2 = pickle.load(open('20200713_finished_table.pickle', 'rb'))

combined_table = pd.concat([table, table_2], axis=1, sort=False)
combined_table.fillna(0, inplace=True)

sum_table = combined_table.sum(axis=0)
normalized_table = combined_table.div(sum_table)
true_number_table = (normalized_table * normalizing_factor).round()

all_barcode_number_list = []
for barcode, row in true_number_table.iterrows():
    new_list = [1, sum(row[5:8]), sum(row[29:32]), sum(row[45:48]), sum(row[61:64]), sum(row[73:76])]
    all_barcode_number_list.append(new_list)

rainbow = cm.get_cmap('rainbow', len(all_barcode_number_list))
rainbow_list = rainbow(range(len(all_barcode_number_list)))

df_all_barcode_number = pd.DataFrame(all_barcode_number_list)
df_total = df_all_barcode_number.sum(axis=0)
df_percentage = (df_all_barcode_number/df_total).sort_values(by=0, ascending=False)
df_percentage_sum = df_percentage.sum(axis=0)

complexity = df_all_barcode_number[df_all_barcode_number > 0.0].count()
timepoint = np.arange(len(new_list))
previous_barcode = None
index = 0
width = 0.5
for i_1, i_2 in tqdm(df_percentage.iterrows(), total=df_percentage.shape[0]):
    barcode = i_2
    if previous_barcode is None:
        for i in range(len(new_list)-1):
            plt.fill_between([i+width/2, i+1-width/2], [0, 0], [barcode[i], barcode[i+1]], color=rainbow_list[index], alpha=0.5)
        plt.bar(timepoint, barcode, color=rainbow_list[index], edgecolor = 'black', linewidth=0.03, width=width)
        previous_barcode = barcode
        for j in range(len(new_list)-1):
            plt.plot([j+width/2, j+1-width/2], [previous_barcode[j], previous_barcode[j+1]], color='black', linewidth=0.05)
    else:
        for k in range(len(new_list)-1):
            plt.fill_between([k+width/2, k+1-width/2], [previous_barcode[k], previous_barcode[k+1]], [previous_barcode[k]+barcode[k], previous_barcode[k+1]+barcode[k+1]], color=rainbow_list[index], alpha=0.75)
        plt.bar(timepoint, barcode, bottom=previous_barcode, color=rainbow_list[index], edgecolor = 'black', linewidth=0.03, width=width)
        previous_barcode += barcode
        for l in range(len(new_list)-1):
            plt.plot([l+width/2, l+1-width/2], [previous_barcode[l], previous_barcode[l+1]], color='black', linewidth=0.05)
    index += 1
plt.xticks(timepoint, ['Timepoint\n\n# Distinct\nLineages'] + ['Day {} WT2 \n\n'.format(i*6) + str(complexity[i+1]) for i in range(5)])
plt.ylabel('Proportion of Total Cells')
plt.savefig('RepeatedExperiment_StackedBarPlot_WT2.svg', bbox_inches='tight', format='svg', dpi=720)
