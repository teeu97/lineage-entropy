import csv
import matplotlib
import scipy.stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_dict = {'Gene': [], 'Sample': [], 'Mean Fold Expression': [], 'Std Fold Expression': []}
fold_expression_list = []

with open('fold_induction_data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for index, row in enumerate(csv_reader):
        if index > 0 and ''.join(row).strip():
            data_dict['Gene'].append(row[0])
            data_dict['Sample'].append(row[1])
            fold_expression = []
            for column in range(2,5):
                fold_expression.append(float(row[column]))
            data_dict['Mean Fold Expression'].append(np.mean(fold_expression))
            data_dict['Std Fold Expression'].append(np.std(fold_expression))
            fold_expression_list.append(fold_expression)

data_pd = pd.DataFrame.from_dict(data_dict)

labels = [val for ind, val in enumerate(list(data_pd['Gene'].values)) if ind % 2 == 0]
x = np.arange(len(labels))  # the label locations

width = 0.35  # the width of the bars

cd24_positive = [val for ind, val in enumerate(list(data_pd['Mean Fold Expression'].values)) if ind % 2 == 0]
cd24_positive_std = [val for ind, val in enumerate(list(data_pd['Std Fold Expression'].values)) if ind % 2 == 0]
cd24_negative = [val for ind, val in enumerate(list(data_pd['Mean Fold Expression'].values)) if ind % 2 != 0]
cd24_negative_std = [val for ind, val in enumerate(list(data_pd['Std Fold Expression'].values)) if ind % 2 != 0]

significance_list = []
for gene in range(len(fold_expression_list) // 2):
    t, p = scipy.stats.ttest_ind(fold_expression_list[2 * gene], fold_expression_list[(2 * gene) + 1])
    significance_list.append(p)

fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, cd24_positive, width, label='CD24+ Cells', yerr=cd24_positive_std)
rects2 = ax.bar(x + width / 2, cd24_negative, width, label='CD24- Cells', yerr=cd24_negative_std)

ax.set_ylabel(r'Fold Expression Compared to $\beta$-actin')
ax.set_title('Differential Gene Expression Induced by Retinoic Acid')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_yscale('log')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.legend()
plt.savefig('RAinduction_qPCR_result.tiff', format='tiff', dpi=720)

print(significance_list)
