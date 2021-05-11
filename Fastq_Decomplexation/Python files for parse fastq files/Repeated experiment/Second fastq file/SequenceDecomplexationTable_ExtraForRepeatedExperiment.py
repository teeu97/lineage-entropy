import pickle
import pandas as pd
import datetime
import os
from SequenceDecomplexationOptimized import Functions

first_table = pd.DataFrame(pickle.load(open('reference.pickle', 'rb')))[:-1]

for col in first_table.columns:
    first_table[col].values[:] = 0

cwd = os.getcwd()
for filename in os.listdir(cwd):
    if filename.endswith('_correct.pickle'):
        print('analyzing ' + filename + ' at ' + str(datetime.datetime.now()))
        other_table = pd.DataFrame(pickle.load(open(filename, 'rb')))[:-1].transpose()
        i = 0
        for other_barcode, other_row in other_table.iterrows():
            i += 1
            added = False
            for reference_barcode, reference_row in first_table.iterrows():
                if Functions.hamming_distance(other_barcode, reference_barcode) <= 5:
                    first_table.loc[reference_barcode] += other_row
                    added = True
                    break
            if not added:
                first_table.append(other_table.loc[other_barcode])

print('dumping pickle')
now = datetime.datetime.now()
with open('{}_finished_table.pickle'.format(now.strftime('%Y%m%d')), 'wb') as handle:
    pickle.dump(first_table, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('saving csv file')
first_table.to_excel('{}_finished_table.xlsx'.format(now.strftime('%Y%m%d')))



