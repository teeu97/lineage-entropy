import pickle
import pandas as pd
import datetime
import os
from SequenceDecomplexationOptimized import Functions

first_table = pd.DataFrame(pickle.load(open('group_1.fastq_raw_read_correct.pickle', 'rb')))[:-1].transpose()

cwd = os.getcwd()
for filename in os.listdir(cwd):
    if filename.endswith('_correct.pickle'):
        if filename == 'group_1.fastq_raw_read_correct.pickle':
            continue
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
with open('190812_finished_table.pickle', 'wb') as handle:
    pickle.dump(first_table, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('saving csv file')
first_table.to_excel('190812_finished_table.xlsx')



