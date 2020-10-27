import pickle
import pandas as pd
import datetime
from SequenceDecomplexationOptimized import Functions

reference_table = pd.DataFrame(pickle.load(open('190812_finished_table.pickle', 'rb'))).copy(deep=True)
additional_table = pd.DataFrame(pickle.load(open('191003Gar.fastq_raw_read_correct.pickle', 'rb')))[:-1].transpose()

sum_additional_table = additional_table.sum(axis=0)
len_additional_table = additional_table.shape[0] 
for barcode, value in sum_additional_table.iteritems():
     if value < 10:
        additional_table[barcode] = [0 for i in range(len_additional_table)]

for additional_barcode, additional_row in additional_table.iterrows():
    added = False
    for reference_barcode, reference_row in reference_table.iterrows():
        if Functions.hamming_distance(additional_barcode, reference_barcode) <= 5:
            reference_table.loc[reference_barcode] += additional_row
            added = True
            break
    if not added:
        reference_table.append(additional_table.loc[additional_barcode])

print('dumping pickle')
now = datetime.datetime.now()
with open('191012_finished_table.pickle', 'wb') as handle:
    pickle.dump(reference_table, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('saving csv file')
reference_table.to_excel('191012_finished_table.xlsx')
