"""
SequenceDecomplexationTableExtra.py combines contingency tables produced by 190812_finished_table.pickle
and 191003Gar_finished_table.pickle together.

191003Gar file is the data from the second NGS run from this first experiment to improve read qualities in some of the
samples in 190812Gar file
"""

import pickle
import pandas as pd
import datetime
from SequenceDecomplexationOptimized import Functions

__author__ = 'Tee Udomlumleart'
__maintainer__ = 'Tee Udomlumleart'
__email__ = ['teeu@mit.edu', 'salilg@mit.edu']
__status__ = 'Production'

# arbitrary assign 190812 file as a reference table
reference_table = pd.DataFrame(pickle.load(open('190812_finished_table.pickle', 'rb'))).copy(deep=True)
additional_table = pd.DataFrame(pickle.load(open('191003Gar_finished_table.pickle', 'rb')))[:-1].transpose()

# only make sure that the reads that are considering have >= 10 total reads
sum_additional_table = additional_table.sum(axis=0)  # find total number of reads for each barcode
len_additional_table = additional_table.shape[0]
for barcode, value in sum_additional_table.iteritems():
     if value < 10:  # iterate through each barcde and make sure that the total number of reads >= 10
        additional_table[barcode] = [0 for i in range(len_additional_table)]  # if not, set all the reads = 10


# check if barcodes in 191003 file similar enough to ones from 190812 file
for additional_barcode, additional_row in additional_table.iterrows():
    added = False
    for reference_barcode, reference_row in reference_table.iterrows():
        # if they are similar enough (<= 5 Hamming distance errors), collapse them together
        if Functions.hamming_distance(additional_barcode, reference_barcode) <= 5:
            reference_table.loc[reference_barcode] += additional_row
            added = True
            break
    # if not, add them to a new row of the table
    if not added:
        reference_table.append(additional_table.loc[additional_barcode])

filename = ''  # add output filename here

print('dumping pickle')
with open(filename + '_finished_table.pickle', 'wb') as handle:
    pickle.dump(reference_table, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('saving csv file')
reference_table.to_excel(filename + '191012_finished_table.xlsx')
