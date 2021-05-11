"""
SequenceDecomplexationTable.py combines multiple contingency tables produced by parallelized
SequenceDecomplexationOptimized.py into one big contingency table, so it will be easier for the downstream analyses.

This software also collapses barcodes that are similar enough together to reduce the dimensionality of the data
"""

import pickle
import pandas as pd
import datetime
import os
from SequenceDecomplexationOptimized import Functions

__author__ = 'Tee Udomlumleart'
__maintainer__ = 'Tee Udomlumleart'
__email__ = ['teeu@mit.edu', 'salilg@mit.edu']
__status__ = 'Production'

# arbitrary define this table as a reference table
first_table = pd.DataFrame(pickle.load(open('group_1.fastq_raw_read_correct.pickle', 'rb')))[:-1].transpose()

cwd = os.getcwd()  # get all files in the current working directory
for filename in os.listdir(cwd):
    # only choose files that end with _correct.pickle (produced by SequenceDecomplexationOptimized.py)
    if filename.endswith('_correct.pickle'):
        # ignore the first file since it is already assigned as a reference table
        if filename == 'group_1.fastq_raw_read_correct.pickle':
            continue
        print('analyzing ' + filename + ' at ' + str(datetime.datetime.now()))

        other_table = pd.DataFrame(pickle.load(open(filename, 'rb')))[:-1].transpose()
        i = 0

        # iterate through barcodes from another table
        for other_barcode, other_row in other_table.iterrows():
            i += 1
            added = False
            # then iterate through barcodes from the reference table
            for reference_barcode, reference_row in first_table.iterrows():
                # and check if the errors between those two barcodes <= 5
                if Functions.hamming_distance(other_barcode, reference_barcode) <= 5:
                    # if they are similar enough, assign them together
                    first_table.loc[reference_barcode] += other_row
                    added = True
                    break  # then move on to other barcodes from non-reference table

            # if this barcode from the non-reference table is unique, just add that entry to the reference table
            if not added:
                first_table.append(other_table.loc[other_barcode])

filename = ''  # add filename here

print('dumping pickle')
with open(filename + '_finished_table.pickle', 'wb') as handle:
    pickle.dump(first_table, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('saving csv file')
first_table.to_excel(filename + '_finished_table.xlsx')



