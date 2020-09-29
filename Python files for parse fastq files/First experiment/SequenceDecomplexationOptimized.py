"""
SequenceDecomplexation analyzes reads in a fastq file, decomplexes them and then save in a pickle-file form.

There are four main modules that this software is running:

1.Quality Control - this software checks the quality of reads according to their Phred score and
their similarities to constant regions based on our assigned Hamming distance threshold.

2.Barcode Collapse - this software combines reads with very few counts, produced by PCR- or NGS-based error, with the ones
with higher abundance. This software assumes reads with higher number of reads are less likely to be created by errors
while ones with fewer are.

3.Count Normalization - this software calculates total number of counts then produces normalized counts for barcodes with
distinct sample indexes.

4.Data Structure Reorganization - this software organizes collapsed reads into a grid-like data structure called
contingency table which makes the downstream application less complicated.

Example:

    Barcode: No.1

                Day 0      Day 6 (ctrl)   ...
    State 1 |           |               |
    State 2 |           |               |
    State 3 |           |               |
    All     |           |               |

Each barcode contains 40 cells; each cell represent number of normalized counts of that barcode in particular conditions.
"""
import Constants
import pickle
import csv
import datetime
import sys

from Bio import SeqIO

__author__ = 'Tee Udomlumleart'
__maintainer__ = 'Tee Udomlumleart'
__email__ = ['teeu@mit.edu', 'salilg@mit.edu']
__status__ = 'Production'

class Reference:
    """
    Reference object is a superclass of reference barcodes that are used to guide barcode decomplexation
    in each sample.
    """
    pass

class AllBarcode(Reference):
    """
    AllBarcode object contains a dictionary that maps a tuple (barcode, sample index) to a corresponding Reads object.
    """
    def __init__(self):
        self.list = {}
        self.sample_index_total_count = {sample_index: 0 for sample_index in list(Constants.sample_index_dict.values())}

    def __len__(self):
        return len(self.list)

    def __getitem__(self, item):
        """
        This function accepts a tuple (barcode, sample index) and return a Reads object, if such object exist.

        :param item: a tuple (barcode, sample index)
        :return: a corresponding Reads object
        """
        return self.list[item[0]][item[1]]

    def add_read(self, id_tuple):
        """
        This function adds a Reads object to self.list if such object has not been already added.

        Otherwise, this function adds one more count to that read.

        :param id_tuple: a tuple (barcode, sample index) used to identify Reads object
        """
        new_barcode, new_sample_index = id_tuple
        added_flag = False
        for parsed_barcode in self.list:
            if Functions.hamming_distance(parsed_barcode, new_barcode) <= 5:
                self.list[parsed_barcode][new_sample_index] += 1
                self.sample_index_total_count[new_sample_index] += 1
                added_flag = True
                break
        if not added_flag:
            self.list[new_barcode] = {sample_index: 0 for sample_index in list(Constants.sample_index_dict.values())}
            self.list[new_barcode][new_sample_index] += 1
            self.sample_index_total_count[new_sample_index] += 1


    def print_raw_read(self):
        """
        This function print read counts and print them via a CSV file.
        """
        with open(str(file)+'_raw_read_correct'+'.csv', mode='w') as csv_file:
            fieldnames = ['Barcodes'] + list(Constants.sample_index_dict.values())
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for parsed_barcode in self.list:
                for sample_index in self.list[parsed_barcode]:
                    self.list[parsed_barcode][sample_index] = self.list[parsed_barcode][sample_index]
                self.list[parsed_barcode]['Barcodes'] = parsed_barcode
                writer.writerow(self.list[parsed_barcode])

    def save_pickle(self):
        """
        This function saves this data structure ({barcodes: {sample index: count}}) in a pickle form that is ready to be
        worked with in the future
        """
        with open(str(file)+'_raw_read_correct'+'.pickle', 'wb') as handle:
            pickle.dump(self.list, handle, protocol=pickle.HIGHEST_PROTOCOL)



class Functions:
    """
    Functions object contains miscellaneous functions that are important for analyzing reads from a fastq file.
    """
    @staticmethod
    def hamming_distance(sequence_1, sequence_2):
        """
        This function evaluates Hamming distance between two sequences. These two sequences must have an equal length.

        :param sequence_1 (str object)
        :param sequence_2 (str object)
        :return: # of differences (int object)
        """
        assert (len(sequence_1) == len(sequence_2)), "Two sequences must have equal length!"
        return sum(alphabet1 != alphabet2 for alphabet1, alphabet2 in zip(sequence_1, sequence_2))

    @staticmethod
    def reading_fastq():
        """
        This function automatically checks the quality of a particular read by checking similarities
        between its constant regions and a priori constant sequences.

        The first constant region must have less than or equal to 5 Hamming distances away from the reference and
        the second constant region must have less or equal to 2 Hamming distances.

        Moreover, all bases must have Phred score greater or equal to 20 – implying that the confidence of peak calling
        is greater than or equal to 99%.

        If a particular read violates at least one of these requirements, it will be disregard.

        :param input_fastq: the name of fastq file containing reads we want to analyze.
        :param output_csv_name: the name of csv file containing
        1.total number of reads analyzed
        2.total number of reads whose min(Phred score) < 20
        3.total number of reads whose constant regions differ more than the threshold
        4.total number of reads whose sample index differ more than the threshold
        5.total number of reads that pass all quality metrics
        6.total number of sample indexes present in this population
        """
        parsed_generator = SeqIO.parse(open(file), 'fastq')
        sample_index_list = list(Constants.sample_index_dict.values())
        all_barcode_list = AllBarcode()

        all_reads = 0
        good_reads = 0
        bad_barcode_reads = 0
        bad_constant_reads = 0
        bad_sample_index_reads = 0
        i = 1

        for seq_record in parsed_generator:
            all_reads += 1
            if sum(seq_record.letter_annotations['phred_quality'][:30]) < 0.8 * 40 * 30:
                bad_barcode_reads += 1
                continue
            read_sequence = seq_record.seq[:129]
            if Functions.hamming_distance(Constants.constant_1, read_sequence[30:95]) > 5 or \
                    Functions.hamming_distance(Constants.constant_2, read_sequence[105:]) > 2:
                bad_constant_reads += 1
                continue
            read_barcode = str(read_sequence[:30])
            read_sample_index = str(read_sequence[95:105])
            sample_index_flag = True
            for sample_index in sample_index_list:
                if Functions.hamming_distance(sample_index, read_sample_index) <= 1:
                    read_sample_index = sample_index
                    sample_index_flag = False
            if sample_index_flag:
                bad_sample_index_reads += 1
                continue
            read_id = (read_barcode, read_sample_index)
            all_barcode_list.add_read(read_id)
            good_reads += 1
            if i % 1000000 == 0:
                print(str(i) + ' reads have been parsed at ' + str(datetime.datetime.now()))
            i += 1

        with open(str(file) + 'read_summary.csv', mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['Summary of ' + str(file)])
            csv_writer.writerow(['Total Number of Reads', str(all_reads)])
            csv_writer.writerow(['Number of Good Reads', str(good_reads)])
            csv_writer.writerow(['Number of Bad Barcode Reads', str(bad_barcode_reads)])
            csv_writer.writerow(['Number of Bad Constant Reads', str(bad_constant_reads)])
            csv_writer.writerow(['Number of Bad Sample Index Reads', str(bad_sample_index_reads)])

        return all_barcode_list

    @staticmethod
    def operate():
        print('parsing fastq and collapsing barcodes')
        all_barcode_list = Functions.reading_fastq()
        print('printing barcodes')
        all_barcode_list.print_raw_read()
        print('dumping pickles')
        all_barcode_list.save_pickle()

if __name__ == '__main__':
    file = sys.argv[1]
    Functions.operate()
