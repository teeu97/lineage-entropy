"""
FisherAnalysis_ChangingThreshold.py perform Fisher Exact Test on different motifs defined by different thresholds.
"""

import pickle
import numpy as np
import copy
import statsmodels.stats.multitest
import pandas as pd
import rpy2.robjects.numpy2ri
import csv
from rpy2.robjects.packages import importr


__author__ = 'Tee Udomlumleart'
__maintainer__ = 'Tee Udomlumleart'
__email__ = ['teeu@mit.edu', 'salilg@mit.edu']
__status__ = 'Production'


def euclidean_distance(coor_1, coor_2):
    return math.sqrt(sum((i - j) ** 2 for i, j in zip(coor_1, coor_2)))


def vector_size(x_displacement, y_displacement):
    return math.sqrt(x_displacement ** 2 + y_displacement ** 2)

# import Fisher exact test module from R
rpy2.robjects.numpy2ri.activate()
stats_ = importr('stats')


# Define a data structure that can keep track of lineage history
# It is a recursive tree structure: node keeps lineages in one timepoint; edge represents different states and point
# to the node on the next timepoint
class ProbabilityTree():
    def __init__(self, initial_state=None, timepoint=0):
        # initialize
        self.timepoint = timepoint
        self.total = 0

        if initial_state is None:  # if this is a starting node, there is no state history
            self.state_history = []
        else:  # if not, keep the input state history
            self.state_history = initial_state

        if self.timepoint < 5:  # if this is not a terminal node, run a recursive program
            self.states = [ProbabilityTree(self.get_state_history() + [i + 1], self.get_timepoint() + 1) for i in
                           range(3)]  # self.states represents 3 branches that point to 3 different states
            self.barcodes = [[] for i in range(3)]  # keep the barcodes that are assigned to three different states
            self.probabilities = [0 for i in range(3)]  # keep the fraction of barcodes that are assigned to states
            self.children_lineage_numbers = [0 for i in range(3)]  # keep the number of barcodes that are assigned
        else:  # if this is a terminal node, there is nothing to do - base case
            self.states = None
            self.barcodes = None
            self.probabilities = None
            self.children_lineage_numbers = None

    def get_children_lineage_number(self):
        return self.children_lineage_numbers

    def get_total(self):
        return self.total

    def get_state_history(self):
        return self.state_history

    def get_probability(self):
        return self.probabilities

    def get_timepoint(self):
        return self.timepoint

    def get_children(self):
        return self.states

    def traverse_tree(self, state_history):
        """
        this function traverses a tree when is given a state history and point to a node
        :arg state_history (list) - a list of states (int)
        """
        # if there is no state_history, just return the node
        if not state_history:
            return self
        # if there is a state history, run down the state history and return the node
        return self.get_children()[state_history[0] - 1].traverse_tree(state_history[1:])

    def retrieve_conditional_probability(self, state_history):
        """
        this function returns a conditional probability at the last transition of the state history
        :arg state_history (list) - a list of states (int)
        """
        # if it is at the transition of interest
        if len(state_history) == 1:
            return self.get_probability()[state_history[0] - 1]
        # if it is nor, then run this function recursively
        return self.get_children()[state_history[0] - 1].retrieve_conditional_probability(state_history[1:])

    def retrieve_all_probability(self, state_history, so_far=None):
        """
        this function computes a total probability given a state history
        :arg state_history (list) - a list of states (int)
        :arg so_far (float) - the total probability up to that transition
        """
        if so_far is None:
            # initialize the variable = 1
            so_far = 1
        if len(state_history) == 1:
            # base case: return the product of all probabilities
            return so_far * self.get_probability()[state_history[0] - 1]
        # iterative case: run a function while passing the update so_far variable
        so_far = so_far * self.get_probability()[state_history[0] - 1]
        return self.get_children()[state_history[0] - 1].retrieve_all_probability(state_history[1:], so_far)

    def update(self, barcode_list):
        """
        this function populates the decision tree
        :arg barcode_list (list) - a list of barcodes
        """
        self.total = len(barcode_list)
        if self.timepoint < 5:
            for barcode in barcode_list:
                # assign barcodes to different list based on their assigned states
                self.barcodes[barcode['assigned_state'][self.timepoint]].append(barcode)
            for index, state in enumerate(self.get_children()):
                state.update(self.barcodes[index]) # iteratively fill out the decision tree
                if self.get_total():  # if there are some lineages, calculate the probabilities
                    self.probabilities[index] = state.get_total() / self.get_total()
                    self.children_lineage_numbers[index] = state.get_total()
                else:  # if not, set the probability to 0
                    self.probabilities[index] = 0

# normalize data
total_cell_number = 10 ** 8

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

states = ['s1', 's2', 's3']
states_coords = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

timepoints = ['d0', 'd6', 'd12', 'd18', 'd24']

top_right_coord = (1, 1)
top_left_coord = (0, 1)
bottom_left_coord = (0, 0)

triangle_vertices = np.array([top_right_coord, top_left_coord, bottom_left_coord])
all_motif_pair_list = []
all_motif_dict_list = []

# iterate through each threshold
for iteration in range(11):
    all_size_set = set()
    all_vector_size_set = set()
    all_barcode_list = []

    for barcode, row in true_number_table.iterrows():
        barcode_dict = {}

        barcode_dict['d0_all'], barcode_dict['d0_s1'], barcode_dict['d0_s2'], barcode_dict['d0_s3'] = row[0:4]
        barcode_dict['d6_all'], barcode_dict['d6_s1'], barcode_dict['d6_s2'], barcode_dict['d6_s3'] = row[4:8]
        barcode_dict['d12_all'], barcode_dict['d12_s1'], barcode_dict['d12_s2'], barcode_dict['d12_s3'] = row[20:24]
        barcode_dict['d18_all'], barcode_dict['d18_s1'], barcode_dict['d18_s2'], barcode_dict['d18_s3'] = row[32:36]
        barcode_dict['d24_all'], barcode_dict['d24_s1'], barcode_dict['d24_s2'], barcode_dict['d24_s3'] = row[36:40]

        barcode_summary = {'ternary_coord': [], 'cartesian_coord': [], 'vector': [], 'size': [], 'assigned_state': [],
                           'vector_size': [], 'monte_ternary_coord': [], 'monte_cartesian_coord': [],
                           'monte_vector': [],
                           'timepoint_size': [], 'total_transition_amount': 0, 'total_size': 0, 'id': barcode}

        barcode_size = [sum(row[1:4]), sum(row[5:8]), sum(row[21:24]), sum(row[33:36]), sum(row[37:40])]

        for timepoint in timepoints:
            timepoint_all_present = all(barcode_size)
            timepoint_total = sum([barcode_dict[timepoint + '_' + state] for state in states])
            if timepoint_total:
                timepoint_size = []
                ternary_coord = []
                cell_number = []
                dist = []
                for state in states:
                    timepoint_size.append(barcode_dict[timepoint + '_' + state])
                    ternary_coord.append(barcode_dict[timepoint + '_' + state] / timepoint_total)
                barcode_summary['timepoint_size'].append(timepoint_size)
                barcode_summary['ternary_coord'].append(ternary_coord)

                cartesian_coord = np.dot(np.array(ternary_coord), triangle_vertices)
                barcode_summary['cartesian_coord'].append(list(cartesian_coord))

                for state_coord in triangle_vertices:
                    dist.append(euclidean_distance(cartesian_coord, state_coord))

                # last iteration - check with plurality vote
                if iteration == 10:
                    barcode_summary['assigned_state'].append(dist.index(min(dist)))
                else:  # threshold for S1 = iteration * 0.1, S2 = 0.5*(1 - S1) and S3 = 0.5*(1 - S1)
                    if np.argmax(ternary_coord) == 0:
                        if ternary_coord[0] > iteration * 0.1:
                            barcode_summary['assigned_state'].append(0)
                        else:
                            barcode_summary['assigned_state'].append(np.argmax(ternary_coord[1:]) + 1)
                    else:
                        barcode_summary['assigned_state'].append(np.argmax(ternary_coord[1:]) + 1)

                barcode_summary['size'].append(timepoint_total)

        if len(barcode_summary['cartesian_coord']) == 5:
            for i in range(4):
                barcode_summary['vector'].append((barcode_summary['cartesian_coord'][i + 1][0] -
                                                  barcode_summary['cartesian_coord'][i][0],
                                                  barcode_summary['cartesian_coord'][i + 1][1] -
                                                  barcode_summary['cartesian_coord'][i][1]))
                barcode_summary['vector_size'].append(
                    vector_size(barcode_summary['vector'][i][0], barcode_summary['vector'][i][1]))
            for size in barcode_summary['size']:
                all_size_set.add(round(size, 3))
                barcode_summary['total_size'] += size
            for size_ in barcode_summary['vector_size']:
                all_vector_size_set.add(round(size_, 3))
            all_barcode_list.append(barcode_summary)

    # populate the probability tree data structure
    p = ProbabilityTree()
    p.update(all_barcode_list)

    motif_pattern = {}

    def generate_probability_matrix(p, info=None):
        timepoint = p.get_timepoint()
        if info == None:
            info = {'Conditional Probability': [], 'Total Probability': 1, 'Number of Lineages': 0}

        if timepoint < 5:
            for index, child in enumerate(p.get_children()):
                info_copy = copy.deepcopy(info)
                info['Total Probability'] *= p.get_probability()[index]
                info['Conditional Probability'].append(p.get_probability()[index])
                generate_probability_matrix(child, info)
                info = info_copy
        else:
            info['Number of Lineages'] = p.get_total()
            motif_pattern[tuple(p.get_state_history())] = info


    def generate_dataframe(motif_matrix):
        pd_dict = {'State History': [], 'Conditional Probability': [], 'Total Probability': [],
                   'Number of Lineages': []}
        for pattern in motif_matrix:
            pd_dict['State History'].append(pattern)
            for property in motif_matrix[pattern]:
                pd_dict[property].append(motif_matrix[pattern][property])
        return pd_dict


    all_motif_dict = {}

    for i in range(1, 4):
        for j in range(1, 4):
            for k in range(1, 4):
                for l in range(1, 4):
                    lineage_list = []
                    for state in range(3):
                        barcodes = p.traverse_tree([i, j, k, l]).get_barcodes()
                        if p.traverse_tree([i, j, k, l]).get_barcodes():
                            lineage_list.extend(barcode['id'] for barcode in barcodes[state])
                    all_motif_dict[(i, j, k, l)] = lineage_list
    all_motif_dict_list.append(all_motif_dict)

    generate_probability_matrix(p)
    pd_dict = generate_dataframe(motif_pattern)
    df = pd.DataFrame.from_dict(pd_dict)

    contingency_table = [[] for state in range(3)]
    state_history_table = [[] for state in range(3)]

    for state in range(3):
        new_table = [0, 0, 0]
        i = 0
        for row, motif in df.iterrows():
            if motif['State History'][3] == state + 1:
                new_table[motif['State History'][4] - 1] += motif['Number of Lineages']
                i += 1
            if motif['State History'][4] == 3 and i == 3:
                if any(new_table):
                    contingency_table[state].append(new_table)
                    state_history_table[state].append(motif['State History'][:4])

                new_table = [0, 0, 0]
                i = 0

    with open('StateHistory_FisherExactTest_{}.csv'.format(iteration), mode='w') as file:
        writer_ = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        test_number = sum(len(x) * (len(x) - 1) / 2 for x in contingency_table)
        corrected_alpha = 0.05
        rejected_number = 0

        p_value_list = []
        no_test_index_list = []

        index = 0
        for state in range(3):
            for ix in range(len(contingency_table[state]) - 1):
                for iy in range(ix + 1, len(contingency_table[state])):
                    matrix = pd.DataFrame([contingency_table[state][ix], contingency_table[state][iy]])
                    matrix = matrix.loc[:, (matrix != 0).any(axis=0)]

                    if matrix.shape[0] > 1 and matrix.shape[1] > 1:
                        res = stats_.fisher_test(np.array(matrix))
                        p_value_list.append(res[0][0])
                    else:
                        no_test_index_list.append(index)

                    index += 1

        significance_list, corrected_p_value_list = statsmodels.stats.multitest.fdrcorrection(p_value_list)

        rejected_number = list(significance_list).count(True)

        test_index = 0
        p_value_index = 0
        motif_pair_set = set()
        for state in range(3):
            for ix in range(len(contingency_table[state]) - 1):
                for iy in range(ix + 1, len(contingency_table[state])):
                    if test_index not in no_test_index_list:
                        if significance_list[p_value_index]:
                            motif_pair_set.add(
                                (tuple(state_history_table[state][ix]), tuple(state_history_table[state][iy])))
                        writer_.writerow(['State History', 'State 1', 'State 2', 'State 3'])
                        writer_.writerow(
                            [state_history_table[state][ix]] + [member for member in contingency_table[state][ix]])
                        writer_.writerow(
                            [state_history_table[state][iy]] + [member for member in contingency_table[state][iy]])
                        writer_.writerow(['corrected p-value', corrected_p_value_list[p_value_index]])
                        writer_.writerow(['Reject null hypothesis', significance_list[p_value_index]])
                        writer_.writerow([])
                        p_value_index += 1
                    test_index += 1
        all_motif_pair_list.append(motif_pair_set)
    print(iteration, rejected_number, test_number)

for i in range(len(all_motif_pair_list)):
    for j in range(i, len(all_motif_pair_list)):
        shared_motif = list(all_motif_pair_list[i] & all_motif_pair_list[j])
        print(i, j, len(shared_motif))
        first_barcodes_group = set()
        second_barcodes_group = set()
        for motif in shared_motif:
            first_barcodes_group |= set(all_motif_dict_list[i][motif[0]])
            first_barcodes_group |= set(all_motif_dict_list[i][motif[1]])
            second_barcodes_group |= set(all_motif_dict_list[j][motif[0]])
            second_barcodes_group |= set(all_motif_dict_list[j][motif[1]])
        print(len(set(first_barcodes_group & second_barcodes_group)))
    print('\n')