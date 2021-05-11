import pickle
import math
import numpy as np
import matplotlib
import pandas as pd
import copy

font = {'family': 'normal',
        'weight': 'bold',
        'size': 16}

matplotlib.rc('font', **font)
matplotlib.rcParams['font.sans-serif'] = "Helvetica"
matplotlib.rcParams['font.family'] = "sans-serif"


def euclidean_distance(coor_1, coor_2):
    return math.sqrt(sum((i - j) ** 2 for i, j in zip(coor_1, coor_2)))


def vector_size(x_displacement, y_displacement):
    return math.sqrt(x_displacement ** 2 + y_displacement ** 2)


top_right_coord = (10, 10)
top_left_coord = (0, 10)
bottom_left_coord = (0, 0)

triangle_vertices = np.array([top_right_coord, top_left_coord, bottom_left_coord])

empirical_stochasticity = {i: [] for i in range(5)}

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

all_size_set = set()
all_vector_size_set = set()
all_barcode_list = []

top_right_coord = (10, 10)
top_left_coord = (0, 10)
bottom_left_coord = (0, 0)

triangle_vertices = np.array([top_right_coord, top_left_coord, bottom_left_coord])

for barcode, row in true_number_table.iterrows():
    barcode_dict = {}

    barcode_dict['d0_all'], barcode_dict['d0_s1'], barcode_dict['d0_s2'], barcode_dict['d0_s3'] = row[0:4]
    barcode_dict['d6_all'], barcode_dict['d6_s1'], barcode_dict['d6_s2'], barcode_dict['d6_s3'] = row[4:8]
    barcode_dict['d12_all'], barcode_dict['d12_s1'], barcode_dict['d12_s2'], barcode_dict['d12_s3'] = row[20:24]
    barcode_dict['d18_all'], barcode_dict['d18_s1'], barcode_dict['d18_s2'], barcode_dict['d18_s3'] = row[32:36]
    barcode_dict['d24_all'], barcode_dict['d24_s1'], barcode_dict['d24_s2'], barcode_dict['d24_s3'] = row[36:40]

    barcode_summary = {'ternary_coord': [], 'cartesian_coord': [], 'vector': [], 'size': [], 'assigned_state': [],
                       'transition_amount': [], 'monte_ternary_coord': [], 'monte_cartesian_coord': [], 'monte_vector': [],
                       'timepoint_size': [], 'total_transition_amount': 0, 'proportion_diff': [],
                       'assigned_motility_state': []}

    barcode_size = [sum(row[1:4]), sum(row[5:8]), sum(row[21:24]), sum(row[33:36]), sum(row[37:40])]

    for timepoint in timepoints:
        timepoint_all_present = all(barcode_size)
        timepoint_total = sum([barcode_dict[timepoint + '_' + state] for state in states])
        if timepoint_all_present:
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
            barcode_summary['assigned_state'].append(dist.index(min(dist)))

            barcode_summary['size'].append(timepoint_total)

    if len(barcode_summary['cartesian_coord']) == 5:
        for i in range(4):
            barcode_summary['vector'].append((barcode_summary['cartesian_coord'][i + 1][0] -
                                              barcode_summary['cartesian_coord'][i][0],
                                              barcode_summary['cartesian_coord'][i + 1][1] -
                                              barcode_summary['cartesian_coord'][i][1]))
            barcode_summary['proportion_diff'].append(tuple(
                np.array(barcode_summary['ternary_coord'][i + 1]) - np.array(barcode_summary['ternary_coord'][i])))
            barcode_summary['transition_amount'].append(
                vector_size(barcode_summary['vector'][i][0], barcode_summary['vector'][i][1]))

        for size in barcode_summary['size']:
            all_size_set.add(round(size, 3))
        for size_ in barcode_summary['transition_amount']:
            all_vector_size_set.add(round(size_, 3))
        all_barcode_list.append(barcode_summary)

for transition in range(4):
    all_barcode_list.sort(key=lambda barcode: barcode['transition_amount'][transition], reverse=True)
    for index, barcode in enumerate(all_barcode_list):
        if index <= int(0.2*len(all_barcode_list)):
            barcode['assigned_motility_state'].append(1)  # change state
        else:
            barcode['assigned_motility_state'].append(0)  # stay

class ProbabilityTree():
    def __init__(self, initial_state=None, timepoint=0):
        self.timepoint = timepoint
        self.total = 0

        if initial_state is None:
            self.state_history = []
        else:
            self.state_history = initial_state

        if self.timepoint < 4:
            self.states = [ProbabilityTree(self.get_state_history() + [i + 1], self.get_timepoint() + 1) for i in
                           range(2)]
            self.barcodes = [[] for i in range(2)]
            self.probabilities = [0 for i in range(2)]
            self.children_lineage_numbers = [0 for i in range(2)]
        else:
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

    def traverse_tree(self, list_):
        if not list_:
            return self
        return self.get_children()[list_[0] - 1].traverse_tree(list_[1:])

    def retrieve_conditional_probability(self, list_):
        if len(list_) == 1:
            return self.get_probability()[list_[0] - 1]
        return self.get_children()[list_[0] - 1].retrieve_conditional_probability(list_[1:])

    def retrieve_all_probability(self, list_, so_far=None):
        if so_far is None:
            so_far = 1
        if len(list_) == 1:
            return so_far * self.get_probability()[list_[0] - 1]
        so_far = so_far * self.get_probability()[list_[0] - 1]
        return self.get_children()[list_[0] - 1].retrieve_all_probability(list_[1:], so_far)

    def update(self, barcode_list):
        self.total = len(barcode_list)
        if self.timepoint < 4:
            for barcode in barcode_list:
                self.barcodes[barcode['assigned_motility_state'][self.timepoint]].append(barcode)
            for index, state in enumerate(self.get_children()):
                state.update(self.barcodes[index])
                if self.get_total():
                    self.probabilities[index] = state.get_total() / self.get_total()
                    self.children_lineage_numbers[index] = state.get_total()
                else:
                    self.probabilities[index] = 0

    def generate_tree_str(self):
        if self.get_timepoint() == 4:
            tree_str = ''.join([str(s) for s in self.get_state_history()])
            return tree_str
        else:
            tree_str = '('
            for index, child in enumerate(self.get_children()):
                tree_str += '{}: {}'.format(child.generate_tree_str(), round(self.get_probability()[index], 3))
                if index < 2:
                    tree_str += ','
            tree_str += ')'
            return tree_str

p = ProbabilityTree()
p.update(all_barcode_list)

motif_pattern = {}

def generate_probability_matrix(p, info=None):
    timepoint = p.get_timepoint()
    if info == None:
        info = {'Conditional Probability': [], 'Total Probability': 1, 'Number of Lineages': 0}

    if timepoint < 4:
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
    pd_dict = {'State History': [], 'Conditional Probability': [], 'Total Probability': [], 'Number of Lineages': []}
    for pattern in motif_matrix:
        pd_dict['State History'].append(pattern)
        for property in motif_matrix[pattern]:
            pd_dict[property].append(motif_matrix[pattern][property])
    return pd_dict

generate_probability_matrix(p)
pd_dict = generate_dataframe(motif_pattern)
df = pd.DataFrame.from_dict(pd_dict)
df.to_csv('Decision_Tree_MotilityState_Percentile_Matrix.csv')