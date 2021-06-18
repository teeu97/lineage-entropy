"""
DecisionTree_NanogSox2States.pu produces a decision tree that shows the distribution of lineages across different
states throughout 5 different timepoints.
"""
import pickle
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

__author__ = 'Tee Udomlumleart'
__maintainer__ = 'Tee Udomlumleart'
__email__ = ['teeu@mit.edu', 'salilg@mit.edu']
__status__ = 'Production'


def euclidean_distance(coor_1, coor_2):
    return math.sqrt(sum((i - j) ** 2 for i, j in zip(coor_1, coor_2)))


def vector_size(x_displacement, y_displacement):
    return math.sqrt(x_displacement ** 2 + y_displacement ** 2)

# initialize and normalize data
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

top_right_coord = (1, 1)
top_left_coord = (0, 1)
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
                       'vector_size': [], 'monte_ternary_coord': [], 'monte_cartesian_coord': [], 'monte_vector': [],
                       'timepoint_size': [], 'total_transition_amount': 0, 'total_size': 0}

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
            barcode_summary['assigned_state'].append(dist.index(min(dist)))

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


p = ProbabilityTree()  # initialize a ProbabilityTree object
p.update(all_barcode_list)  # populate the whole tree

# a dict where each key represent a timepoint and each value has probability of each transition
probability_dict = {i: [] for i in range(6)}
# a dict where each key represent a timepoint and each value has number of lineage of each transition
number_dict = {i: [] for i in range(6)}


def generate_probability_igraph(p):
    # full those two dicts above for making igraph
    timepoint = p.get_timepoint()
    probability_dict[timepoint] += p.get_probability()
    number_dict[timepoint] += p.get_children_lineage_number()
    if timepoint < 4:
        for child in p.get_children():
            generate_probability_igraph(child)


generate_probability_igraph(p)

# unflatten the value lists for the igraph format
prob_list = list(probability_dict.values())
prob_list_ready = []
number_list = list(number_dict.values())
number_list_ready = []
for item in prob_list:
    prob_list_ready += item
prob_list_ready = list(np.round(prob_list_ready, 2))
for item in number_list:
    number_list_ready += item

# probability - number list for each transition
prob_number_list_ready = [(x, y) for x, y in zip(prob_list_ready, number_list_ready)]

G = nx.balanced_tree(3, 5)  # create a balance tree that has 5 layers and each node has 3 outgoing edges
# set colors
node_color_list = ['black'] + ['#FFA6B3', '#80D4FF', '#C3DF86'] * (sum(3**i for i in range(0, 5)))
edge_color_list = ['#FFA6B3', '#80D4FF', '#C3DF86'] * (sum(3**i for i in range(0, 5)))

# set font size for the edges
edge_labels_large = {edge: prob_num for edge, prob_num in zip(list(G.edges)[:120], prob_number_list_ready[:120])}
edge_labels_small = {edge: prob_num for edge, prob_num in zip(list(G.edges)[120:], prob_number_list_ready[120:])}

# set the edge weights
weights = np.array([prob[0] for prob in prob_number_list_ready])*5+1

# plot the figure
pos = graphviz_layout(G, prog="twopi")
plt.figure(figsize=(20, 20))
nx.draw(G, pos, alpha=1, node_size=200, node_color=color_list, with_labels=False, width=weights, edge_color=edge_color_list)
nx.draw_networkx_edge_labels(G, pos, edge_labels_large, font_size=15, label_pos=0.45, bbox=dict(facecolor='white', edgecolor='none', pad=0.0))
nx.draw_networkx_edge_labels(G, pos, edge_labels_small, font_size=10, label_pos=0.35, bbox=dict(facecolor='white', edgecolor='none', pad=0.0))
plt.axis("equal")
plt.savefig('Decision_Tree_Networkx.svg', format='svg', dpi=720)