import sciunit
import networkx as nx
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
from quantities import ms
import numpy as np
from copy import copy


class graph_centrality_helperclass(sciunit.Test):
    """
    Abstract test class to be combined with a test which generates a prediction
    in form of matrix. From this matrix the chosen graph measure is calculated
    for each node and passed on in matrix form.
    The executable test has to inherit from the graph_measure_test and the
    matrix generating test.
    """
    __metaclass__ = ABCMeta

    def generate_prediction(self, model, **kwargs):
        matrix = super(graph_centrality_helperclass, self).\
            generate_prediction(model)
        weight_matrix = copy(matrix)
        if 'edge_threshold' in self.params:
            edge_threshold = self.params['edge_threshold']
        else:
            edge_threshold = 0.
        non_edges = np.where(weight_matrix <= edge_threshold)
        weight_matrix[non_edges[0], non_edges[1]] = 0.
        np.fill_diagonal(weight_matrix, 0)
        N = len(matrix)
        triu_idx = np.triu_indices(N, 1)
        weight_list = weight_matrix[triu_idx[0],triu_idx[1]]
        graph_list = [(i,j,w) for i,j,w in zip(triu_idx[0],triu_idx[1],weight_list)
                      if w]
        G = nx.Graph()
        G.add_weighted_edges_from(graph_list)

        if 'graph measure' in self.params \
                          and self.params['graph measure'] is not None:
            if self.params['graph measure'] == 'degree strength':
                degrees = nx.degree(G, weight='weight')
                return np.array([d[1] for d in degrees])

            if self.params['graph measure'] == 'closeness':
                weight_sum = float(np.sum(weight_list))
                for edge in G.edges:
                    G.edges[edge].update(distance = G.edges[edge]['weight'] / weight_sum)
                closeness = nx.closeness_centrality(G, distance='distance')
                return np.array([closeness[i] for i in closeness.keys()])

            if self.params['graph measure'] == 'betweenness':
                betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True)
                return np.array([betweenness[i] for i in range(len(betweenness))])

            if self.params['graph measure'] == 'edge betweenness':
                edge_betweenness = nx.edge_betweenness_centrality(G, weight='weight', normalized=True)
                betweenness_matrix = np.zeros((N, N))
                for edge in edge_betweenness:
                    betweenness_matrix[edge] = edge_betweenness[edge]
                    betweenness_matrix[edge[1],edge[0]] = edge_betweenness[edge]
                return betweenness_matrix

            if self.params['graph measure'] == 'katz':
                katz = nx.katz_centrality(G, weight='weight', normalized=True)
                return np.array([katz[i] for i in range(len(katz))])





