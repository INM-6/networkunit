import sciunit
import networkx as nx
from networkunit.tests.test_correlation_matrix_test import correlation_matrix_test
from networkunit.tests.test_two_sample_test import two_sample_test
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
            generate_prediction(model, **kwargs)
        self.prediction_dim = 2
        if 'graph_measure' in self.params \
                and self.params['graph_measure'] is not None:
            if hasattr(self, 'graph') and 'graph_{}'.format(model.name) in self.graph:
                G = self.graph['graph_{}'.format(model.name)]
            else:
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
                graph_list = [(i,j,w) for i,j,w in
                              zip(triu_idx[0],triu_idx[1],weight_list) if w]
                G = nx.Graph()
                G.add_weighted_edges_from(graph_list)
                if not hasattr(self, 'graph'):
                    self.graph = {}
                self.graph['graph_{}'.format(model.name)] = G

            if self.params['graph_measure'] == 'degree strength':
                degrees = nx.degree(G, weight='weight')
                self.prediction_dim = 1
                return np.array([d[1] for d in degrees])

            if self.params['graph_measure'] == 'closeness':
                weight_dict = nx.get_edge_attributes(G, 'weight')
                weight_sum = float(np.sum(np.array([weight_dict[edge]
                                                    for edge in weight_dict.keys()])))
                for edge in G.edges:
                    G.edges[edge].update(distance = G.edges[edge]['weight'] / weight_sum)
                closeness = nx.closeness_centrality(G, distance='distance')
                self.prediction_dim = 1
                return np.array([closeness[i] for i in closeness.keys()])

            if self.params['graph_measure'] == 'betweenness':
                betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True)
                self.prediction_dim = 1
                return np.array([betweenness[i] for i in betweenness.keys()])

            if self.params['graph_measure'] == 'edge betweenness':
                edge_betweenness = nx.edge_betweenness_centrality(G, weight='weight', normalized=True)
                N = max(G.nodes) + 1
                betweenness_matrix = np.zeros((N, N))
                for edge in edge_betweenness:
                    betweenness_matrix[edge] = edge_betweenness[edge]
                    betweenness_matrix[edge[1],edge[0]] = edge_betweenness[edge]
                return betweenness_matrix

            if self.params['graph_measure'] == 'katz':
                katz = nx.katz_centrality(G, weight='weight', normalized=True)
                self.prediction_dim = 1
                return np.array([katz[i] for i in katz.keys()])

            else:
                raise KeyError, 'Graph measure not know!'
        else:
            return matrix


    def visualize_sample(self, model1=None, model2=None, ax=None,
                         palette=None, remove_autocorr=True,
                         sample_names=['observation', 'prediction'],
                         var_name='Measured Parameter', sort=False, **kwargs):

        if hasattr(self, 'prediction_dim') and self.prediction_dim == 1:
            if ax is None:
                fig, new_ax = plt.subplots()
            else:
                new_ax = ax
            samples, palette = super(graph_centrality_helperclass, self)._create_plotting_samples(
                                                          model1=model1,
                                                          model2=model2,
                                                          palette=palette)
            if sort:
                samples[0] = np.sort(samples[0])[::-1]
                samples[1] = np.sort(samples[1])[::-1]
                nodes1 = np.arange(len(samples[0]))
                nodes2 = np.arange(len(samples[1]))
            else:
                nodes1 = self.graph['graph_{}'.format(model1.name)].nodes
                nodes2 = self.graph['graph_{}'.format(model2.name)].nodes

            N = len(samples[0])

            new_ax.bar(nodes1, samples[0], color=palette[0], label=model1.name)
            new_ax.bar(nodes2, -1*samples[1], color=palette[1], label=model2.name)
            new_ax.set_title(self.params['graph_measure']
                             + '{}'.format(' sorted' if sort else ''))
            new_ax.set_xlabel('Neurons')
            plt.legend()

        else:
            super(graph_centrality_helperclass,self).visualize_sample(
                                                           model1=model1,
                                                           model2=model2,
                                                           ax=ax,
                                                           palette=palette,
                                                           remove_autocorr=remove_autocorr,
                                                           sample_names=sample_names,
                                                           var_name=var_name,
                                                           **kwargs)


