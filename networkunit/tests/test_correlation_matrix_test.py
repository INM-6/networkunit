from networkunit.tests.test_correlation_test import correlation_test
from networkunit.capabilities.cap_ProducesSpikeTrains import ProducesSpikeTrains
from networkunit.plots.plot_correlation_matrix import plot_correlation_matrix
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from copy import copy
from abc import ABCMeta, abstractmethod


class correlation_matrix_test(correlation_test):
    """
    Test to compare the pairwise correlations of a set of neurons in a network.
    """
    __metaclass__ = ABCMeta

    required_capabilities = (ProducesSpikeTrains, )

    def generate_prediction(self, model, **kwargs):
        # call the function of the required capability of the model
        # and pass the parameters of the test class instance in case the
        if kwargs:
            self.params.update(kwargs)
        # check if model has already stored prediction!
        spiketrains = model.produce_spiketrains(**self.params)
        matrix = self.generate_cc_matrix(spiketrains=spiketrains,
                                         **self.params)
        return matrix

    def visualize_sample(self, model1=None, model2=None, ax=None,
                         palette=None, remove_autocorr=True,
                         sample_names=['observation', 'prediction'],
                         var_name='Measured Parameter', **kwargs):

        matrices, palette = self._create_plotting_samples(model1=model1,
                                                          model2=model2,
                                                          palette=palette)
        fig, ax = plt.subplots(nrows=1, ncols=2)
        plot_correlation_matrix(matrices[0], ax=ax[0], remove_autocorr=remove_autocorr,
                                labels=None, sort=False, cluster=False,
                                linkmethod='ward', dendrogram_args={},
                                **kwargs)
        ax[0].set_title(sample_names[0])
        plot_correlation_matrix(matrices[1], ax=ax[1], remove_autocorr=remove_autocorr,
                                labels=None, sort=False, cluster=False,
                                linkmethod='ward', dendrogram_args={},
                                **kwargs)
        ax[1].set_title(sample_names[1])
        return ax

    def draw_graph(self, model, **kwargs):
        spiketrains = model.produce_spiketrains(**self.params)
        matrix = self.generate_cc_matrix(spiketrains=spiketrains, **self.params)
        weight_matrix = copy(matrix)
        if 'edge_threshold' in self.params:
            edge_threshold = self.params['edge_threshold']
        else:
            edge_threshold = 0.
        non_edges = np.where(weight_matrix <= edge_threshold)
        if len(non_edges[0]):
            weight_matrix[non_edges[0], non_edges[1]] = 0.
        np.fill_diagonal(weight_matrix, 0)
        N = len(matrix)
        triu_idx = np.triu_indices(N, 1)
        weight_list = weight_matrix[triu_idx[0],triu_idx[1]]
        graph_list = [(i,j,w) for i,j,w in zip(triu_idx[0],triu_idx[1],weight_list)
                      if w]
        G = nx.Graph()
        G.add_weighted_edges_from(graph_list)
        test_measure = self.generate_prediction(model)
        test_measure = test_measure / float(max(test_measure))
        if len(np.shape(test_measure)) == 1 and len(test_measure) == len(G.nodes):
            node_measure = test_measure
        else:
            node_measure = np.ones(len(graph_list))
        if len(np.shape(test_measure)) == 2:
            edge_measure = test_measure
        else:
            pos_weight_list = [w for w in weight_list if w]
            edge_measure = pos_weight_list
        nx.draw_networkx(G, edge_color=edge_measure, node_size=node_measure*300., **kwargs)
        return plt.gca()