from networkunit.tests.test_correlation_test import correlation_test
from networkunit.capabilities.cap_ProducesSpikeTrains import ProducesSpikeTrains
from networkunit.plots.plot_correlation_matrix import plot_correlation_matrix
from scipy.cluster.hierarchy import linkage, dendrogram
import fastcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from copy import copy
from abc import ABCMeta, abstractmethod


class correlation_matrix_test(correlation_test):
    """
    Test to compare the pairwise correlations of a set of neurons in a network.
    """
    # __metaclass__ = ABCMeta

    required_capabilities = (ProducesSpikeTrains, )

    def generate_prediction(self, model, **kwargs):
        # call the function of the required capability of the model
        # and pass the parameters of the test class instance in case the
        if not hasattr(model, 'prediction'):
            model.prediction = {}
        if self.test_hash in model.prediction:
            cc_matrix = model.prediction[self.test_hash]
        else:
            if kwargs:
                self.params.update(kwargs)
            spiketrains = model.produce_spiketrains(**self.params)
            cc_matrix = self.generate_cc_matrix(spiketrains=spiketrains,
                                                model=model, **self.params)
            if 'cluster_matrix' in self.params and self.params['cluster_matrix']:
                np.fill_diagonal(cc_matrix, 1.)
                if 'cluster_method' not in self.params:
                    self.params.update(cluster_method='ward')
                try:
                    try:
                        linkagematrix = linkage(squareform(1. - cc_matrix),
                                                method=self.params['cluster_method'])
                    except:
                        print 'using fastcluster'
                        linkagematrix = fastcluster.linkage(squareform(1. - cc_matrix),
                                                        method=self.params['cluster_method'])
                    dendro = dendrogram(linkagematrix, no_plot=True)
                    order = dendro['leaves']
                    model.cluster_order = order
                    cc_matrix = cc_matrix[order, :][:, order]
                except Exception as e:
                    print 'Clustering failed!'
                    print e
            # np.fill_diagonal(cc_matrix, 0.)
            model.prediction[self.test_hash] = cc_matrix
        return cc_matrix

    def visualize_samples(self, model1=None, model2=None, ax=None, labels=None,
                         palette=None, remove_autocorr=True, vmin=None, vmax=None,
                         sample_names=['observation', 'prediction'], sort=False,
                         var_name='Measured Parameter', linkmethod='ward',
                         **kwargs):

        matrices, palette, names  = self._create_plotting_samples(model1=model1,
                                                          model2=model2,
                                                          palette=palette)
        if ax is None:
            fig, ax = plt.subplots(ncols=len(matrices))
        if len(matrices) == 1:
            ax = [ax]

        if self.observation is None:
            sample_names[0] = model1.name
            if model2 is not None:
                sample_names[1] = model2.name
        else:
            sample_names[1] = model1.name

        if 'cluster_matrix' not in self.params:
            self.params.update(cluster_matrix=False)

        plot_correlation_matrix(matrices[0], ax=ax[0], remove_autocorr=remove_autocorr,
                                labels=labels, sort=sort, cluster=self.params['cluster_matrix'],
                                linkmethod=linkmethod, dendrogram_args={}, vmin=vmin, vmax=vmax,
                                **kwargs)
        ax[0].set_title(sample_names[0])
        if len(matrices) > 1:
            plot_correlation_matrix(matrices[1], ax=ax[1], remove_autocorr=remove_autocorr,
                                    labels=labels, sort=sort, cluster=self.params['cluster_matrix'],
                                    linkmethod=linkmethod, dendrogram_args={},  vmin=vmin, vmax=vmax,
                                    **kwargs)
            ax[1].set_title(sample_names[1])
        return ax

    def draw_graph(self, model, draw_edge_threshold=None, **kwargs):
        if draw_edge_threshold is None and hasattr(self, 'graph') \
                            and 'graph_{}'.format(model.name) in self.graph:
            G = self.graph['graph_{}'.format(model.name)]
        else:
            spiketrains = model.produce_spiketrains(**self.params)
            matrix = self.generate_cc_matrix(spiketrains=spiketrains, **self.params)
            weight_matrix = copy(matrix)
            if draw_edge_threshold is None:
                if 'edge_threshold' in self.params:
                    draw_edge_threshold = self.params['edge_threshold']
                else:
                    draw_edge_threshold = 0.
            non_edges = np.where(weight_matrix <= draw_edge_threshold)
            if len(non_edges[0]):
                weight_matrix[non_edges[0], non_edges[1]] = 0.
            np.fill_diagonal(weight_matrix, 0)
            N = len(matrix)
            triu_idx = np.triu_indices(N, 1)
            weight_list = weight_matrix[triu_idx[0],triu_idx[1]]
            graph_list = [(i,j,w) for i,j,w in
                          zip(triu_idx[0],triu_idx[1],weight_list) if w]
            G = nx.Graph()
            G.add_weighted_edges_from(graph_list)
            if draw_edge_threshold is None:
                if not hasattr(self, 'graph'):
                    self.graph = {}
                self.graph['graph_{}'.format(model.name)] = G

        test_measure = self.generate_prediction(model)
        test_measure = test_measure / float(np.max(test_measure))
        if len(np.shape(test_measure)) == 1 and len(test_measure) == len(G.nodes):
            node_measure = test_measure
        else:
            node_measure = np.ones(len(G.nodes))
        if False: #len(np.shape(test_measure)) == 2:
            edge_measure = test_measure
        else:
            weight_dict = nx.get_edge_attributes(G, 'weight')
            edge_measure = [weight_dict[edge] for edge in weight_dict.keys()]
        nx.draw_networkx(G, edge_color=edge_measure, node_size=node_measure*300., **kwargs)
        plt.gca().set_title(model.name)
        return plt.gca()