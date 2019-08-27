import sciunit
import networkx as nx
from networkunit.tests.test_correlation_matrix_test import correlation_matrix_test
from networkunit.tests.test_two_sample_test import two_sample_test
import matplotlib.pyplot as plt
import seaborn as sns
from quantities import ms
import numpy as np
from copy import copy
import re

class graph_centrality_helperclass(sciunit.Test):
    """
    Abstract test class to compare graph centrality measures of a set of
    spiking neurons in a network. This test needs to be combined with a test
    which generates a prediction in form of matrix.
    From this matrix the chosen graph measure is calculated
    and passed on in scalar, vector, or matrix form, depending on whether the
    measure is network-wise, node-wise, or node-pair-wise.
    The executable test has to inherit from the graph_measure_test and the
    matrix generating test
    in the order (TestM2M), graph_centrality_helperclass, matrix_test.

    Parameters
        ----------
        edge_threshold: float (default: 0)
            Threshold for the matrix values to create the graph.
        graph_measure: 'degree strength', 'closeness', 'betweenness',
            'edge betweenness', 'katz', 'clustering coefficient',
            'transitivity', 'small-worldness'
    """

    def generate_prediction(self, model, **kwargs):
        prediction = self.get_prediction(model)
        if prediction is None:
            if kwargs:
                self.params.update(kwargs)
            if 'graph_measure' not in self.params:
                raise ValueError('No graph_measure set!')
            matrix = super(graph_centrality_helperclass, self).\
                                        generate_prediction(model, **kwargs)
            self.prediction_dim = 2
            N = len(matrix)

            G = self.get_prediction(model, key=self.test_hash + '_graph')
            if G in None:
                G = self.build_graph(matrix)
                self.set_prediction(model, G, key=self.test_hash + '_graph')

            if self.params['graph_measure'] == 'degree strength':
                prediction = self.degree_strength(G, N)

            if self.params['graph_measure'] == 'closeness':
                prediction = self.closeness(G, N)

            if self.params['graph_measure'] == 'betweenness':
                prediction = self.betweenness(G, N)

            if self.params['graph_measure'] == 'edge betweenness':
                prediction = self.edge_betweenness(G, N)

            if self.params['graph_measure'] == 'katz':
                prediction = self.katz(G, N)

            if self.params['graph_measure'] == 'clustering coefficient':
                prediction = self.clustering_coefficent(G, N)

            if self.params['graph_measure'] == 'transitivity':
                prediction = self.transitivity(G, N, matrix)

            if self.params['graph_measure'] == 'small-worldness':
                prediction = self.small_worldness(G, N)

            else:
                raise KeyError('Graph measure not know!')

            self.set_prediction(model, prediction)
        return prediction

    def degree_strength(self, G, N):
        degrees = nx.degree(G, weight='weight')
        self.prediction_dim = 1
        degree_array = np.array([d[1] for d in degrees])
        return np.append(degree_array, np.zeros(N - len(degree_array)))

    def closeness(self, G, N):
        weight_dict = nx.get_edge_attributes(G, 'weight')
        weight_sum = float(np.sum(np.array([weight_dict[edge]
                                            for edge in weight_dict.keys()])))
        for edge in G.edges:
            G.edges[edge].update(distance = G.edges[edge]['weight'] / weight_sum)
        closeness = nx.closeness_centrality(G, distance='distance')
        self.prediction_dim = 1
        closeness_array =  np.array([closeness[i] for i in closeness.keys()])
        return np.append(closeness_array, np.zeros(N-len(closeness_array)))

    def betweenness(self, G, N):
        betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True)
        self.prediction_dim = 1
        betweenness_array = np.array([betweenness[i] for i in betweenness.keys()])
        return np.append(betweenness_array, np.zeros(N - len(betweenness_array)))

    def edge_betweenness(self, G, N):
        edge_betweenness = nx.edge_betweenness_centrality(G, weight='weight', normalized=True)
        N = max(G.nodes) + 1
        betweenness_matrix = np.zeros((N, N))
        for edge in edge_betweenness:
            betweenness_matrix[edge] = edge_betweenness[edge]
            betweenness_matrix[edge[1],edge[0]] = edge_betweenness[edge]
        return betweenness_matrix

    def katz(self, G, N):
        katz = nx.katz_centrality(G, weight='weight', normalized=True)
        self.prediction_dim = 1
        return np.array([katz[i] for i in katz.keys()])

    def clustering_coefficent(self, G, N):
        clustering = nx.clustering(G, weight='weight')
        self.prediction_dim = 1
        cc_array =  np.array([clustering[i] for i in clustering.keys()])
        return np.append(cc_array, np.zeros(N-len(cc_array)))

    def transitivity(self, G, N, matrix):
        B = None
        if 'bin_num' in self.params:
            B = self.params['bin_num']
        elif 'binsize' in self.params:
            if 't_start' in self.params and 't_stop' in self.params:
                B = float((self.params['t_stop']-self.params['t_start'])/self.params['binsize'])
        if 'edge_threshold' in self.params:
            edge_threshold = self.params['edge_threshold']
        elif B is not None:
            Z = 1.96 / np.sqrt(B - 3.)
            edge_threshold = (np.exp(2. * Z) - 1.) / (np.exp(2. * Z) + 1.)
            self.params['edge_threshold'] = edge_threshold
        else:
            raise ValueError('Not enough information to threshold graph!')
        G = self.build_graph(matrix)
        self.prediction_dim = 0
        return nx.transitivity(G)

    def small_worldness(self, G, N, matrix):
        G_rand = nx.gnm_random_graph(G.number_of_nodes(), G.number_of_edges())
        path_length = nx.average_shortest_path_length(G, weight='weight')

        # G_rand path length
        weight_matrix = copy(matrix)
        if 'edge_threshold' in self.params:
            edge_threshold = self.params['edge_threshold']
        else:
            edge_threshold = 0.
        non_edges = np.where(weight_matrix <= edge_threshold)
        weight_matrix[non_edges[0], non_edges[1]] = 0.
        np.fill_diagonal(weight_matrix, 0)
        triu_idx = np.triu_indices(N, 1)
        rand_weights = weight_matrix[triu_idx[0], triu_idx[1]]
        rand_weights = np.array([w for w in rand_weights if w])
        np.random.shuffle(rand_weights)
        # print len(rand_weights), len(G_rand.edges())
        for count, e in enumerate(G_rand.edges()):
            G_rand[e[0]][e[1]]['weight'] = rand_weights[count]

        path_length_rand = nx.average_shortest_path_length(G_rand, weight='weight')

        # transitivity
        transitivity = self.transitivity(G, N, matrix)

        # G_rand transitivity
        G_rand2 = nx.gnm_random_graph(G.number_of_nodes(), G.number_of_edges())
        transitivity_rand = nx.transitivity(G_rand2)
        self.prediction_dim = 0
        return (transitivity/transitivity_rand) / (path_length/path_length_rand)

    def build_graph(self, matrix):
        weight_matrix = copy(matrix)
        if 'edge_threshold' in self.params:
            edge_threshold = self.params['edge_threshold']
        else:
            edge_threshold = 0.
        non_edges = np.where(weight_matrix <= edge_threshold)
        weight_matrix[non_edges[0], non_edges[1]] = 0.
        np.fill_diagonal(weight_matrix, 0)
        triu_idx = np.triu_indices(N, 1)
        weight_list = weight_matrix[triu_idx[0],triu_idx[1]]
        graph_list = [(i,j,w) for i,j,w in
                      zip(triu_idx[0],triu_idx[1],weight_list) if w]
        G = nx.Graph()
        G.add_weighted_edges_from(graph_list)
        return G

    def visualize_samples(self, model1=None, model2=None, ax=None,
                         palette=None, remove_autocorr=True,
                         sample_names=['observation', 'prediction'],
                         var_name='Measured Parameter', sort=False, **kwargs):

        if hasattr(self, 'prediction_dim') and self.prediction_dim == 1:
            if ax is None:
                fig, new_ax = plt.subplots()
            else:
                new_ax = ax
            samples, palette, names = super(graph_centrality_helperclass, self)._create_plotting_samples(
                                                          model1=model1,
                                                          model2=model2,
                                                          palette=palette)
            nodes = [[]]*len(samples)
            if sort:
                for count, sample in enumerate(samples):
                    samples[count] = np.sort(samples[count])[::-1]
                    nodes[count] = np.arange(len(samples[count]))
            else:
                for count, sample in enumerate(samples):
                    nodes[count] = self.graph['graph_{}'.format(names[count])].nodes.keys()
                    samples[count] = sample[:len(nodes[count])]

            for count, sample in enumerate(samples):
                sign = -1 if count else 1
                new_ax.bar(nodes[count], sign*sample, color=palette[count],
                           label=names[count], width=1., edgecolor='w')
            new_ax.set_ylabel(self.params['graph_measure'])
            new_ax.set_xlabel('Neurons')

            ymax = np.max(np.abs(new_ax.get_ylim()))
            if len(samples) == 2:
                new_ax.set_ylim((-ymax, ymax))
                plt.draw()
                mplticklabels = new_ax.get_yticklabels(which='both')
                ticklabels = np.array([])
                for label in mplticklabels:
                    if label.get_text():
                        m = re.search(r'\d+', label.get_text())
                        numeric = m.group()  # retrieve numeric string
                        sign = 1
                        if '-' in label.get_text() or u'\u2212' in label.get_text():
                            sign = -1
                        ticklabels = np.append(ticklabels, sign*float(numeric))
                new_ax.set_yticks(ticklabels)
                ticklabels = np.abs(ticklabels)
                new_ax.set_yticklabels([str(label) for label in ticklabels])

            sns.despine()
            plt.legend()

        else:
            super(graph_centrality_helperclass,self).visualize_samples(
                                                           model1=model1,
                                                           model2=model2,
                                                           ax=ax,
                                                           palette=palette,
                                                           remove_autocorr=remove_autocorr,
                                                           sample_names=sample_names,
                                                           var_name=var_name,
                                                           **kwargs)
