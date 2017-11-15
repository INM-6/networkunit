import sciunit
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from abc import ABCMeta, abstractmethod
from networkunit.plots import sample_histogram


class two_sample_test(sciunit.Test):
    """
    Parent class for specific two sample test scenarios which enables
    initialization via a data model instead of a direct observation,
    interchangeable test scores, and basic sample visualization.
    """
    __metaclass__ = ABCMeta

    # required_capabilites = (ProducesSample, ) # Replace by more appropriate
                                              # capability in child class
                                              # i.e ProduceCovariances

    def __init__(self, observation=None, name=None, **params):
        super(two_sample_test,self).__init__(observation, name=name, **params)

    def generate_prediction(self, model, **kwargs):
        """
        To be overwritten by child class
        """
        self.params.update(kwargs)
        try:
            return model.produce_sample(**self.params)
        except:
            raise NotImplementedError("")

    def compute_score(self, observation, prediction):
        score = self.score_type.compute(observation, prediction, **self.params)
        return score

    def visualize_sample(self, model=None, ax=None, bins=100, palette=None,
                         sample_names=['observation', 'prediction'],
                         var_name='Measured Parameter', **kwargs):
        if palette is None:
            try:
                color_0 = self.observation_params['color']
            except:
                color_0 = sns.color_palette()[0]
            try:
                color_1 = model.params['color']
            except:
                color_1 = sns.color_palette()[1]
            palette = [color_0, color_1]
        if model is None:
            sample2 = None
        else:
            sample2 = self.generate_prediction(model, **self.params)
        sample1 = self.observation

        sample_histogram(sample1=sample1, sample2=sample2, ax=ax, bins=bins,
                         palette=palette, sample_names=sample_names,
                         var_name=var_name, **kwargs)
        return ax

    def visualize_score(self, model, ax=None, palette=None, **kwargs):
        """
        When there is a specific visualization function called plot() for the
        given score type, score_type.plot() is called;
        else call visualize_sample()
        Parameters
        ----------
        ax : matplotlib axis
            If no axis is passed a new figure is created.
        palette : list of color definitions
            Color definition may be a RGB sequence or a defined color code
            (i.e 'r'). Defaults to current color palette.
        Returns : matplotlib axis
        -------
        """
        # try:
        if palette is None:
            try:
                color_0 = self.observation_params['color']
            except:
                color_0 = sns.color_palette()[0]
            try:
                color_1 = model.params['color']
            except:
                color_1 = sns.color_palette()[1]
            palette = [color_0, color_1]
        ax = self.score_type.plot(self.observation,
                                  self.generate_prediction(model),
                                  ax=ax, palette=palette, **kwargs)
        return ax
        # except:
        #     self.visualize_sample(model=model, ax=ax, palette=palette)
        return ax