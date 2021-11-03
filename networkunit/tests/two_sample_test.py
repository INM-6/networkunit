import sciunit
import seaborn as sns
from uuid import uuid4
from networkunit.plots.sample_histogram import sample_histogram
from networkunit.utils import use_cache
from elephant.parallel import SingleProcess


class two_sample_test(sciunit.Test):
    """
    Parent class for specific two sample test scenarios which enables
    initialization via a data model instead of a direct observation,
    interchangeable test scores, and basic sample visualization.
    """

    # required_capabilites = (ProducesSample, ) # Replace by more appropriate
                                              # capability in child class
                                              # i.e ProduceCovariances

    default_params = {**sciunit.Test.default_params,
                      'parallel_executor': SingleProcess()}

    def __init__(self, observation=None, name=None, **params):
        self.test_hash = uuid4().hex
        super(two_sample_test, self).__init__(observation, name=name, **params)

    @use_cache
    def generate_prediction(self, model):
        """
        To be overwritten in child class. The following example code
        should be reused to enable cache storage and prevent multiple
        calculation.
        """
        #############################
        # calculate prediction here #
        raise NotImplementedError("")
        #############################
        return prediction

    def compute_score(self, observation, prediction):
        score = self.score_type.compute(observation, prediction, **self.params)
        return score

    def get_cache(self, model, key=None):
        if key is None:
            key = self.test_hash
        elif not key:
            return None
        prediction = None
        if hasattr(model, 'backend'):
            if model._backend.use_memory_cache:
                prediction = model._backend.get_memory_cache(key=key)
            elif model._backend.use_disk_cache:
                prediction = model._backend.get_disk_cache(key=key)
        return prediction

    def set_cache(self, model, prediction, key=None):
        if key is None:
            key = self.test_hash
        elif not key:
            return False
        if hasattr(model, 'backend'):
            if model._backend.use_memory_cache:
                model._backend.set_memory_cache(prediction, key=key)
                return True
            elif model._backend.use_disk_cache:
                model._backend.set_disk_cache(prediction, key=key)
                return True
        return False

    def _create_plotting_samples(self, model1=None, model2=None, palette=None):
        samples = []
        names = []
        if palette is None:
            palette = []
            fill_palette = True
        else:
            fill_palette = False
        if self.observation is not None:
            samples += [self.observation]
            if hasattr(self, 'observation_model'):
                names += [self.observation_model.name]
            else:
                names += ['observation']
            if fill_palette:
                try:
                    palette = palette + [self.observation_params['color']]
                except:
                    palette = palette + [sns.color_palette()[0]]
        if model1 is not None:
            samples += [self.generate_prediction(model1)]
            names += [model1.name]
            if fill_palette:
                try:
                    palette = palette + [model1.params['color']]
                except:
                    palette = palette + [sns.color_palette()[len(samples)-1]]
        if model2 is not None:
            samples += [self.generate_prediction(model2)]
            names += [model2.name]
            if fill_palette:
                try:
                    palette = palette + [model2.params['color']]
                except:
                    palette = palette + [sns.color_palette()[len(samples)-2]]

        return samples, palette, names

    def visualize_samples(self, model1=None, model2=None, ax=None, bins=100,
                          palette=None, density=True,
                          sample_names=['observation', 'prediction'],
                          var_name='Measured Parameter', **kwargs):

        samples, palette, names = self._create_plotting_samples(model1=model1,
                                                         model2=model2,
                                                         palette=palette)
        if self.observation is None:
            sample_names[0] = model1.name
            if model2 is not None:
                sample_names[1] = model2.name
        else:
            sample_names[1] = model1.name

        if len(samples) == 1:
            sample_2 = None
        else:
            sample_2 = samples[1]

        sample_histogram(sample1=samples[0], sample2=sample_2,
                         ax=ax, bins=bins,
                         palette=palette, sample_names=sample_names,
                         var_name=var_name, density=density, **kwargs)
        return ax

    def visualize_score(self, model1, model2=None, ax=None, palette=None,
                        sample_names=None, **kwargs):
        """
        When there is a specific visualization function called plot() for the
        given score type, score_type.plot() is called;
        else call visualize_sample()

        Parameters:
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
        samples, palette, names = self._create_plotting_samples(model1=model1,
                                                         model2=model2,
                                                         palette=palette)
        s_names = ['observation', 'prediction']
        if self.observation is None:
            s_names[0] = model1.name
            if model2 is not None:
                s_names[1] = model2.name
        else:
            s_names[1] = model1.name

        if sample_names is None:
            sample_names = s_names

        kwargs.update(self.params)
        ax = self.score_type.plot(samples[0], samples[1],
                                  ax=ax, palette=palette,
                                  sample_names=sample_names,
                                  **kwargs)
        # except:
        #     self.visualize_sample(model=model, ax=ax, palette=palette)

        return ax
