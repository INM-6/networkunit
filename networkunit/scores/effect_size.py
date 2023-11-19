from __future__ import division
import numpy as np
import sciunit
import matplotlib.pyplot as plt

class effect_size(sciunit.Score):
    """
    Calculates the effect size between samples.
    """

    score = np.nan

    @classmethod
    def compute(self, observation, prediction, **kwargs):

        def s_pooled(sample1, sample2):
            n = len(sample1)
            s = np.std(sample1)
            nn = len(sample2)
            sn = np.std(sample2)
            return np.sqrt(
                ((n - 1.) * s ** 2 + (nn - 1.) * sn ** 2) / (n + nn - 2.))

        def func_effect_size(sample1, sample2):
            return abs(np.mean(sample1) - np.mean(sample2)) / s_pooled(sample1,
                                                                       sample2)

        def CI(sample1, sample2):
            """
            95% confidence interval
            """
            n = len(sample1)
            nn = len(sample2)
            es = func_effect_size(sample1, sample2)
            return 1.96 * np.sqrt(
                (n + nn) / (n * nn) + es ** 2 / (2. * (n + nn - 2.)))

        es = func_effect_size(observation, prediction)
        ci = CI(observation, prediction)
        self.score = effect_size(es)
        self.score.data_size = [len(observation), len(prediction)]
        # self.CI = (es - ci, es + ci)
        self.score.CI = (es - ci, es + ci)
        return self.score


    @property
    def sort_key(self):
        return self.score

    def __str__(self):
        return "\n\n\033[4mEffect Size\033[0m" \
             + "\n\tdatasize: {} \t {}" \
               .format(self.data_size[0], self.data_size[1]) \
             + "\n\tEffect Size = {:.3f} \t CI = ({:.3f}, {:.3f})\n\n" \
               .format(self.score, self.CI[0], self.CI[1])