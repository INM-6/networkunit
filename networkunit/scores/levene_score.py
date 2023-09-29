import sciunit
import numpy as np
from scipy.stats import levene

# COMMENTS ##############################
#
# all open issues marked with "## TODO ##"
#
#
# COMMENTS ##############################


#==============================================================================

class levene_score(sciunit.Score):
    """
    A Levene Test score.
    Null hypothesis: homogeneity of variance or homoscedasticity
    """

#    _allowed_types = (bool,) ## TODO ## don't know what to set here

    _description = ("Levene's test is an inferential statistic used to assess the equality of variances. "
                  + "It tests the null hypothesis that the population variances are equal "
                  + "(called homogeneity of variance or homoscedasticity). "
                  + "If the resulting p-value of Levene's test is less than some significance level "
                  + "(typically 0.05), the obtained differences in sample variances are unlikely to have "
                  + "occurred based on random sampling from a population with equal variances.")

    @classmethod
    def compute(cls, observation, prediction):
        """
        Computes p-value of probability that variances are equal.
        """

        x = prediction[~np.isnan(prediction)]
        y = observation[~np.isnan(observation)]
        pvalue = levene(x, y).pvalue
        return levene_score(pvalue)

    @property
    def sort_key(self):
        return self.score

    def __str__(self):
        return 'pvalue = {:.3}'.format(self.score)
