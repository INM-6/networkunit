import numpy as np
from scipy.stats import ttest_ind
import sciunit
from networkunit.scores import to_precision


class students_t(sciunit.Score):
    """
    Student's T-test
    """
    score = np.nan

    # TODO: should be named ttest only because if equal_var=False -> Welch's t-test
    # The computation is performed by the scipy.stats.ttest_ind() function.

    @classmethod
    def compute(self, data_sample_1, data_sample_2, equal_var=True, **kwargs):
        # Filter out nans and infs
        if len(np.shape(data_sample_1)) > 1:
            data_sample_1 = data_sample_1.flatten()
        if len(np.shape(data_sample_2)) > 1:
            data_sample_2 = data_sample_2.flatten()
        init_length = [len(smpl) for smpl in [data_sample_1, data_sample_2]]
        sample1 = np.array(data_sample_1)[np.isfinite(data_sample_1)]
        sample2 = np.array(data_sample_2)[np.isfinite(data_sample_2)]

        if init_length[0] - len(sample1) or init_length[1] - len(sample2):
            print("Warning: {} non-finite elements of the data samples were "
                  "filtered."
                  .format(sum(init_length)
                          - sum([len(s) for s in [sample1, sample2]])))

        t, pvalue = ttest_ind(sample1, sample2, equal_var=equal_var)
        score = students_t(t)
        score.pvalue = pvalue
        score.data_size = [len(sample1), len(sample2)]
        return score

    @property
    def sort_key(self):
        return self.score

    def __str__(self):
        return "\n\n\033[4mStudent's t-test\033[0m" \
             + "\n\tdatasize: {} \t {}" \
               .format(self.data_size[0], self.data_size[1]) \
             + "\n\tt = {:.3f} \t p value = {}\n\n" \
               .format(self.score, to_precision(self.pvalue,3))
