import matplotlib
matplotlib.use('pdf')
from matplotlib import rc, rcParams
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'/home/robin/Projects/INM6/elephant')
sys.path.append('/home/robin/Projects/INM6/python-neo')
sys.path.append('/home/robin/Projects/NetworkUnit')
sys.path.insert(0,'/home/robin/Projects/sciunit')
from networkunit import models, tests, scores
from networkunit.scores import to_precision
import sciunit
import seaborn as sns
from quantities import Hz, ms
rc('text', usetex=True)


# SET UP MODELS TO COMPARE
size = 100
A = [10,5]
cc = .4
rate = 10*Hz
tstart = 0*ms
tstop = 10000*ms
binsize = 2*ms

model_A = models.stochastic_activity(size=size, correlations=cc, assembly_sizes=A,
                                correlation_method='CPP', t_start=tstart, t_stop=tstop,
                                shuffle=True, shuffle_seed=321, name='modelA')
model_B = models.stochastic_activity(size=size, correlations=cc, assembly_sizes=A,
                                correlation_method='CPP', t_start=tstart, t_stop=tstop,
                                shuffle=True, shuffle_seed=321, name='modelB')

# SET UP TESTS
class m2m_cov_kl_test_2msbins_100sample(sciunit.TestM2M, tests.correlation_dist_test):
    score_type = scores.ks_distance
    params = {'max_subsamplesize': 100,
              'align_to_0' : True,
              'binsize' : 2 * ms,
              't_start' : 0 * ms,
              't_stop' : 10000 * ms}


class angle_test(sciunit.TestM2M, tests.correlation_matrix_test):
    score_type = scores.eigenvector_angle
    params = {'all_to_all': False,
              'two_sided': False,
              'alpha': 0.0001,
              }
    def compute_score(self, prediction1, prediction2):
        score = self.score_type.compute(prediction1, prediction2, **self.params)
        return score

class generalized_angle_test(sciunit.TestM2M, tests.generalized_correlation_matrix_test):
    score_type = scores.eigenvector_angle
    params = {'all_to_all': False,
              'two_sided': False,
              'alpha': 0.0001,
              'maxlag': 100,  # in bins
              'binsize': 2 * ms,
              'time_reduction': 'max'
              }
    def compute_score(self, prediction1, prediction2):
        score = self.score_type.compute(prediction1, prediction2, **self.params)
        return score

if __name__ == '__main__':

    test_list = [m2m_cov_kl_test_2msbins_100sample(),
                 angle_test(),
                 generalized_angle_test()
                 ]
    score_list = [[]] * len(test_list)

    task_id = int(sys.argv[-1])

    for count, test in enumerate(test_list[:-1]):
        score = test.judge([model_A, model_B]).iloc[0,1]
        test.visualize_sample(model_A, model_B)
        plt.savefig('/home/r.gutzen/Output/networkunit/figures/task-{}_test-{}_sample.pdf'.format(task_id, test.name))
        test.visualize_score(model_A, model_B)
        plt.savefig('/home/r.gutzen/Output/networkunit/figures/task-{}_test-{}_score.pdf'.format(task_id, test.name))
        print "\nTest: {}".format(test.name)
        print "Score: {} \t\t pvalue: {}".format(to_precision(score.score,3),
                                                 score.pvalue if 'pvalue' in dir(score) \
                                                 else 'nan')
        score_list[count] = score