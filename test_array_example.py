import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import sys
sys.path.insert(0,'/home/robin/Projects/INM6/elephant')
sys.path.append('/home/robin/Projects/INM6/python-neo')
sys.path.append('/home/robin/Projects/NetworkUnit')
sys.path.insert(0,'/home/robin/Projects/sciunit')
from networkunit import models, tests, scores
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
                                shuffle=False, name='modelA')
model_B = models.stochastic_activity(size=size, correlations=cc, assembly_sizes=A,
                                correlation_method='CPP', t_start=tstart, t_stop=tstop,
                                shuffle=False, name='modelB')

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
              'alpha': 0.0001}
    def compute_score(self, prediction1, prediction2):
        score = self.score_type.compute(prediction1, prediction2, **self.params)
        return score

test_list = [m2m_cov_kl_test_2msbins_100sample(),
             angle_test(),
             ]
score_list = [[]] * len(test_list)

for count, test in enumerate(test_list):
    score_list[count] = test.judge([model_A, model_B])
    print score_list[count].score