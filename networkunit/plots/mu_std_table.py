# Currently meant only for CellDensityTest
# Will be made generic soon
import numpy as np
from tabulate import tabulate

#==============================================================================

class mu_std_table:
    """
    Displays data in table inside text file
    Note: can be extended in future to provide more flexibility
    """

    def __init__(self, testObj):
        self.testObj = testObj
        self.filename = "score_summary"

    def create(self, mid_keys = []):
        ## TODO ## add separate p-values (scores) to table
        # check how to make LevenScore pass dict of pvalues instead of max_pval
        filepath = self.testObj.path_test_output + self.filename + '.txt'
        dataFile = open(filepath, 'w')
        dataFile.write("==============================================================================\n")
        dataFile.write("Test Name: %s\n" % self.testObj.name)
        dataFile.write("Neuron type: %s\n" % self.testObj.neu_type)
        dataFile.write("Resting state (RS) or movement (M): %s\n" % self.testObj.state)
        dataFile.write("Model Name: %s\n" % self.testObj.model_name)
        dataFile.write("Score Type: %s\n" % self.testObj.score.description)
        dataFile.write("------------------------------------------------------------------------------\n")
        header_list = ["Exp. mean", "Exp. std", "Model mean", "Model std"] #, "p-value"]
        row_list = []
        obs = self.testObj.observation
        prd = self.testObj.prediction
        C_mu_std = self.get_mu_std(obs, prd)
        row_list.append([C_mu_std['obs']['mu'], 
                         C_mu_std['obs']['std'], 
                         C_mu_std['prd']['mu'],
                         C_mu_std['prd']['std'],
                         ])
        dataFile.write(tabulate(row_list, headers=header_list, tablefmt='orgtbl'))
        dataFile.write("\n------------------------------------------------------------------------------\n")
        dataFile.write("Final Score: %s\n" % self.testObj.score)
        dataFile.write("==============================================================================\n")
        dataFile.close()
        return filepath
        

    def get_mu_std(self, observation, prediction):
        """
        Calculates mean and standard deviation of values in observation and 
        prediction assuming Gaussian distribution.
        """
        C_mu_std = {
            'prd' : {
                'mu' : np.nanmean(prediction),
                'std': np.nanstd (prediction),
            },
            'obs' : {
                'mu' : np.nanmean(observation),
                'std': np.nanstd (observation),
            }
        }
        return C_mu_std        