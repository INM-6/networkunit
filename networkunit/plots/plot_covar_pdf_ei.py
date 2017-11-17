import numpy as np
#import matplotlib
# Force matplotlib to not use any Xwindows backend.
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

#==============================================================================

class covar_pdf_ei:
    """
    Plots the probability density distributions of prediction and observation 
    for both excitatory-excitatory and inhibitory-inhibitory cross-covariances.
    """

    def __init__(self, testObj):
        self.testObj = testObj
        self.filename = "covar_pdf_ei"

    def create(self):
        fig = plt.figure()
        obs = self.testObj.observation
        prd = self.testObj.prediction
        pdf_obs, __   = self.get_pdf(obs)
        pdf_prd, bins = self.get_pdf(prd)
        for i, key in enumerate(prd.keys()):
            plt.subplot(1,2,i+1)
            plt.plot(bins, pdf_obs[key], '-k', label='obs')
            plt.plot(bins, pdf_prd[key], '-r', label='prd')
            plt.title(key+'-'+key)
        plt.legend()
        plt.tight_layout()
        plt.show()
        fig = plt.gcf()
        fig.set_size_inches(12, 5)
        filepath = self.testObj.path_test_output + self.filename + '.pdf'
        plt.savefig(filepath, dpi=600,)
        return filepath
    
            
    def get_pdf(self, C, 
                binrange=[-0.4, 0.4],
                nbins=80):
        '''
        Calculates probability density function of cross-covariances.
        INPUT:
            C: dictionary of exc/inh containing elements covariance matrices
            binrange: binrange used for histogram
            nbins: number of bins within binrange
        OUTPUT: 
            pdf: dictionary of probability density distributions for 
                 cross-covariances of 'exc' and 'inh'
            bins: bin centers for pdf
        '''
        pdf = dict()   
        for key in C.keys():
            pdf[key], bins = np.histogram(C[key], bins=nbins, 
                                          range=binrange, density=True)
        bins = bins[1:]-(bins[1]-bins[0])/2
        return pdf, bins