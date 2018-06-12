import numpy as np
#import matplotlib
# Force matplotlib to not use any Xwindows backend.
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


class covar_pdf:
    """
    Plots the probability density distributions of prediction and observation 
    for cross-covariances.
    """

    def __init__(self, testObj):
        self.testObj = testObj
        self.filename = "covar_pdf_"+self.testObj.neu_type

    def create(self):
        fig = plt.figure(figsize=(4,4))
        obs = self.testObj.observation
        prd = self.testObj.prediction
        pdf_obs, __   = self.get_pdf(obs)
        pdf_prd, bins = self.get_pdf(prd)
        plt.plot(bins, pdf_obs, '-k', label='obs')
        plt.plot(bins, pdf_prd, '-r', label='prd')
        plt.legend()
        plt.tight_layout()
        plt.show()
        fig = plt.gcf()
        fig.set_size_inches(5, 5)
        filepath = self.testObj.path_test_output + self.filename + '.pdf'
        plt.savefig(filepath, dpi=600,)
        return filepath
    
            
    def get_pdf(self, C, 
                binrange=[-0.4, 0.4],
                nbins=80):
        '''
        Calculates probability density function of cross-covariances.
        INPUT:
            C: covariance matrix
            binrange: binrange used for histogram
            nbins: number of bins within binrange
        OUTPUT: 
            pdf: dictionary of probability density distributions for 
                 cross-covariances of 'exc' or 'inh'
            bins: bin centers for pdf
        ''' 
        pdf, bins = np.histogram(C, bins=nbins, 
                                 range=binrange, density=True)
        bins = bins[1:]-(bins[1]-bins[0])/2
        return pdf, bins