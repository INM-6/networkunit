from networkunit.capabilities.cap_ProducesSpikeTrains import ProducesSpikeTrains
from networkunit.models.model_experimental_data import experimental_data
#import helpers
import numpy as np
import quantities as pq


class resting_state_data(experimental_data, ProducesSpikeTrains):
    """
    A model class to wrap network activity data (in form of spike trains) from
    a resting state experiment on a macaque monkey.
    """
    
    @property
    def datfile(self):
        return self.monkey


    def load(self, monkey, **kwargs):
        '''
        Loads resting state data from file_path and segments according to given
        movieSegmentation. Annotates unit_type (exc, inh, mixed).
        '''
    
        # Set user paths
        self.paths = helpers.set_user_paths(monkey)
        datfile = self.paths['datapath']
        eiThres = self.paths['eiThres']
        consFile = self.paths['consFile']
        
        # Load all spiketrains and annotate
        sts, _ = helpers.loadRS(dfname=datfile, load_lfp=False)
        spiketrains = self._neuron_type_separation(sts, class_file=consFile, eiThres=eiThres) 
    
        for i in xrange(len(spiketrains)):
            spiketrains[i] = spiketrains[i].rescale('ms')

        spiketrains = self.preprocess(spiketrains.tolist(), **self.params)
        
        return spiketrains
    

    def preprocess(self, spiketrain_list, state='RS', Lslice=1.*pq.s, 
                   movieSegmentation='fine', stack_slices=False, 
                   concat_in_time=False, **kwargs):
        """
        Performs preprocessing on the spiketrain data according to the given
        parameters which are passed down from the test parameters.
        """
        
        segfile = self.paths['segFileC'].replace('coarse', movieSegmentation)
        sts_array, _ = helpers.load_movie_segmentation(spiketrain_list, 
                                                       movieSegmentation, 
                                                       dfsegname=segfile)
        sts_sliced = helpers.select_sts(sts_array, state, Lslice)
        Ntrial_RS, Nunits = np.shape(sts_sliced)
        print 'Obtained {} slices of length {}s for monkey at state {}.'.format(
                Ntrial_RS, Lslice, state)
        
        if stack_slices:
            spiketrains = sts_sliced.flatten().tolist()
            for i in xrange(len(spiketrains)):
                t0 = spiketrains[i].t_start.rescale('ms')
                spiketrains[i].t_start = 0.*pq.ms
                spiketrains[i].t_stop = Lslice.rescale('ms')
                spiketrains[i] = spiketrains[i]-t0
            print 'Stacked slices to list with {} elements.'.format(len(spiketrains))
        elif concat_in_time:
            raise NotImplementedError
        else:
            raise NotImplementedError
            
        return spiketrains


    def _neuron_type_separation(self, sts,
                                class_file=None,
                                eiThres=None):
        '''
        This function loads the consistencies for each unit.
        The consistencies are the percentages of single waveforms with
        trough-to-peak times (t2p) larger than 350ms.

        Single units with small/large t2p are narrow/broad spiking units
        that are putative inhibitory/excitatory units.

        The input neo SpikeTrain objects will be anotated with neu_type 'exc',
        'inh', or 'mix' if too many inconsistent waveforms are present

        INPUT:
        eiThres [0-1]: threshold for the consistency. A small value will
                       result in highly consistent waveforms. However, a
                       large amount of units will then not be classified.

        OUTPUT:
        annotated sts
        '''
        Nunits = len(sts)
        consistency = np.loadtxt(class_file,
                                 dtype=np.float16)
        exc = np.where(consistency >= 1 - eiThres)[0]
        inh = np.where(consistency <= eiThres)[0]
        mix = np.where(np.logical_and(consistency > eiThres,
                                      consistency < 1 - eiThres))[0]
        for i in exc:
            sts[i].annotations['unit_type'] = 'exc'
        for i in inh:
            sts[i].annotations['unit_type'] = 'inh'
        for i in mix:
            sts[i].annotations['unit_type'] = 'mix'

        print '\n## Classification of waveforms resulted in:'
        print '{}/{} ({:0.1f}%) neurons classified as putative excitatory'.format(
            len(exc), Nunits, float(len(exc)) / Nunits * 100.)
        print '{}/{} ({:0.1f}%) neurons classified as putative inhibitory'.format(
            len(inh), Nunits, float(len(inh)) / Nunits * 100.)
        print '{}/{} ({:0.1f}%) neurons unclassified (mixed)\n'.format(
            len(mix), Nunits, float(len(mix)) / Nunits * 100.)
        return sts