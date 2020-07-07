import sciunit

class ProducesSpikeTrains(sciunit.Capability):
    """
    Spike trains are to be represented as a list of neo.SpikeTrain objects.
    """
    def produce_spiketrains(self, **kwargs):
        self.unimplemented()

    def produce_grouped_spiketrains(self, **kwargs):
        return list(self.produce_spiketrains(**kwargs))
