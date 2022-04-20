import sciunit


class ProducesSpikeTrains(sciunit.Capability):
    """
    Spike trains are to be represented as a list of neo.SpikeTrain objects.
    """
    def produce_spiketrains(self, **kwargs):
        return [st for st_list in self.produce_grouped_spiketrains(**kwargs)
                   for st      in st_list]
        # self.unimplemented()

    def produce_grouped_spiketrains(self, **kwargs):
        return [self.produce_spiketrains(**kwargs)]
