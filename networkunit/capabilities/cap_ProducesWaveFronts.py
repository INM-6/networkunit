import sciunit

class ProducesWaveFronts(sciunit.Capability):
    """
    Wave fronts can either be the transitions from down to up states in the
    case of 'Slow Waves', or the peak of a wave cylce (phase = pi/2) in the
    case of oscillatory waves.
    The wavefronts are to be represented as a list of neo.SpikeTrain objects.
    Each SpikeTrain has to have the following annotations:
        "coordinates"       : tuple, (x, y)
        "grid_size"         : tuple, (dim_x, dim_y)
        "electrode_distance": Quantity, d * quantities.mm
    """
    def produce_wavefronts(self, **kwargs):
        self.unimplemented()
