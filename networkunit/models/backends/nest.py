from sciunit.models.backends import Backend
import os
import time
try:
    import nest
    nest_available = True
except ImportError:
    nest_available = False
    nest = None

class NestBackend(Backend):
    name = 'Nest'

    def init_backend(self, **kwargs):
        if nest_available:
            print("Initialize {} backend".format(self.name))
            print("Nest version: {}".format(nest.version()))
            print("Use memory chache: {}"\
                  .format(kwargs.get('use_memory_cache', True)))
            print("Use disk chache: {}"\
                  .format(kwargs.get('use_disk_cache', True)))
            super(NestBackend, self).init_backend(**kwargs)
            return None
        else:
            raise ImportError('Nest not found!')

    def _backend_run(self):
        """Run the model via the backend."""

        ## Init Nest
        self.model.init_simulation()

        ## Build Network
        starttime = time.time()
        self.model.init_model()
        endtime = time.time()
        print("Network build time  : {:.2} s".format(endtime-starttime))

        ## Run Simulation
        starttime = time.time()
        if callable(getattr(self.model, 'simulate', None)):
            results = model.simulate(self.model.run_params['simtime'])
        else:
            results = nest.Simulate(self.model.run_params['simtime'])
        endtime = time.time()
        print("Simulation time  : {:.2} s".format(endtime-starttime))
        return results


    def save_results(self, path='.'):
        # ToDo: use NixIO or similar
        """Save results on disk."""
        with open(path, 'wb') as f:
            pickle.dump(self.results, f)
