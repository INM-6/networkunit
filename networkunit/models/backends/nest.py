from sciunit.models.backends import Backend
import os
try:
    import nest
    nest_available = True
except:
    nest_available = False

class NestBackend(Backend):
    name = 'Nest'

    def init_backend(self, **kwargs):
        print("Initialize {} backend".format(self.name))
        print("Nest version: {}".format(nest.__version__))
        print("Use memory chache: {}"\
              .format(kwargs.get('use_memory_cache', True)))
        print("Use disk chache: {}"\
              .format(kwargs.get('use_disk_cache', True)))
        super(NestBackend, self).init_backend(**kwargs)
        return None


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
        if callable(getattr(model, 'simulate', None)):
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
