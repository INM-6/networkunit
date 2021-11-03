from sciunit.models.backends import Backend
import time


class NestBackend(Backend):
    name = 'Nest'
    nest_instance = None

    def init_backend(self, **kwargs):
        print("Initialize {} backend".format(self.name))
        # print("Nest version: {}".format(nest.version()))
        print("Use memory chache: {}"\
              .format(kwargs.get('use_memory_cache', True)))
        print("Use disk chache: {}"\
              .format(kwargs.get('use_disk_cache', False)))
        super(NestBackend, self).init_backend(**kwargs)
        return None

    def _backend_run(self):
        """Run the model via the backend."""
        if self.nest_instance is None:
            raise AttributeError("You need to set the nest instance explicitly. "\
                               + "model._backend.nest_instance = nest")

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
            self.model.simulate(self.model.run_params['simtime'])
        else:
            self.nest_instance.Simulate(self.model.run_params['simtime'])
        endtime = time.time()
        print("Simulation time  : {:.2} s".format(endtime-starttime))
        return self.nest_instance


    def save_results(self, path='.'):
        # ToDo: use NixIO or similar
        """Save results on disk."""
        with open(path, 'wb') as f:
            pickle.dump(self.results, f)
