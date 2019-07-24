
from sciunit.models.backends import Backend
import os

class storage(Backend):
    name = 'storage'

    def init_backend(self, *args, **kwargs):
        """Initialize the backend."""
        self.model.attrs = {}

        self.use_memory_cache = kwargs.get('use_memory_cache', True)
        if self.use_memory_cache:
            self.init_memory_cache()
        self.use_disk_cache = kwargs.get('use_disk_cache', False)
        if self.use_disk_cache:
            self.init_disk_cache()
        self.load_model()
        self.model.unpicklable += ['_backend']

    def _backend_run(self):
        if not os.path.exists(self.model.file_path):
            raise NotImplementedError('The model class must specify a \
            file_path from which the simulation results can be loaded.')
        return self.model.load()
