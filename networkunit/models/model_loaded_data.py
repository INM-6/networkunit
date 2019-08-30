import sciunit
from .backends import available_backends

class loaded_data(sciunit.Model):
    """
    Abstract model class which initializes self.params.
    Child classes need to define a load() function and a file_path.
    """
    @property
    def file_path(self):
        raise NotImplementedError

    def __init__(self, name=None, backend='storage', **params):
        """
        Parameters
        ----------
        name : string
            Name of model instance
        **params :
            class attributes to be stored in self.params
        """
        if params is None:
            params = {}
        if hasattr(self, 'params'):
            self.params.update(params)
        else:
            self.params = params
        super(loaded_data, self).__init__(name=name, **self.params)
        if backend is not None:
            self.set_backend(backend)

    def load(self, file_path=None, **kwargs):
        """
        To be called in the produce_xy() function,
        associated with the capability class ProducesXY.
        """
        raise NotImplementedError

    def get_backend(self):
        """Return the simulation backend."""
        return self._backend

    def set_backend(self, backend):
        """Set the simulation backend."""
        if isinstance(backend, str):
            name = backend
            args = []
            kwargs = {}
        elif isinstance(backend, (tuple, list)):
            name = ''
            args = []
            kwargs = {}
            for i in range(len(backend)):
                if i == 0:
                    name = backend[i]
                else:
                    if isinstance(backend[i], dict):
                        kwargs.update(backend[i])
                    else:
                        args += backend[i]
        else:
            raise TypeError("Backend must be string, tuple, or list")
        if name in available_backends:
            self.backend = name
            self._backend = available_backends[name]()
        elif name is None:
            # The base class should not be called.
            raise Exception(("A backend must be selected"))
        else:
            raise Exception("Backend %s not found in backends" % name)
        self._backend.model = self
        self._backend.init_backend(*args, **kwargs)
