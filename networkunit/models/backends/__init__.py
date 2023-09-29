import sciunit.models.backends as su_backends
import inspect

try:
    from .storage import storage
except ImportError:
    storage_backend = None
    print('Could not load storage backend')

try:
    from .nest import NestBackend
except ImportError:
    nest_backend = None
    print('Could not load Nest backend')

available_backends = {x.replace('Backend', ''): cls for x, cls
                      in locals().items()
                      if inspect.isclass(cls) and
                      issubclass(cls, su_backends.Backend)}
su_backends.register_backends(locals())
