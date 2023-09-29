from sciunit.models.backends import Backend
import os

class storage(Backend):
    name = 'storage'

    def _backend_run(self):
        try:
            data = self.model.load()
        except Exception as e:
            print(e)
            raise NotImplementedError('Hint: the model class must define a '
                                      'load() function!')
        return data

    def load_model(self) -> None:
        """Load the model into memory."""

    def set_attrs(self, **attrs) -> None:
        """Set model attributes on the backend."""

    def set_run_params(self, **run_params) -> None:
        """Set model attributes on the backend."""
