import inspect
from elephant.parallel import SingleProcess


def use_prediction_cache(generate_prediction_func):
    """
    Decorator for the `generate_prediction()` function of the tests, handles
    cached prediction loading, parameter update and prediction saving.
    """
    def wrapper(self, model):

        # Check if predictions were already calculated
        prediction = self.get_prediction(model)

        # If any parameter was specified by the user in the generate_prediction function the predictions are recalculated
        if prediction is None:

            # Generate and save prediction
            prediction = generate_prediction_func(self, model)
            self.set_prediction(model, prediction)

        return prediction

    return wrapper


class filter_valid_params:
    """
    Context manager that enables to pass any non-valid arguments to a function
    that are subsequently ignored.
    """
    def __init__(self, func):
        self.func = func

    def __enter__(self):
        sig = inspect.signature(self.func)
        filter_keys = [param.name for param in sig.parameters.values()
                       if param.kind == param.POSITIONAL_OR_KEYWORD]

        def _func(*args, **kwargs):
            filtered_kwargs = {filter_key:kwargs[filter_key] for filter_key in filter_keys
                               if filter_key in kwargs.keys()}
            return self.func(*args, **filtered_kwargs)

        return _func

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass
        return False


class parallelize:
    """
    Context manager that applies elephant.parallel executors.
    Default executor: SingleProcess()

    Example:
    ```
    results = [my_function(arg) for arg in iterables_list]
    ```
    becomes
    ```
    with parallelize(my_function) as parallel_func:
        results = parallel_func(iterables_list, **kwargs)
    ```
    """
    def __init__(self, func, test_class=None, executor=SingleProcess()):
        self.executor = executor
        self.func = func

        if test_class is not None and'parallel_executor' in test_class.params:
            self.executor = test_class.params['parallel_executor']

    def __enter__(self):

        def _func(iterable_list, **kwargs):

            def arg_func(iterable_list):
                return self.func(iterable_list, **kwargs)

            result = self.executor.execute(arg_func, iterable_list)
            return result

        return _func

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass
