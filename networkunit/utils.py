def generate_prediction_wrapper(generate_prediction_func):
    """
    Wraps the `generate_prediction()` function of the tests, handles cached
    prediction loading, parameter update and prediction saving.
    """
    def wrapper(self, model, **kwargs):

        # Check if predictions were already calculated
        prediction = self.get_prediction(model)

        if prediction is None:

            # Use user determined parameters without overwriting attribute
            if kwargs:
                params = {**self.params, **kwargs}
            else:
                params = self.params

            # Generate and save prediction
            prediction = generate_prediction_func(self, model, **params)
            self.set_prediction(model, prediction)

        return prediction

    return wrapper
