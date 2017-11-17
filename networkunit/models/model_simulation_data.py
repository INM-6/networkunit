import sciunit


class simulation_data(sciunit.Model):
    """
    A data model is representation of the experimental observation. But instead
    of containing the observation data to validate against, the data is in the
    same preprocessed form as the outcome of the model simulation.
    This requires the __init__ function of the test class to generate the
    observation data from the data_model instance.

    Minimal example of such an __init__ function:
    def __init__(self, reference_data, name=None, **params):
        observation = self.generate_prediction(reference_data)
        super(test_class_name,self).__init__(observation, name=name, **params)

    The use of a data_model enables to perfom the data analysis step more
    equivalently on both the experimental data and the simulation data.
    """

    def __init__(self, file_path, name=None, **params):
        self.data = self.load(file_path, **params)
        super(simulation_data, self).__init__(name=name, **params)

    def load(self, file_path, **kwargs):
        # ToDo: write generic loading routine
        data = None
        return data