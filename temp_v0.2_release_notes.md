
## NetworkUnit v0.2 release notes

* improved parameter handling
* improved caching of intermediate test- and simulation results
* various bug fixes
* generate_prediction() and other custom class function no long take optional extra parameter as arguments, but only use self.params
    * no class function should accept arguments that override class parameters
* default_params test class attribute are inherited by using `default_params = {**parent.default_params, 'new_param':0}`
* adding the `joint_test` class enabling to combine multiple neuron-wise tests for multidimensional testing with the Wasserstein score
* new models:
    * brunel
    * cortical microcircuit (ref)
* new tests:
    * joint_test
    * power_spectrum
        * freqband
    * timescale
    * avg_correlation
    * std_correlation
* new scores:
    * Wasserstein
    * Eigenangles

### ToDo:
* parallelization (use Elephant feature?)
* unit tests
* more example notebooks (combine with unit tests?)
