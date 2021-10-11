
## NetworkUnit v0.2 release notes

* improved parameter handling
* improved caching of intermediate test- and simulation results
* various bug fixes
* generate_prediction() and other custom class function no long take optional extra parameter as arguments, but only use self.params
    * no class function should accept arguments that override class parameters
* default_params test class attribute are inherited by using `default_params = {**parent.default_params, 'new_param':0}`
* adding the `joint_test` class enabling to combine multiple neuron-wise tests for multidimensional testing with the Wasserstein score
* new models:
    * --
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
* apply parallelization
    * parallelize correlation matrix calculation??
    * parallelize joint test
* clean up parameter handling and defaults in all tests (e.g. graph)
* unit tests
* more example notebooks (combine with unit tests?)
* merge logic of correlation and covariance tests (avg, std tests)
* consistent handling and setting of correlation/covariance diagonal
    * aligning avg_correlation, and std_correlation test
* how to cache the correlation matrix
* should prediction be quantities?
