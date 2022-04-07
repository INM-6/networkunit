
## NetworkUnit v0.2 release notes

* parameter handling
    * generate_prediction() and other custom class function no long take optional extra parameter as arguments, but only use self.params
    * no class function should accept arguments that override class parameters
    * default_params test class attribute are inherited by using `default_params = {**parent.default_params, 'new_param':0}`
* caching
    * improved caching of intermediate test- and simulation results, e.g. for the correlation matrix
    * improving backend definitions
* parallelization
    * automatic parallelization for loops over spiketrains or lists of spiketrains. To use set params['parallel executor'] to ProcessPoolExecutor(), MPIPoolExecutor(), or MPICommExecutor() [see elephant ref]
* various bug fixes
* new features
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
* merge logic of correlation and covariance tests (avg, std tests)
* consistent handling and setting of correlation/covariance diagonal
    * aligning avg_correlation, and std_correlation test
* should prediction be quantities?
* clean up parameter handling and defaults in all tests (e.g. graph)
* more example notebooks (combine with unit tests?)
* unit tests
* update readmes

#### SciUnit PR
* create helper function `get_cache` and `set_cache` to handle both disk and memory
* create decorator to use cached results from function
* create a decorated `_generate_prediction` function that interfaces between `judge` and `generate_prediction`
    * calling `generate_prediction` in any test would still recalculate, only `_generate_prediction` uses the cache
* add solution for models without backend
