
## NetworkUnit v0.2 release notes

* improved parameter handling
* improved caching of intermediate test- and simulation results
* various bug fixes
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
* parallelization
* unit tests
* more example notebooks (combine with unit tests?)
