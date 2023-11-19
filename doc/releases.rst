*************
Release Notes
*************

NetworkUnit 0.2.0
=================
- parameter handling
    - `generate_prediction()` and other custom class function no longer take optional extra parameter as arguments, but only use `self.params`
    - no class function should accept arguments that override class parameters
    - `default_params` test class attribute are inherited by using `default_params = {--parent.default_params, 'new_param':0}`
- caching
    - improved caching of intermediate test- and simulation results, e.g. for the correlation matrix
    - improving backend definitions
- parallelization
    - automatic parallelization for loops over spiketrains or lists of spiketrains. To use set `params['parallel executor']` to `ProcessPoolExecutor()`, `MPIPoolExecutor()`, or `MPICommExecutor()` (see [documentation in Elephant package](https://elephant.readthedocs.io/en/latest/reference/parallel.html))
- various bug fixes
- new features
    - adding the `joint_test` class that enables the combination of multiple neuron-wise tests for multidimensional testing with the Wasserstein score
- new test classes
    - joint_test
    - power_spectrum_test
        - freqband_power_test
    - timescale_test
    - avg_std_correlation_test
- new score classes
    - wasserstein_distance
    - eigenangle (see publication [Gutzen et al. 2022](https://doi.org/10.1016/j.biosystems.2022.104813)


NetworkUnit 0.1.2
=================
- a fix for an issue where the setup script was failing to properly install the backend directory (see issue #20)


NetworkUnit 0.1.1
=================
- a new backend class, which handles the storage of generated predictions in memory or on disk. To make use of it just set `backend='storage'` in the model instantiation. By default predictions are stored in memory. To change that set `model.get_backend().use_disk_cache = True ` and `model.get_backend().use_memory_cache = False `.
- various bug fixes
- updated requirements.txt and environment.yaml


NetworkUnit 0.1.0
=================
Initial release.