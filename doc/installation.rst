************
Installation
************

The easiest way to install NetworkUnit is by creating a conda environment, followed by ``pip install networkunit``.
Below is the explanation of how to proceed with these two steps.

Prerequisites
-------------

Create new conda environment:

.. code:: bash

    conda create --name networkunit python=3.8

Activate conda environment:

.. code:: bash

    conda activate networkunit

Installation
------------

Install NetworkUnit to the currently activated environment:

.. code:: bash

    pip install networkunit

Requirements
------------

.. code:: bash

    elephant >= 0.7.0
    neo >= 0.9
    sciunit >= 0.2.2
    jupyter >= 1.0.0
    tabulate >= 0.8.2
    networkx >= 2.1
    seaborn >= 0.8.1
    numpy >= 1.15.3
    odml >= 1.3.0
    opencv-python