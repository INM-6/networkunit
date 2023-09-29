# -*- coding: utf-8 -*-
"""
NetworkUnit is a SciUnit library for validation testing of neural network models.

:copyright: Copyright2018 by the NetworkUnit team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

from . import tests, models, capabilities, scores, plots

def _get_version():
    import os
    networkunit_dir = os.path.dirname(__file__)
    with open(os.path.join(networkunit_dir, 'VERSION')) as version_file:
        version = version_file.read().strip()
    return version

__version__ = _get_version()
