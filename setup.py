# -*- coding: utf-8 -*-

from setuptools import setup
import os
import sys

long_description = open("README.rst").read()
with open('requirements.txt', 'r') as fp:
    install_requires = fp.read()
extras_require = {}
for extra in ['docs']:
    with open('requirements-{0}.txt'.format(extra)) as fp:
        extras_require[extra] = fp.read()

setup(
    name="networkunit",
    version='0.2.0',
    packages=['networkunit', 'figures', 'examples'],
    package_data={'networkunit':[
        os.path.join('tests','*.py'),
        os.path.join('models','*.py'),
        os.path.join('models/backends','*.py'),
        os.path.join('capabilities','*.py'),
        os.path.join('scores','*.py'),
        os.path.join('plots','*.py'),
        'VERSION'],
        'figures':['*.png'],
        'examples':['*.py', '*.ipynb']
        },
    install_requires=install_requires,
    extras_require=extras_require,

    author="NetworkUnit authors and contributors",
    author_email="r.gutzen@fz-juelich.de",
    description="A SciUnit library for validation testing of neural network models.",
    long_description=long_description,
    license="BSD",
    url='https://github.com/INM-6/NetworkUnit',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering']
)
