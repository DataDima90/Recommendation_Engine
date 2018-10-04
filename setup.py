#!/usr/bin/env python

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='neue-fische_data-science',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.1',

    description='Neue Fische - Data Science',

    # The project's main homepage.
    url='',

    # Author details
    author='Matthias Rettenmeier',
    author_email='m.rettenmeier@outlook.com',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Alpha'
    ],

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    #packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    package_dir={'': 'lib'},
    packages=find_packages(where=path.join(here, 'lib')),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['setuptools'],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
         'dev': ['pytest >= 2.8.5',
                 'pytest-mock >= 0.11.0',
                 'pytest-pythonpath >= 0.7'],
    },

    zip_safe=True,
)
