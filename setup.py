"""
#
# SinSH setup.py
#
# Copyright(c) 2020, Antoine Emil Zambelli.
#
"""

from setuptools import setup, find_packages


version = '0.1.1'

setup(
    name='sinsh',
    version=version,
    url='https://github.com/antoinezambelli/SinSH',
    license='MIT',
    author='Antoine Zambelli',
    author_email='antoine.zambelli@gmail.com',
    description='Simple Python SSH Library',
    long_description='Convenience wrappers for SSH in Python and bonus batch processing class',
    packages=find_packages(
        exclude=(
           '.*',
           'EGG-INFO',
           '*.egg-info',
           '_trial*',
           "*.tests",
           "*.tests.*",
           "tests.*",
           "tests",
           "examples.*",
           "examples",
        )
    ),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'tqdm'
    ]
)
