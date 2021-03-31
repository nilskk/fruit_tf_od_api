import os
from setuptools import find_packages, setup

setup(
    name='fruitod',
    install_requires=['seaborn',
                      'lxml',
                      'numpy',
                      'pandas'],
    include_package_data=True,
    packages=([p for p in find_packages() if p.startswith('fruitod')])
)