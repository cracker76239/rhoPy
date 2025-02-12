import math
import random
from setuptools import setup, find_packages # type: ignore


setup(
    name="statsLib",
    version="0.1",
    packages=find_packages(),
    install_requires=[random, math],
)