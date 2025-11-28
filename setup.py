from setuptools import setup
from Cython.Build import cythonize
import numpy

# This script tells Python how to find and compile the .pyx file.
setup(
    ext_modules=cythonize("src/mcts/node.pyx"),
    include_dirs=[numpy.get_include()]
)