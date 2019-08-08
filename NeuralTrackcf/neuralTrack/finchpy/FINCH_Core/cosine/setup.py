# build the modules
from distutils.core import setup, Extension

setup(name="matSimilarity", version="1.0",
      ext_modules=[Extension("matSimilarity", sources=["pycossim.cpp"])])