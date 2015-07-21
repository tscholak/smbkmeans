# -*- coding: utf-8 -*-
from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

tfidf_module = Extension(
        "tfidf_module",
        ["tfidf_module.pyx"],
        extra_compile_args=['-O3', '-fopenmp'],
        extra_link_args=['-fopenmp'],)

setup(cmdclass = {'build_ext': build_ext},
      ext_modules = [tfidf_module,],
      include_dirs = [numpy_include,],)
