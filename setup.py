import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy

if os.name == 'nt':
    ext_modules = [
        Extension(
            "_BLP",
            ["_BLP.pyx"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=['/openmp'],
        )
    ]
else:
    ext_modules = [
        Extension(
            "_BLP",
            ["_BLP.pyx"],
            libraries=["m"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp'],
        )
    ]

setup(
    name='pyBLP',
    ext_modules=cythonize(ext_modules),
)
