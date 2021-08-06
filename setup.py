from setuptools import setup, Extension
# from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extension = Extension(
    name="ldpc.bp_decoder",
    sources=["src/ldpc/include/mod2sparse.c","src/ldpc/bp_decoder.pyx"],
    libraries=[],
    library_dirs=[],
    include_dirs=[numpy.get_include(),'src/ldpc/include'],
    extra_compile_args=['-std=c11']
    )

extension2 = Extension(
    name="ldpc.mod2sparse",
    sources=["src/ldpc/include/mod2sparse.c","src/ldpc/mod2sparse.pyx"],
    libraries=[],
    library_dirs=[],
    include_dirs=[numpy.get_include(),'src/ldpc/include'],
    extra_compile_args=['-std=c11']
    )

extension3 = Extension(
    name="ldpc.c_util",
    sources=["src/ldpc/c_util.pyx","src/ldpc/include/mod2sparse.c"],
    libraries=[],
    library_dirs=[],
    include_dirs=[numpy.get_include(),'src/ldpc/include'],
    extra_compile_args=['-std=c11']
    )

setup(
    python_requires='>=3.6',
    name='ldpc',
    version='0.0.6',
    description='Python tools for low density parity check (LDPC) codes',
    long_description='This module provides a suite of tools for building and\
        benmarking low density parity check (LDPC) codes. Features include\
        functions for mod2 (binary) arithmatic, tools for constructing quasi-cyclic\
        codes and a fast implementation of the belief propagation decoder ',
    url='https://roffe.eu',
    author='Joschka Roffe',
    packages=["ldpc"],
    package_dir={'':'src'},
    # package_data = {'ldpc': ['*.pxd']},
    ext_modules=cythonize([extension,extension2,extension3]),
    classifiers=['Development Status :: 1 - Planning'],
    install_requires=["tqdm","scipy"],
    include_package_data=True,
    zip_safe=False
)

