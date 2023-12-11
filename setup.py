from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

VERSION="0.1.51"
f=open("src/ldpc/VERSION","w+")
f.write(VERSION)
f.close()

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
    sources=["src/ldpc/include/mod2sparse.c","src/ldpc/include/mod2sparse_extra.c","src/ldpc/mod2sparse.pyx"],
    libraries=[],
    library_dirs=[],
    include_dirs=[numpy.get_include(),'src/ldpc/include'],
    extra_compile_args=['-std=c11']
    )

extension3 = Extension(
    name="ldpc.c_util",
    sources=["src/ldpc/c_util.pyx","src/ldpc/include/mod2sparse.c","src/ldpc/include/binary_char.c","src/ldpc/include/sort.c"],
    libraries=[],
    library_dirs=[],
    include_dirs=[numpy.get_include(),'src/ldpc/include'],
    extra_compile_args=['-std=c11']
    )

extension4 = Extension(
    name="ldpc.osd",
    sources=["src/ldpc/osd.pyx","src/ldpc/include/mod2sparse.c","src/ldpc/include/mod2sparse_extra.c","src/ldpc/include/binary_char.c","src/ldpc/include/sort.c"],
    libraries=[],
    library_dirs=[],
    include_dirs=[numpy.get_include(),'src/ldpc/include'],
    extra_compile_args=['-std=c11']
    )

setup(
    version=VERSION,
    ext_modules=cythonize([extension,extension2,extension3,extension4]),
)
