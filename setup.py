from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import Cython.Compiler.Options
import numpy

#  from setuptools import setup
# from setuptools import Extension
# from Cython.Build import cythonize
# import numpy

extension = Extension(
    name="ldpc.bp_decoder",
    sources=["include/mod2sparse.c","src/ldpc/bp_decoder.pyx"],
    libraries=[],
    library_dirs=[],
    include_dirs=[numpy.get_include(),'include'],
    extra_compile_args=['-std=c11']
    )

extension2 = Extension(
    name="ldpc.mod2sparse",
    sources=["include/mod2sparse.c","src/ldpc/mod2sparse.pyx"],
    libraries=[],
    library_dirs=[],
    include_dirs=[numpy.get_include(),'include'],
    extra_compile_args=['-std=c11']
    )

extension3 = Extension(
    name="ldpc.c_util",
    sources=["src/ldpc/c_util.pyx","include/mod2sparse.c"],
    libraries=[],
    library_dirs=[],
    include_dirs=[numpy.get_include(),'include'],
    extra_compile_args=['-std=c11']
    )

setup(
    python_requires='>=3.7',
    name='ldpc',
    version='0.0.1',
    description='',
    long_description='A belief propagation decoder for low density parity check (LDPC) codes',
    url='https://roffe.eu',
    author='Joschka Roffe',
    packages=["ldpc"],  #same as name
    package_dir={'':'src'},
    package_data = {'ldpc': ['*.pxd']},
    ext_modules=cythonize([extension,extension2,extension3]),
    classifiers=['Development Status :: 1 - Planning'],
    install_requires=["tqdm","scipy"],
    include_package_data=True,
    zip_safe=False
)

