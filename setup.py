from setuptools import setup
from distutils.extension import Extension
import numpy

from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.docstrings = False

extension = Extension(
    name="cybp",
    sources=["include/mod2sparse.c","src/pybp/bp_decoder.pyx"],
    libraries=[],
    library_dirs=[],
    include_dirs=[numpy.get_include(),"include"],
    extra_compile_args=['-std=c11']
    )

setup(
    python_requires='>=3.7',
    name='pybp',
    version='0.0.1',
    description='',
    long_description='A belief propagation decoder for low density parity check (LDPC) codes',
    url='https://www.roffe.eu',
    author='Joschka Roffe',
    author_email='joschka@roffe.eu',
    packages=["pybp"],  #same as name
    package_dir={'':'src'},
    ext_modules=cythonize([extension]),
    classifiers=['Development Status :: 1 - Planning'],
    install_requires=["tqdm","scipy"]
)

