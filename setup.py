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
    name="pybp.bp_decoder",
    sources=["include/mod2sparse.c","src/pybp/bp_decoder.pyx"],
    libraries=[],
    library_dirs=[],
    include_dirs=[numpy.get_include(),'include'],
    extra_compile_args=['-std=c11']
    )

extension4 = Extension(
    name="pybp.bp_decoder2",
    sources=["include/mod2sparse.c","src/pybp/bp_decoder2.pyx"],
    libraries=[],
    library_dirs=[],
    include_dirs=[numpy.get_include(),'include'],
    extra_compile_args=['-std=c11']
    )

extension2 = Extension(
    name="pybp.mod2sparse",
    sources=["include/mod2sparse.c","src/pybp/mod2sparse.pyx"],
    libraries=[],
    library_dirs=[],
    include_dirs=[numpy.get_include(),'include'],
    extra_compile_args=['-std=c11']
    )

extension3 = Extension(
    name="pybp.c_util",
    sources=["src/pybp/c_util.pyx","include/mod2sparse.c"],
    libraries=[],
    library_dirs=[],
    include_dirs=[numpy.get_include(),'include'],
    extra_compile_args=['-std=c11']
    )

setup(
    python_requires='>=3.7',
    name='pybp',
    version='0.0.1',
    description='',
    long_description='A belief propagation decoder for low density parity check (LDPC) codes',
    url='https://roffe.eu',
    author='Joschka Roffe',
    packages=["pybp"],  #same as name
    package_dir={'':'src'},
    package_data = {'pybp': ['*.pxd']},
    ext_modules=cythonize([extension,extension2,extension3,extension4]),
    classifiers=['Development Status :: 1 - Planning'],
    install_requires=["tqdm","scipy"],
    include_package_data=True,
    zip_safe=False
)

