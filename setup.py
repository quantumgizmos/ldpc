# from setuptools import setup
# from distutils.extension import Extension
# import numpy

# from Cython.Build import cythonize
# import Cython.Compiler.Options
# Cython.Compiler.Options.docstrings = False

from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
import numpy

# setup(
#     name = 'pybp',
#     version='1.0.0',
#     author='Cython Demo',
#     url='http://example.com',
#     packages = ['pybp'],
#     ext_modules = cythonize(Extension(name="pybp.c_math", sources=["src/pybp/c_math.pyx"],include_dirs=[numpy.get_include()])),
#     package_dir={'':'src'},
#     package_data = {'pybp': ['c_math.pxd']},
#     include_package_data=True,
#     setup_requires=["cython >= 0.26"],
#     zip_safe=False
#     )


extension = Extension(
    name="pybp.bp_decoder",
    sources=["include/mod2sparse.c","src/pybp/bp_decoder.pyx"],
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
    ext_modules=cythonize([extension,extension2,extension3]),
    classifiers=['Development Status :: 1 - Planning'],
    install_requires=["tqdm","scipy"],
    include_package_data=True,
    zip_safe=False
)

