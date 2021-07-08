from setuptools import setup
from distutils.extension import Extension
import numpy

from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.docstrings = False
cython_installed=True

source_files=[
      "include/mod2sparse.c"]

if cython_installed:
   source_files = ["src/pybp/bp_decoder.pyx"] + source_files
# else:
#    source_files = ["include/bp_osd_cython/cypybp.c"] + source_files

extension = Extension(
    name="cybp",
    sources=source_files,
    libraries=[],
    library_dirs=[],
    include_dirs=[numpy.get_include(),"include"],
    extra_compile_args=['-std=c11']
    )

def ext_class(extension, cython_installed):
   if cython_installed:
      return cythonize([extension])
   else:
      return [extension]

setup(
    python_requires='>=3.6',
    name='pybp',
    version='0.0.1',
    description='',
    long_description='A belief propagation decoder for decoding low density parity check (LDPC)',
    url='https://www.roffe.eu',
    author='Joschka Roffe',
    author_email='joschka@roffe.eu',
    packages=["pybp"],  #same as name
    package_dir={'':'src'},
    ext_modules=cythonize([extension]),
    classifiers=['Development Status :: 1 - Planning'],
    install_requires=["tqdm","scipy"]
)

