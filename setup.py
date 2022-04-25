from setuptools import setup, Extension
# from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

VERSION="0.1.5"
f=open("src/ldpc/VERSION","w+")
f.write(VERSION)
f.close()

from shutil import copyfile
files=["README.md","LICENSE"]
for f in files:
    copyfile(f,"src/ldpc/"+f)

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
    python_requires='>=3.6',
    name='ldpc',
    version=VERSION,
    description='Python tools for low density parity check (LDPC) codes',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/quantumgizmos/ldpc',
    author='Joschka Roffe',
    packages=["ldpc"],
    package_dir={'':'src'},
    ext_modules=cythonize([extension,extension2,extension3,extension4]),
    classifiers=['Development Status :: 4 - Beta'],
    install_requires=["tqdm","scipy",f"numpy=={numpy.__version__}"],
    include_package_data=True,
    zip_safe=False
)

