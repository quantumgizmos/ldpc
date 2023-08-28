from distutils import extension
from setuptools import setup, Extension, find_namespace_packages
# from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
from pathlib import Path
import sys
import os
if sys.platform == "darwin":
    os.environ["CC"] = "gcc-12"
    os.environ["CXX"] = "g++-12"

this_directory = Path(__file__).parent
# long_description = (this_directory / "README.md").read_text()

# VERSION="0.0.1"
# f=open("VERSION","w+")
# f.write(VERSION)
# f.close()

# from shutil import copyfile
# files=["README.md","LICENSE"]
# for f in files:
#     copyfile(f,"src/ldpc/"+f)




cpp_modules = ["bp_decoder","bposd_decoder","mbp_decoder",
                "uf_decoder","bf_decoder","bp_decoder2","gf2sparse"]

cpp_modules = ["gf2sparse", "bp_decoder", "bposd_decoder"]

c_extensions = []
for module in cpp_modules:

    c_extensions.append(Extension(
        name=f"ldpc2.{module}._{module}",
        sources=[f"src/ldpc2/{module}/_{module}.pyx"],
        libraries=[],
        library_dirs=[],
        include_dirs=[numpy.get_include(),'src_cpp', 'src_cpp/include/robin_map'],
        extra_compile_args=['-std=c++2a', '-O3','-fopenmp'],
        extra_link_args=['-lgomp','-fopenmp'],
        language="c++"
    ))



setup(
    python_requires='>=3.6',
    name='ldpc2',
    # version=VERSION,
    description='ldpc2 Decoder',
    # long_description=long_description,
    # long_description_content_type='text/markdown',
    # url='https://github.com/quantumgizmos/ldpc',
    author='Joschka Roffe',
    packages=find_namespace_packages(where="src",include=['ldpc2','ldpc2.*']),
    package_dir={'':'src'},
    ext_modules=cythonize(c_extensions,
    compiler_directives={'boundscheck': False,'wraparound': False,
                        'initializedcheck':False,'cdivision':True,'embedsignature': True}),
    classifiers=['Development Status :: 4 - Beta'],
    # install_requires=["tqdm","scipy",f"numpy=={numpy.__version__}"],
    include_package_data=True,
    zip_safe=False
)
