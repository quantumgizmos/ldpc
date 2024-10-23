from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
from pathlib import Path
import sys
import os
import re

## cython stub files

def generate_cython_stub_file(pyx_filepath: str, output_filepath: str) -> None:
    with open(pyx_filepath, 'r') as f:
        pyx_content = f.read()

    # Match function, class, and method definitions, and cdef/cpdef/cclass declarations
    pattern = re.compile(r'(cdef class|class|cpdef|def)\s+[\w\[\],\s\*&\<\>\=\:]*')
    # cimport_pattern = re.compile(r'from\s+libc\..+\s+cimport\s+')

    # Split by lines and filter out lines without definitions
    lines = pyx_content.split('\n')
    new_lines = []
    inside_docstring = False
    inside_function = False

    for line in lines:

        if "__cinit__(" in line:
            continue

        if "__del__(" in line:
            continue

        line = line.replace('cdef class', 'class')

        stripped_line = line.strip()

        # Remove comments
        if stripped_line.startswith('#'):
            continue
        
        # # Skip cimport statements
        # if stripped_line.startswith('cimport'):
        #     new_lines.append(line)
        #     continue

        # if cimport_pattern.match(stripped_line):
        #     new_lines.append(line)
        #     continue

        # Include import statements
        if stripped_line.startswith('import') or stripped_line.startswith('from') or stripped_line.startswith('cimport'):
            new_lines.append(line)
            continue

        # Handle docstrings
        if inside_function:
            if stripped_line.startswith('"""') or stripped_line.startswith("'''"):
                new_lines.append(line)
                # print(line)
                inside_docstring = not inside_docstring
                if not inside_docstring:
                    inside_function = False
                    new_lines.append('\n')
                continue

            elif inside_docstring:
                new_lines.append(line)
                # print(line)
                continue

        # Handle decorators
        if stripped_line.startswith('@'):
            decorator_found = True
            new_lines.append(line)
            continue


        # Handle function and class definitions
        if pattern.match(stripped_line):
            print(stripped_line)

            new_lines.append(line)
            inside_function = True
            # new_lines.append('\n')
            continue

    # Write the stripped content to an output file
    with open(output_filepath, 'w') as f:
        f.write('\n'.join(new_lines))


## BUILD

if sys.platform == "darwin":
    os.environ["CC"] = "clang"
    os.environ["CXX"] = "clang++"

if sys.platform == "win32":
    compile_flags = ["/Ox", "/std:c++20"]
    extra_link_args =[]
    # compile_flags = ["/Ox", "/std:c++20",'-fopenmp']
    # extra_link_args =['-lgomp','-fopenmp'],
else:
    compile_flags = ["-std=c++2a", "-O3"]
    extra_link_args =[]
    # compile_flags = ["-std=c++2a", "-O3", "-fopenmp"]
    # extra_link_args =['-lgomp','-fopenmp'],

this_directory = Path(__file__).parent

cpp_modules = ["bp_decoder", "bposd_decoder", "bp_flip", "belief_find_decoder", "mod2", "union_find_decoder", "bplsd_decoder"]

c_extensions = []
for module in cpp_modules:

    generate_cython_stub_file(f"src_python/ldpc/{module}/_{module}.pyx", f"src_python/ldpc/{module}/__init__.pyi")

    c_extensions.append(
        Extension(
            name=f"ldpc.{module}._{module}",
            sources=[f"src_python/ldpc/{module}/_{module}.pyx"],
            libraries=[],
            library_dirs=[],
            include_dirs=[np.get_include(),'src_cpp', 'include/robin_map','include/ldpc/src_cpp'],
            extra_compile_args=compile_flags,
            extra_link_args=extra_link_args,
            language="c++",
        )
    )


setup(
    ext_modules=cythonize(
        c_extensions,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
            "embedsignature": True,
        },
    ),
)