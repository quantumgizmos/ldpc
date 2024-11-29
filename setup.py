from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
from pathlib import Path
import sys
import os
import re

## cython stub files

def generate_cython_stub_file(pyx_filepath: str, output_filepath: str) -> None:
    pyi_content = ""

    # load file contents
    pyx_content = open(pyx_filepath, "r").read()

    # strip cython syntax, empty lines, and comments
    pyx_content = re.sub("cdef ", "", pyx_content)
    pyx_content = re.sub(r"^\s*\n", "", pyx_content, flags=re.MULTILINE)
    pyx_content = re.sub(r"^\s*#.*\n", "", pyx_content, flags=re.MULTILINE)

    # identify top-level import lines
    pattern = re.compile(r"^(import|from)\s+.*\n", re.MULTILINE)
    for match in pattern.finditer(pyx_content):
        pyi_content += pyx_content[match.start():match.end()]

    # identify patterns to ignore
    ignore_pattern = re.compile(r"__cinit__\(|__del__\(")

    # identify class or function declarations
    decorator = r"^\s*@.*\n"
    declaration = r"^\s*(?:class|def)\s+.*(?:.|\n)\n"
    docstring_double = r"\"\"\"(?:.|\n)*?\"\"\""
    docstring_single = r"'''(?:.|\n)*?'''"
    docstring = rf"\s*(?:{docstring_double}|{docstring_single})\s*\n"
    pattern = re.compile(rf"({decorator})?({declaration})({docstring})?", re.MULTILINE)
    for match in pattern.finditer(pyx_content):
        content = pyx_content[match.start():match.end()]
        if not ignore_pattern.match(content, re.MULTILINE):
            pyi_content += content.rstrip()
            if match.group(3):
                # there is a docstring!
                pyi_content += "\n"
            else:
                # there is no docstring
                pyi_content += " ...\n"

    open(output_filepath, "w").write(pyi_content)

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
