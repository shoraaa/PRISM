from setuptools import Extension, setup
import sys

import pybind11


if sys.platform == "darwin":
    extra_compile_args = ["-O3", "-std=c++17", "-Xpreprocessor", "-fopenmp"]
    extra_link_args = ["-lomp"]
else:
    extra_compile_args = ["-O3", "-std=c++17", "-fopenmp"]
    extra_link_args = ["-fopenmp"]

ext_modules = [
    Extension(
        "prism_decoder",
        ["src/binding.cpp", "src/decoder.cpp", "src/legacy_schema.cpp"],
        include_dirs=[
            pybind11.get_include(),
            "src",
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
]

setup(
    name="prism_decoder",
    version="1.0.0",
    ext_modules=ext_modules,
)
