from setuptools import setup
from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='example.cpp',
    ext_modules=[cpp_extension.CppExtension('example.cpp', [example.cpp])],
    cmdclass = {'build_ext':cpp_extension.BuildExtension})
