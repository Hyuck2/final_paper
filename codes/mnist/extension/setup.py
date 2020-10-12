import sys
sys.path.append('/home/hyuck2/.local/lib/python3.8/site-packages')
from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='fc_cpp',
      ext_modules=[cpp_extension.CppExtension('fc_cpp', ['fc_cpp.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

setup(name='cnn_cpp',
      ext_modules=[cpp_extension.CppExtension('cnn_cpp', ['cnn_cpp.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

setup(name='cnn_cuda',
      ext_modules=[cpp_extension.CppExtension('cnn_cuda', ['cnn_cuda.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})