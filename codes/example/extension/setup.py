import sys
sys.path.append('/home/hyuck2/.local/lib/python3.8/site-packages')
from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='lltm_extension',
      ext_modules=[cpp_extension.CppExtension('lltm_extension', ['lltm_extension.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})