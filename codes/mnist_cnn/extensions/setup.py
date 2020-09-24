import sys
sys.path.append('/home/hyuck2/.local/lib/python3.8/site-packages')
from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='mnist_cnn',
      ext_modules=[cpp_extension.CppExtension('mnist_cnn', ['mnist_cnn.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})