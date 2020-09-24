#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cnn_forward, "CNN forward Cpp");
    m.def("backward", &cnn_backward, "CNN backward Cpp");
    m.def("cu_forward", &cnn_cu_forward, "CNN forward CUDA");
    m.def("cu_backward", &cnn_cu_backward, "CNN backward CUDA");
  }