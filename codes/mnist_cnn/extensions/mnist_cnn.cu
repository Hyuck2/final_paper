#include <iostream>
#include <vector>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

std::vector<at::Tensor> cnn_forward();
std::vector<at::Tensor> cnn_backward();
std::vector<at::Tensor> cnn__cu_forward();
std::vector<at::Tensor> cnn_cu_backward();

__global__ void cnn_cu_forward_kernel(){

}
__global__ void cnn_cu_backward_kernel(){
  
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cnn_forward, "CNN forward Cpp");
    m.def("backward", &cnn_backward, "CNN backward Cpp");
    m.def("cu_forward", &cnn_cu_forward, "CNN forward CUDA");
    m.def("cu_backward", &cnn_cu_backward, "CNN backward CUDA");
  }