#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

torch::Tensor forward(torch::Tensor input, std::vector<torch::Tensor> parameter){
    torch::Tensor output;
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "cnn forward (CUDA)");
    //m.def("backward", &backward, "cnn backward (CUDA)");
  }