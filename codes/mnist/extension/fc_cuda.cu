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
  }

/*
linear
found on torch/csrc/api/include/torch/nn/functional/linear.h
inline Tensor linear(const Tensor& input, const Tensor& weight,
                     const Tensor& bias = {}) {
  if (input.dim() == 2 && bias.defined()) {
    // fused op is marginally faster 
    return torch::addmm(bias, input, weight.t());
  } else {
    auto output = input.matmul(weight.t());
    if (bias.defined()) {
        output += bias;
    }
    return output;
  }
}

relu
found on torch/csrc/api/include/torch/nn/functional/activation.h
*/
