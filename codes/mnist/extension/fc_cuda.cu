#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void forward_kernel(){
  /*
  input
  weight
  bias

  fused multiply add operation
  
  batch size
  input
    784 -> 500
  weight : 500
  bias : 500
  parameter --> output size
  
  block
  - batch size
  - parameter size

  thread
  - input

  
  
  fma result should be added --> in shared memory + parallel reduction?


  
  */
}

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