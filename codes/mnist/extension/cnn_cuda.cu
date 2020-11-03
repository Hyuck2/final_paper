#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void cuda_forward_kernel(){

}


torch::Tensor forward(torch::Tensor input, std::vector<torch::Tensor> parameter){
    torch::Tensor output;

    // AT_DISPATCH_FLOATING_TYPES(,"forward",([&] {cuda_forward_kernel<scalar_t><<<>>>();}));
    

    output = squeeze(output);
    
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "cnn forward (CUDA)");
}