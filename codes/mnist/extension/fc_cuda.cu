#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void forward_kernel(
  torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input,
  torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> weight,
  torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> bias,
  torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output
){
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
  int batch = blockIdx.x;
  int img_ = threadIdx.x;
  __shared__ float image[blockDim.x]; // declare s_mem size with kernel launch
  //input size --> blockdim?
  image[threadIdx.x] = input[threadIdx.x]*weight[blockIdx.x] + bias[blockIdx.x]; // fused operation should be used here

  // sequential addition
  __syncthreads();
  if (threadIdx.x==0){
    for(int tid=1;tid<blockDim.x;tid++){
      image[threadIdx.x] += image[threadIdx.x+tid];
    }
    
    output[blockIdx.x] = image[threadIdx.x];
  }
  // reduction
  /*
  __syncthreads();
  for (int step=1; step<input_size;step*=2){
    if ((threadIDx.x%(2*step)==0)&&(threadIdx.x<input_size)){
      image[threadIdx.x] += image[threadIdx.x+step];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0){
    // ReLU
    if (image[threadIdx.x]<0){
      output[blockIdx.x] = 0.0;
    }
    else{
      output[blockIdx.x] = image[threadIdx.x];
    }
  }
  */
}

/*
784 -> 500 -> 400 -> 300 -> 200 -> 100
*/

torch::Tensor forward(torch::Tensor input, std::vector<torch::Tensor> parameter){
  torch::Tensor output = torch::zeros({500});
  
  AT_DISPATCH_FLOATING_TYPES(input.type(), "cuda_forward", ([&] {
    forward_kernel<scalar_t><<<64,784>>>( // block/thread : batchsize/imgsize
      input.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
      parameter[0].packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
      parameter[1].packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
      output.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
    }));
  
    output = torch::nn::functional::relu(output);
    output = torch::nn::functional::linear(output, parameter[2], parameter[3]);
    output = torch::nn::functional::relu(output);
    output = torch::nn::functional::linear(output, parameter[4], parameter[5]);
    output = torch::nn::functional::relu(output);
    output = torch::nn::functional::linear(output, parameter[6], parameter[7]);
    output = torch::nn::functional::relu(output);
    output = torch::nn::functional::linear(output, parameter[8], parameter[9]);
    output = torch::nn::functional::relu(output);
    output = torch::nn::functional::linear(output, parameter[10], parameter[11]);
    output = torch::nn::functional::relu(output);
    output = torch::nn::functional::linear(output, parameter[12], parameter[13]);
    output = torch::nn::functional::softmax(output, -1);
  
    std::cout<<output<<std::endl;
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