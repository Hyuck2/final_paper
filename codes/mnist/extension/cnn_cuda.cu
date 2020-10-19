#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void conv2d_relu_kernel(
    torch::Tensor input, 
    torch::Tensor output, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    int stride, 
    int padding,
    int width,
    int height){
    // conv2d
    __shared__ conv_out[];
    
    // plus bias
    // reduction
    __syncthreads();
    for (int step = 1; step < 64; step*=2){
        if ((threadIdx.x % (2* step) == 0) && (threadIdx.x < 64)){
            conv_out[threadIdx.x] += thread_result[threadIdx.x + step];
        }
        __syncthreads();
    }
    if (threadsIdx.x == 0){

    }

    // ReLU

}

torch::Tensor forward(torch::Tensor input, std::vector<torch::Tensor> parameter){
    torch::Tensor output;
    torch::Tensor* d_input;
    torch::Tensor* d_output;
    torch::Tensor* d_weight;
    torch::Tensor* d_bias;
    cudaMalloc((void**)&d_input, *sizeof(torch::Tensor));
    cudaMalloc((void**)&d_output, *sizeof(torch::Tensor));
    cudaMalloc((void**)&d_weight, *sizeof(torch::Tensor));
    cudaMalloc((void**)&d_bias, *sizeof(torch::Tensor));
    cudaMemcpy(,,,cudaMemcpyHostToDevice); // copy input to CUDA
    cudaMemcpy(,,,cudaMemcpyHostToDevice); // copy input to CUDA
    cudaMemcpy(,,,cudaMemcpyHostToDevice); // copy weight to CUDA
    cudaMemcpy(,,,cudaMemcpyHostToDevice); // copy bias to CUDA


    
    
    output = squeeze(output);
    // squeeze needed
    cudaMemcpy(,,,cudaMemcpyDeviceToHost); // return from CUDA output size: 10 * sizeof(torch::Tensor)
    output = torch::nn::functional::linear(output, parameter[20], parameter[21]); // 50, 50
    output = torch::nn::functional::relu(output);
    output = torch::nn::functional::linear(output, parameter[22], parameter[23]); // 10, 10
    output = torch::nn::functional::softmax(output, -1); // softmax on cuda?
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "cnn forward (CUDA)");
}

/*  
    1, 28, 28
    32, 14, 14
    64, 7, 7
    128, 4, 4output = squeeze(output);
    256, 2, 2
    512, 1, 1
    512
    512 -> 50
    50 -> 10
    softmax(10)

    parameters
    Size of parameter : 32 //  1, 32 weight 32 3x3 size kernels
    Size of parameter : 32 //  1, 32 bias

    Size of parameter : 32 // 32, 32 weight
    Size of parameter : 32 // 32, 32 bias
    
    Size of parameter : 64 // 32, 64 weight
    Size of parameter : 64 // 32, 64 bias
    
    Size of parameter : 64 // 64, 64 weight
    Size of parameter : 64 // 64, 64 bias
    
    Size of parameter : 128// 64,128 weight
    Size of parameter : 128
    
    Size of parameter : 128
    Size of parameter : 128
    
    Size of parameter : 256
    Size of parameter : 256
    
    Size of parameter : 256
    Size of parameter : 256
    
    Size of parameter : 512
    Size of parameter : 512
    
    Size of parameter : 512
    Size of parameter : 512
    
    Size of parameter : 50 // 512, 50 linear weight
    Size of parameter : 50 // 512, 50 linear bias
    Size of parameter : 10 // 50, 10  linear weight
    Size of parameter : 10 // 50, 10  linear bias
*/