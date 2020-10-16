#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void conv2d_relu_kernel(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding){
    // conv2d and relu on CUDA
    
}

__global__ void linear_relu_kernel(torch::Tensor input, torch::Tensor weight, torch::Tensor bias){
    // squeeze needed before linear layer
    // linear
    // relu
    // linear
    
}

torch::Tensor forward(torch::Tensor input, std::vector<torch::Tensor> parameter){
    torch::Tensor output;
    torch::Tensor* d_input;
    torch::Tensor* d_output;
    cudaMalloc((void**)&d_input, *sizeof(torch::Tensor));
    cudaMalloc((void**)&d_output, *sizeof(torch::Tensor));
    cudaMemcpy(,,,cudaMemcpyHostToDevice); // copy input to CUDA
    conv2d_relu_kernel<<<>>>( /*input*/, d_output /*output*/, parameter[0]/*weight*/, parameter[1]/*bias*/, 1/*stride*/, 1/*padding*/);
    conv2d_relu_kernel<<<>>>( /*input*/, d_output /*output*/, parameter[2]/*weight*/, parameter[3]/*bias*/, 2/*stride*/, 1/*padding*/);
    conv2d_relu_kernel<<<>>>( /*input*/, d_output /*output*/, parameter[4]/*weight*/, parameter[5]/*bias*/, 1/*stride*/, 1/*padding*/);
    conv2d_relu_kernel<<<>>>( /*input*/, d_output /*output*/, parameter[6]/*weight*/, parameter[7]/*bias*/, 2/*stride*/, 1/*padding*/);
    conv2d_relu_kernel<<<>>>( /*input*/, d_output /*output*/, parameter[8]/*weight*/, parameter[9]/*bias*/, 1/*stride*/, 1/*padding*/);
    conv2d_relu_kernel<<<>>>( /*input*/, d_output /*output*/, parameter[10]/*weight*/, parameter[11]/*bias*/, 2/*stride*/, 1/*padding*/);
    conv2d_relu_kernel<<<>>>( /*input*/, d_output /*output*/, parameter[12]/*weight*/, parameter[13]/*bias*/, 1/*stride*/, 1/*padding*/);
    conv2d_relu_kernel<<<>>>( /*input*/, d_output /*output*/, parameter[14]/*weight*/, parameter[15]/*bias*/, 2/*stride*/, 1/*padding*/);
    conv2d_relu_kernel<<<>>>( /*input*/, d_output /*output*/, parameter[16]/*weight*/, parameter[17]/*bias*/, 1/*stride*/, 1/*padding*/);
    conv2d_relu_kernel<<<>>>( /*input*/, d_output /*output*/, parameter[18]/*weight*/, parameter[19]/*bias*/, 2/*stride*/, 1/*padding*/);
    // squeeze needed
    linear_relu_kernel<<<>>>( /*input*/, d_output /*output*/, parameter[20]/*weight 1*/, parameter[21]/*bias 1*/, parameter[22]/*weight 2*/, parameter[23]/*bias 2*/);
    cudaMemcpy(,,,cudaMemcpyDeviceToHost); // return from CUDA output size: 10 * sizeof(torch::Tensor)
    output = torch::nn::functional::softmax(output, -1); // softmax on cuda?
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "cnn forward (CUDA)");
}