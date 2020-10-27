#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void conv2d_ex(torch::Tensor *d_input, torch::Tensor *d_weight, torch::Tensor *d_bias){
    /*
    thread : output channel size
    block : input channel size
    */
    __shared__ float input[];
    __shared__ float output[];
    if(threadIdx.x == 0){
        // load 1 channel from input and save to Shared Memory
    }
    __syncthreads();
}

/*
    __syncthreads();
    for (int step = 1; step < 64; step*=2){
        if ((threadIdx.x % (2* step) == 0) && (threadIdx.x < 64)){
            conv_out[threadIdx.x] += thread_result[threadIdx.x + step];
        }
        __syncthreads();
    }
    if (threadsIdx.x == 0){

    }

*/

__global__ void convolutionGPU(
    float *d_Result,
    float *d_Data,
    int dataW,
    int dataH
    ){
    // Data cache: threadIdx.x , threadIdx.y
    __shared__ float data[TILE_W + KERNEL_RADIUS * 2][TILE_W + KERNEL_RADIUS * 2];
    
    // global mem address of this thread
    const int gLoc = threadIdx.x +
    IMUL(blockIdx.x, blockDim.x) +
    IMUL(threadIdx.y, dataW) +
    IMUL(blockIdx.y, blockDim.y) * dataW;
    
    // load cache (32x32 shared memory, 16x16 threads blocks)
    // each threads loads four values from global memory into shared mem
    // if in image area, get value in global mem, else 0
    int x, y; // image based coordinate
    
    // original image based coordinate
    const int x0 = threadIdx.x + IMUL(blockIdx.x, blockDim.x);
    const int y0 = threadIdx.y + IMUL(blockIdx.y, blockDim.y);
    
    // case1: upper left
    x = x0 - KERNEL_RADIUS;
    y = y0 - KERNEL_RADIUS;
    if ( x < 0 || y < 0 )
        data[threadIdx.x][threadIdx.y] = 0;
    else
        data[threadIdx.x][threadIdx.y] = d_Data[ gLoc - KERNEL_RADIUS - IMUL(dataW, KERNEL_RADIUS)];
    
    // case2: upper right
    x = x0 + KERNEL_RADIUS;
    y = y0 - KERNEL_RADIUS;
    if ( x > dataW-1 || y < 0 )
        data[threadIdx.x + blockDim.x][threadIdx.y] = 0;
    else
        data[threadIdx.x + blockDim.x][threadIdx.y] = d_Data[gLoc + KERNEL_RADIUS - IMUL(dataW, KERNEL_RADIUS)];
    
    // case3: lower left
    x = x0 - KERNEL_RADIUS;
    y = y0 + KERNEL_RADIUS;
    if (x < 0 || y > dataH-1)
        data[threadIdx.x][threadIdx.y + blockDim.y] = 0;
    else
        data[threadIdx.x][threadIdx.y + blockDim.y] = d_Data[gLoc - KERNEL_RADIUS + IMUL(dataW, KERNEL_RADIUS)];
    
    // case4: lower right
    x = x0 + KERNEL_RADIUS;
    y = y0 + KERNEL_RADIUS;
    if ( x > dataW-1 || y > dataH-1)
        data[threadIdx.x + blockDim.x][threadIdx.y + blockDim.y] = 0;
    else
        data[threadIdx.x + blockDim.x][threadIdx.y + blockDim.y] = d_Data[gLoc + KERNEL_RADIUS + IMUL(dataW, KERNEL_RADIUS)];
    
    __syncthreads();
    
    // convolution
    float sum = 0;
    x = KERNEL_RADIUS + threadIdx.x;
    y = KERNEL_RADIUS + threadIdx.y;
    for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++)
    for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
    sum += data[x + i][y + j] * d_Kernel[KERNEL_RADIUS + j] * d_Kernel[KERNEL_RADIUS + i];
    
    d_Result[gLoc] = sum;
    }
    
    

__global__ void conv2d_00(
    torch::Tensor input, 
    torch::Tensor output, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    //int stride, // implement stride on another kernel
    int padding,
    int width,
    int height){
    // kernel size = (3,3)
    // filter(weight, bias)
    __shared__ conv_out[];
    /*
    image
    input + padding

    */
}

__global__ void conv2d_01(){

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
    /*  Cpp version
    output = torch::nn::functional::detail::conv2d(input, parameter[0], parameter[1], 1, 1, 1, 1); // (1, 32, (3, 3), padding = 1)
    output = torch::nn::functional::relu(output);
    output = torch::nn::functional::detail::conv2d(output, parameter[2], parameter[3], 2, 1, 1, 1); // (32, 32, (3, 3), stride=2, padding=1)
    output = torch::nn::functional::relu(output);
    output = torch::nn::functional::detail::conv2d(output, parameter[4], parameter[5], 1, 1, 1, 1); // (32, 64, (3, 3), padding=1)
    output = torch::nn::functional::relu(output);
    output = torch::nn::functional::detail::conv2d(output, parameter[6], parameter[7], 2, 1, 1, 1); // (64, 64, (3, 3), stride=2, padding=1)
    output = torch::nn::functional::relu(output);
    output = torch::nn::functional::detail::conv2d(output, parameter[8], parameter[9], 1, 1, 1, 1); // (64, 128, (3, 3), padding=1)
    output = torch::nn::functional::relu(output);
    output = torch::nn::functional::detail::conv2d(output, parameter[10], parameter[11], 2, 1, 1, 1); // (128, 128, (3, 3), stride=2, padding=1)
    output = torch::nn::functional::relu(output);
    output = torch::nn::functional::detail::conv2d(output, parameter[12], parameter[13], 1, 1, 1, 1); // (128, 256, (3, 3), padding=1)
    output = torch::nn::functional::relu(output);
    output = torch::nn::functional::detail::conv2d(output, parameter[14], parameter[15], 2, 1, 1, 1); // (256, 256, (3, 3), stride=2, padding=1)
    output = torch::nn::functional::relu(output);
    output = torch::nn::functional::detail::conv2d(output, parameter[16], parameter[17], 1, 1, 1, 1); // (256, 512, (3, 3), padding=1)
    output = torch::nn::functional::relu(output);
    output = torch::nn::functional::detail::conv2d(output, parameter[18], parameter[19], 2, 1, 1, 1); // (512, 512, (3, 3), stride=2, padding=1)
    output = torch::nn::functional::relu(output);
    */
    output = squeeze(output);
    cudaMemcpy(,,,cudaMemcpyDeviceToHost); // return from CUDA output size: 10 * sizeof(torch::Tensor)
    /*  Cpp version
    output = torch::nn::functional::linear(output, parameter[20], parameter[21]); // 50, 50
    output = torch::nn::functional::relu(output);
    output = torch::nn::functional::linear(output, parameter[22], parameter[23]); // 10, 10
    output = torch::nn::functional::softmax(output, -1); // softmax on cuda?
    */
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "cnn forward (CUDA)");
}

/*  
    in sequential approach
for h in range(channel):
    for i in range(parameter):
        for j in range(width):
            for k in range(height):
                for l in range(kernel_size):
                    conv

    1, 28, 28
    32, 14, 14
    64, 7, 7
    128, 4, 4
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