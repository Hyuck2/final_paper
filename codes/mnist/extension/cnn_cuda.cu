#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


__global__ void cuda_forward_kernel(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int kernel_num, int input_channel, int input_width, int input_height){
    torch::Tensor sum;
    torch::Tensor output;

    int k_num = blockIdx.x;
    int i_ch = threadIdx.x;
    __shared__ torch::Tensor thread_sum[kernel_num];
    // thread result --> input channel
    // parallel reduction -->output channel

    for (int i_y=0; i_y<input_width; i_y++){
        for(int i_x=0; i_x<input_height;i_x++){
          if (i_y == 0 && i_x==0){ // top left
            for (int k_y=0; k_y<2; k_y++){
              for (int k_x=0; k_x<2; k_x++){
                torch::add(sum,torch::add(torch::matmul(input[i_ch][i_x + k_x][i_y + k_y], weight[k_num][1+k_x][1+k_y]),bias[k_num][1+k_x][1+k_y]));
              }
            }
          }
          else if (i_y==input_width-1 && i_x==0){ // top right
            for (int k_y=0; k_y<2; k_y++){
              for (int k_x=-1; k_x<1; k_x++){
                sum += torch::matmul(input[i_ch][i_x + k_x][i_y + k_y], weight[k_num][1+k_x][1+k_y]) + bias[k_num][1+k_x][1+k_y];
              }
            }
          }
          else if (i_y==0 && i_x==input_height-1){ // bottom left
            for (int k_y=-1; k_y<1; k_y++){
              for (int k_x=0; k_x<2; k_x++){
                sum += torch::matmul(input[i_ch][i_x + k_x][i_y + k_y], weight[k_num][1+k_x][1+k_y]) + bias[k_num][1+k_x][1+k_y];
              }
            }
          }
          else if (i_y==input_width-1 && i_x==input_height-1){ // bottom right
            for (int k_y=-1; k_y<1; k_y++){
              for (int k_x=-1; k_x<1; k_x++){
                sum += torch::matmul(input[i_ch][i_x + k_x][i_y + k_y], weight[k_num][1+k_x][1+k_y]) + bias[k_num][1+k_x][1+k_y];
              }
            }
          }
          else if (i_x==0){ //top middle
            for (int k_y=0; k_y<2; k_y++){
              for (int k_x=-1; k_x<2; k_x++){
                sum += torch::matmul(input[i_ch][i_x + k_x][i_y + k_y], weight[k_num][1+k_x][1+k_y]) + bias[k_num][1+k_x][1+k_y];
              }
            }
          }
          else if (i_x==input_height-1){ // bottom middle
            for (int k_y=-1; k_y<1; k_y++){
              for (int k_x=-1; k_x<2; k_x++){
                sum += torch::matmul(input[i_ch][i_x + k_x][i_y + k_y], weight[k_num][1+k_x][1+k_y]) + bias[k_num][1+k_x][1+k_y];
              }
            }
          }
          else if (i_y==0){ // left middle
            for (int k_y=-1; k_y<2; k_y++){
              for (int k_x=0; k_x<2; k_x++){
                sum += torch::matmul(input[i_ch][i_x + k_x][i_y + k_y], weight[k_num][1+k_x][1+k_y]) + bias[k_num][1+k_x][1+k_y];
              }
            }
          }
          else if (i_y==input_width-1){ // right middle
            for (int k_y=-1; k_y<2; k_y++){
              for (int k_x=-1; k_x<1; k_x++){
                sum += torch::matmul(input[i_ch][i_x + k_x][i_y + k_y], weight[k_num][1+k_x][1+k_y]) + bias[k_num][1+k_x][1+k_y];
              }
            }
          }
          else{ // else
            for (int k_y=-1; k_y<2; k_y++){
              for (int k_x=-1; k_x<2; k_x++){
                sum += torch::matmul(input[i_ch][i_x + k_x][i_y + k_y], weight[k_num][1+k_x][1+k_y]) + bias[k_num][1+k_x][1+k_y];
              }
            }
          }
        }
    }
    output[blockIdx.x] = sum;
}

torch::Tensor forward(torch::Tensor input, std::vector<torch::Tensor> parameter){
    torch::Tensor output;

    AT_DISPATCH_FLOATING_TYPES(,"forward",([&] {cuda_forward_kernel<scalar_t><<<>>>();}));
    

    output = squeeze(output);
    
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "cnn forward (CUDA)");
}