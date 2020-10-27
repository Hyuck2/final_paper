#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void conv2d_ex(
    int *d_input, 
    int *d_output, 
    int *d_weight, 
    int *d_bias,
    int in_channel,
    int out_channel,
    int padding_size,
    int stride_size,
    int img_size){
    /*
    thread : output channel size
    block : input channel size
    */
    __shared__ int input[img_size**2];
    __shared__ int output[16];
    __shared__ int weight[16]; // real size = 9
    

    input[threadIdx.x] = d_input[threadIdx.x];
    if(threadIdx.x < 9){
        // load 1 channel from input and save to Shared Memory
        weight[threadIdx.x] =  d_weight[blockIdx.x*9 + threadIdx.x];
    }
    __syncthreads();


    for(int i=0; i<img_size + padding_size - 3 + 1; i++){
        for(int j=0; j<img_size + padding_size - 3 + 1; j++){
            
        }
    }
}

int main(void){
    int *d_input;
    int *d_output;
    int *d_weight;
    int *d_bias;

    cudaMalloc((void**)&d_input, 16 * sizeof(int));
    cudaMalloc((void**)&d_output, 16 * 3 * sizeof(int));
    cudaMalloc((void**)&d_weight, 27 * sizeof(int));
    cudaMalloc((void**)&d_bias, 27 * sizeof(int));

    int input[4][4]={
        {1,1,1,1},
        {1,1,1,1},
        {1,1,1,1},
        {1,1,1,1}
    };
    int output[3][4][4]={
        {
            {1,2,3,4},
            {5,6,7,8},
            {9,10,11,12},
            {13,14,15,16}
        },
        {
            {17,18,19,20},
            {21,22,23,24},
            {25,26,27,28},
            {29,30,31,32}
        },
        {
            {0,0,0,0},
            {0,0,0,0},
            {0,0,0,0},
            {0,0,0,0}
        }
    };
    int weight[3][3][3] = {
        {
            {1,1,1}, 
            {1,1,1}, 
            {1,1,1}
        },
        {
            {2,2,2},
            {2,2,2},
            {2,2,2}
        },
        {
            {3,3,3},
            {3,3,3},
            {3,3,3}
        }
    };
    int bias[3][3][3] = {
        {
            {1,1,1}, 
            {1,1,1}, 
            {1,1,1}
        },
        {
            {2,2,2},
            {2,2,2},
            {2,2,2}
        },
        {
            {3,3,3},
            {3,3,3},
            {3,3,3}
        }
    };

    cudaMemcpy(d_input,input, 16*sizeof(int) ,cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight,weight, 27*sizeof(int) ,cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias,bias, 27*sizeof(int) ,cudaMemcpyHostToDevice);

    //cudaMemcpy(output, d_output, 48*sizeof(int), cudaMemcpyDeviceToHost);
    for(int channel=0;channel<3;channel++){
        printf("Channel %d\n", channel);
        for(int i=0;i<4;i++){
            for(int j=0;j<4;j++){
                printf("%d ", output[channel][i][j]);
            }
            printf("\n");
        }
    }
    return 0;
}