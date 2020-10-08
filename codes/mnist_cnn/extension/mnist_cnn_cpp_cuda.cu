#include <iostream>
#include <vector>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

torch::Tensor forward(torch::Tensor input);
torch::Tensor backward(torch::Tensor loss);
torch::Tensor cu_forward(torch::Tensor input);
torch::Tensor cu_backward(torch::Tensor loss);

__global__ void forward_kernel(){

}

__global__ void backward_kernel(){
  
}

torch::Tensor forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias){
    torch::Tensor output;
    output = torch::conv2d(input, weight, bias, 0/*stride*/, 1 /*padding*/, /*dilation*/, /*groups*/); // stride , padding, dilation, groups
    output = torch::relu(output, 0) // if inplace == True: return relu_, else: return relu
    output = torch::batch_norm(ouput, weight, bias)

    output = torch::conv2d(input, weight, bias, 2 /*stride*/, 1 /*padding*/, /*dilation*/, /*groups*/); // stride , padding, dilation, groups
    output = torch::relu(output, 0) // if inplace == True: return relu_, else: return relu
    output = torch::batch_norm(ouput, weight, bias)

    output = torch::conv2d(input, weight, bias, /*stride*/, 1 /*padding*/, /*dilation*/, /*groups*/); // stride , padding, dilation, groups
    output = torch::relu(output, 0) // if inplace == True: return relu_, else: return relu
    output = torch::batch_norm(ouput, weight, bias)

    output = torch::conv2d(input, weight, bias, 2 /*stride*/, 1 /*padding*/, /*dilation*/, /*groups*/); // stride , padding, dilation, groups
    output = torch::relu(output, 0) // if inplace == True: return relu_, else: return relu
    output = torch::batch_norm(ouput, weight, bias)

    output = torch::conv2d(input, weight, bias, /*stride*/, 1 /*padding*/, /*dilation*/, /*groups*/); // stride , padding, dilation, groups
    output = torch::relu(output, 0) // if inplace == True: return relu_, else: return relu
    output = torch::batch_norm(ouput, weight, bias)

    output = torch::conv2d(input, weight, bias, 2 /*stride*/, 1 /*padding*/, /*dilation*/, /*groups*/); // stride , padding, dilation, groups
    output = torch::relu(output, 0) // if inplace == True: return relu_, else: return relu
    output = torch::batch_norm(ouput, weight, bias)

    output = torch::conv2d(input, weight, bias, /*stride*/, 1 /*padding*/, /*dilation*/, /*groups*/); // stride , padding, dilation, groups
    output = torch::relu(output, 0) // if inplace == True: return relu_, else: return relu
    output = torch::batch_norm(ouput, weight, bias)

    output = torch::conv2d(input, weight, bias, 2 /*stride*/, 1 /*padding*/, /*dilation*/, /*groups*/); // stride , padding, dilation, groups
    output = torch::relu(output, 0) // if inplace == True: return relu_, else: return relu
    output = torch::batch_norm(ouput, weight, bias)

    output = torch::conv2d(input, weight, bias, /*stride*/, 1 /*padding*/, /*dilation*/, /*groups*/); // stride , padding, dilation, groups
    output = torch::relu(output, 0) // if inplace == True: return relu_, else: return relu
    output = torch::batch_norm(ouput, weight, bias)

    output = torch::conv2d(input, weight, bias, 2 /*stride*/, 1 /*padding*/, /*dilation*/, /*groups*/); // stride , padding, dilation, groups
    output = torch::relu(output, 0) // if inplace == True: return relu_, else: return relu
    output = torch::batch_norm(ouput, weight, bias)

    // squeeze
    output = torch::linear()
    output = torch::relu(output, 0) // if inplace == True: return relu_, else: return relu
    output = torch::batch_norm(ouput, weight, bias)
    output = torch::linear()
    // softmax

    return output
}

torch::Tensor backward(torch::Tensor loss){

}

torch::Tensor cu_forward(torch::Tensor input){
/*
    cudamemcpy h2d input
    forward kernel
    cudamemcpy d2h output
*/
}

torch::Tensor cu_backward(torch::Tensor loss){
/*


*/
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "CNN forward Cpp");
    m.def("backward", &backward, "CNN backward Cpp");
    m.def("cu_forward", &cu_forward, "CNN forward CUDA");
    m.def("cu_backward", &cu_backward, "CNN backward CUDA");
}

/*
self.layer1 = torch.nn.Sequential(
    torch.nn.Conv2d(1, 32, (3, 3), padding=1),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(32),
    torch.nn.Conv2d(32, 32, (3, 3), stride=2, padding=1),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(32),
    torch.nn.Conv2d(32, 64, (3, 3), padding=1),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(64),
    torch.nn.Conv2d(64, 64, (3, 3), stride=2, padding=1),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(64),
    torch.nn.Conv2d(64, 128, (3, 3), padding=1),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(128),
    torch.nn.Conv2d(128, 128, (3, 3), stride=2, padding=1),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(128),
    torch.nn.Conv2d(128, 256, (3, 3), padding=1),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(256),
    torch.nn.Conv2d(256, 256, (3, 3), stride=2, padding=1),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(256),
    torch.nn.Conv2d(256, 512, (3, 3), padding=1),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(512),
    torch.nn.Conv2d(512, 512, (3, 3), stride=2, padding=1),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(512),)
self.layer2 = torch.nn.Sequential(
    torch.nn.Linear(512, 50),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(50),
    torch.nn.Linear(50, output_size), # MNIST--> outputsize = 10
    torch.nn.Softmax(dim=-1),)
*/