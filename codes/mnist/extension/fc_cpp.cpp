#include <torch/extension.h>
#include <vector>

torch::Tensor forward(torch::Tensor input, std::vector<torch::Tensor> parameter){
    torch::Tensor output;
    output = torch::nn::functional::linear(input, parameter[0], parameter[1]);
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
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Fully connected forward (CPP)");
}