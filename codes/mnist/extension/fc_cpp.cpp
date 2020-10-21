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