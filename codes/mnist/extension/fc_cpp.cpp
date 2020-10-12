#include <torch/extension.h>
#include <vector>

torch::Tensor forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias){
    torch::Tensor output;
    output = torch::nn::functional::linear(input, weight, bias);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Fully connected forward (CPP)");
  //m.def("backward", &backward, "Fully connected backward (CPP)");
}