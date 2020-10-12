#include <torch/extension.h>
#include <vector>

torch::Tensor forward(){

    torch::Tensor output;
    return output;
}

torch::Tensor backward(){

    torch::Tensor output;
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "cnn forward (CPP)");
  m.def("backward", &backward, "cnn backward (CPP)");
}