# Cpp & CUDA extensions

## Files
1. mnist_cnn.cu
2. setup.py

## How to use

## Function references
aten/src/ATen/native/Convolution.cpp
at::Tensor _convolution(
    const Tensor& input_r, const Tensor& weight_r, const Tensor& bias_r,
    IntArrayRef stride_, IntArrayRef padding_, IntArrayRef dilation_,
    bool transposed_, IntArrayRef output_padding_, int64_t groups_,
    bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32)
    

at::Tensor convolution(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
    bool transposed, IntArrayRef output_padding, int64_t groups)

at::Tensor conv2d(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, 
    int64_t groups)