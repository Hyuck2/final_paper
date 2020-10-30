#include <torch/extension.h>
#include <vector>

torch::Tensor forward(torch::Tensor input, std::vector<torch::Tensor> parameter){
    torch::Tensor output;
    
    output = torch::nn::functional::detail::conv2d(input, parameter[0] /*weight*/, parameter[1] /*bias*/, 1, 1, 1, 1); // (1, 32, (3, 3), padding = 1)
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
    output = squeeze(output);
    output = torch::nn::functional::linear(output, parameter[20], parameter[21]); // 50, 50
    output = torch::nn::functional::relu(output);
    output = torch::nn::functional::linear(output, parameter[22], parameter[23]); // 10, 10
    output = torch::nn::functional::softmax(output, -1);
    return output;
}
/*
check input tensor shape..


*/

torch::Tensor conv2d_relu(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int input_channel_size, int input_width, int input_height, int kernel_channel){
  int current_location;
  torch::Tensor sum;
  torch::Tensor output; // with padding==1, size = input_channel * input_hight * input_width * kernel_channel
  // input channel, width, height
  // kernel channel, kernel width, kernel height
  for (int ch=0; ch<input_channel_size; ch++){
    for (int y=0; y<input_height; y++){
      for(int x=0;x<input_width;x++){
        current_location = ch*input_height*input_width + y*input_width + x;
        // top left
        if (x==0 && y==0){
          for (int k_ch=0; k_ch<kernel_channel; k_ch++){ // kernel channel
            sum = 0;
            for (int k_y=0; k_y<2;k_y++){ // kernel height
              for(int k_x=0; k_x<2;k_x++){ // kernel width
                sum += input[current_location + k_y*input_height + k_x] * weight[k_y*3 + k_x] + bias[k_y*3 + k_x];
              }
            }
            if (sum<0){
              sum = 0; // relu
            }
        }
        // top right
        else if (x==input_width-1 && y==0){
          for (int k_ch=0; k_ch<kernel_channel; k_ch++){ // kernel channel
            sum = 0;
            for (int k_y=0; k_y<2;k_y++){ // kernel height
              for(int k_x=-1; k_x<1;k_x++){ // kernel width
                sum += input[current_location + k_y*input_height + k_x] * weight[k_y*3 + k_x] + bias[k_y*3 + k_x];
              }
            }
            if (sum<0){
              sum = 0; // relu
            }
        }
        // top middle
        else if (y==0){
          for (int k_ch=0; k_ch<kernel_channel; k_ch++){ // kernel channel
            sum = 0;
            for (int k_y=0; k_y<2;k_y++){ // kernel height
              for(int k_x=-1; k_x<2;k_x++){ // kernel width
                sum += input[current_location + k_y*input_height + k_x] * weight[k_y*3 + k_x] + bias[k_y*3 + k_x];
              }
            }
            if (sum<0){
              sum = 0; // relu
            }
        }
        // bottom left
        else if (x==0 && y==input_height-1){
          for (int k_ch=0; k_ch<kernel_channel; k_ch++){ // kernel channel
            sum = 0;
            for (int k_y=-1; k_y<1;k_y++){ // kernel height
              for(int k_x=0; k_x<2;k_x++){ // kernel width
                sum += input[current_location + k_y*input_height + k_x] * weight[k_y*3 + k_x] + bias[k_y*3 + k_x];
              }
            }
            if (sum<0){
              sum = 0; // relu
            }
        }
        // bottom right
        else if (x==input_width-1 && y==input_height-1){
          for (int k_ch=0; k_ch<kernel_channel; k_ch++){ // kernel channel
            sum = 0;
            for (int k_y=-1; k_y<1;k_y++){ // kernel height
              for(int k_x=-1; k_x<1;k_x++){ // kernel width
                sum += input[current_location + k_y*input_height + k_x] * weight[k_y*3 + k_x] + bias[k_y*3 + k_x];
              }
            }
            if (sum<0){
              sum = 0; // relu
            }
        }
        // bottom middle
        else if (y==input_height-1){
          for (int k_ch=0; k_ch<kernel_channel; k_ch++){ // kernel channel
            sum = 0;
            for (int k_y=-1; k_y<1;k_y++){ // kernel height
              for(int k_x=-1; k_x<2;k_x++){ // kernel width
                sum += input[current_location + k_y*input_height + k_x] * weight[k_y*3 + k_x] + bias[k_y*3 + k_x];
              }
            }
            if (sum<0){
              sum = 0; // relu
            }
        }
        // left middle
        else if (x==0){
          for (int k_ch=0; k_ch<kernel_channel; k_ch++){ // kernel channel
            sum = 0;
            for (int k_y=-1; k_y<2;k_y++){ // kernel height
              for(int k_x=0; k_x<2;k_x++){ // kernel width
                sum += input[current_location + k_y*input_height + k_x] * weight[k_y*3 + k_x] + bias[k_y*3 + k_x];
              }
            }
            if (sum<0){
              sum = 0; // relu
            }
        }
        // right middle
        else if (x==input_width-1){
          for (int k_ch=0; k_ch<kernel_channel; k_ch++){ // kernel channel
            sum = 0;
            for (int k_y=-1; k_y<2;k_y++){ // kernel height
              for(int k_x=-1; k_x<1;k_x++){ // kernel width
                sum += input[current_location + k_y*input_height + k_x] * weight[k_y*3 + k_x] + bias[k_y*3 + k_x];
              }
            }
            if (sum<0){
              sum = 0; // relu
            }
        }
        // inside
        else{
          for (int k_ch=0; k_ch<kernel_channel; k_ch++){ // kernel channel
            sum = 0;
            for (int k_y=-1; k_y<2;k_y++){ // kernel height
              for(int k_x=-1; k_x<2;k_x++){ // kernel width
                sum += input[current_location + k_y*input_height + k_x] * weight[k_y*3 + k_x] + bias[k_y*3 + k_x];
              }
            }
            if (sum<0){
              sum = 0; // relu
            }

            // kernel channel result
          }
        }
      }
    }
    // save to channel
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "cnn forward (CPP)");
}