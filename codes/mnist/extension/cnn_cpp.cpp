#include <torch/extension.h>
#include <vector>
#include <iostream>

torch::Tensor conv2d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int kernel_num, int input_channel, int input_width, int input_height){
  torch::Tensor sum = torch::zeros({});
  torch::Tensor output = torch::zeros({64,32,28,28});
  /*
  std::cout << "\nweight\n" << std::endl;
  std::cout << weight << std::endl;
  std::cout << "\nbias\n" << std::endl;
  std::cout << bias << std::endl;
  std::cout << "\nsum\n" << std::endl;
  std::cout << "\ninput\n" << std::endl;
  std::cout << input << std::endl;
  */
  
  for (int batch=0;batch<64;batch++){
    for (int k_num=0; k_num<kernel_num; k_num++){
      for (int i_y=0; i_y<input_width; i_y++){
        for(int i_x=0; i_x<input_height;i_x++){
          for (int i_ch=0; i_ch<input_channel; i_ch++){                
            sum = torch::zeros({});
            // top left
            if (i_y == 0 && i_x==0){
              for (int k_y=0; k_y<2; k_y++){
                for (int k_x=0; k_x<2; k_x++){
                  sum += input[batch][i_ch][i_x + k_x][i_y + k_y] * weight[k_num][0][1+k_x][1+k_y];
                }
              }
            }
            // top right
            else if (i_y==0 && i_x==input_width-1){
              for (int k_y=0; k_y<2; k_y++){
                for (int k_x=-1; k_x<1; k_x++){
                  sum += input[batch][i_ch][i_x + k_x][i_y + k_y] * weight[k_num][0][1+k_x][1+k_y];
                }
              }
            }
            // bottom left
            else if (i_y==input_height-1 && i_x==0){
              for (int k_y=-1; k_y<1; k_y++){
              for (int k_x=0; k_x<2; k_x++){
                sum += input[batch][i_ch][i_x + k_x][i_y + k_y] * weight[k_num][0][1+k_x][1+k_y];
              }
            }
          }
          // bottom right
          else if (i_y==input_height-1 && i_x==input_width-1){
            for (int k_y=-1; k_y<1; k_y++){
              for (int k_x=-1; k_x<1; k_x++){
                sum += input[batch][i_ch][i_x + k_x][i_y + k_y] * weight[k_num][0][1+k_x][1+k_y];
              }
            }
          }
          // top middle
          else if (i_y==0){
            for (int k_y=0; k_y<2; k_y++){
              for (int k_x=-1; k_x<2; k_x++){
                sum += input[batch][i_ch][i_x + k_x][i_y + k_y] * weight[k_num][0][1+k_x][1+k_y];
              }
            }
          }
          // bottom middle
          else if (i_y==input_height-1){
            for (int k_y=-1; k_y<1; k_y++){
              for (int k_x=-1; k_x<2; k_x++){
                sum += input[batch][i_ch][i_x + k_x][i_y + k_y] * weight[k_num][0][1+k_x][1+k_y];
              }
            }
          }
          // left middle
          else if (i_x==0){
            for (int k_y=-1; k_y<2; k_y++){
              for (int k_x=0; k_x<2; k_x++){
                sum += input[batch][i_ch][i_x + k_x][i_y + k_y] * weight[k_num][0][1+k_x][1+k_y];
              }
            }
          }
          // right middle
          else if (i_x==input_width-1){
            for (int k_y=-1; k_y<2; k_y++){
              for (int k_x=-1; k_x<1; k_x++){
                sum += input[batch][i_ch][i_x + k_x][i_y + k_y] * weight[k_num][0][1+k_x][1+k_y];
              }
            }
          }
          // else
          else{
            for (int k_y=-1; k_y<2; k_y++){
              for (int k_x=-1; k_x<2; k_x++){
                sum += input[batch][i_ch][i_x + k_x][i_y + k_y] * weight[k_num][0][1+k_x][1+k_y];
              }
            }
          }
          output[batch][k_num][i_x][i_y] = sum + bias[k_num];
        }
      }
    }
    // relu
  }
  }
  return output;
}

torch::Tensor forward(torch::Tensor input, std::vector<torch::Tensor> parameter){
    torch::Tensor output;    
    output = torch::nn::functional::detail::conv2d(input, parameter[0] /*weight*/, parameter[1] /*bias*/, 1, 1, 1, 1); // (1, 32, (3, 3), padding = 1)
    //output = conv2d_forward(input, parameter[0], parameter[1], 32, 1, 28, 28);
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "cnn forward (CPP)");
}