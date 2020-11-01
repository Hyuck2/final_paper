# Extension

parameters(weight, bias) --> already on device
how to access?
check example/train.py --> .to(device)


conv2d

input channel
- 1
- 32
- 64
- 128
- 256
- 512

input size
- 28 x 28
- 14 x 14
- 7 x 7
- 4 x 4
- 2 x 2
- 1 x 1

output channel

kernel size
- 3 x 3
kernel number
- 32
- 64
- 128
- 256
- 512

padding
- 1
stride
- 1 or 2

output channel == kernel number

for k_num in kernel_number:
    sum = 0
    for i_ch in input_channel:
        for i_width in input_width:
            for i_height in input_height:
                current_location = (i_ch, i_width, i_height)
                if current_location == top_left:
                    for k_x in range(0,1):
                        for k_y in range(0,1):
                            sum += input_img[current_location + k_y*input_width + k_x] * kernel_weight[4 + k_y*3+k_x] + kernel_bias[4 + k_y*3+k_x]
                elif current_location == top_right:
                    for k_x in range(-1,0):
                        for k_y in range(0,1):
                            sum += input_img[current_location + k_y*input_width + k_x] * kernel_weight[4 + k_y*3+k_x] + kernel_bias[4 + k_y*3+k_x]
                elif current_location == top_middle:
                    for k_x in range(-1,1):
                        for k_y in range(0,1):
                            sum += input_img[current_location + k_y*input_width + k_x] * kernel_weight[4 + k_y*3+k_x] + kernel_bias[4 + k_y*3+k_x]
                elif current_location == bottom_left:
                    for k_x in range(-1,0):
                        for k_y in range(0,1):
                            sum += input_img[current_location + k_y*input_width + k_x] * kernel_weight[4 + k_y*3+k_x] + kernel_bias[4 + k_y*3+k_x]
                elif current_location == bottom_right:
                    for k_x in range(-1,0):
                        for k_y in range(-1,0):
                            sum += input_img[current_location + k_y*input_width + k_x] * kernel_weight[4 + k_y*3+k_x] + kernel_bias[4 + k_y*3+k_x]
                elif current_location == bottom_middle:
                    for k_x in range(-1,1):
                        for k_y in range(-1,0):
                            sum += input_img[current_location + k_y*input_width + k_x] * kernel_weight[4 + k_y*3+k_x] + kernel_bias[4 + k_y*3+k_x]
                elif current_location == left_middle:
                    for k_x in range(0,1):
                        for k_y in range(-1,1):
                            sum += input_img[current_location + k_y*input_width + k_x] * kernel_weight[4 + k_y*3+k_x] + kernel_bias[4 + k_y*3+k_x]
                elif current_location == left_right:
                    for k_x in range(-1,0):
                        for k_y in range(-1,1):
                            sum += input_img[current_location + k_y*input_width + k_x] * kernel_weight[4 + k_y*3+k_x] + kernel_bias[4 + k_y*3+k_x]
                else:
                    for k_x in range(-1,1):
                        for k_y in range(-1,1):
                            sum += input_img[current_location + k_y*input_width + k_x] * kernel_weight[4 + k_y*3+k_x] + kernel_bias[4 + k_y*3+k_x]
    output[k_num] = sum