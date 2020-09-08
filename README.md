# Improving Pytorch Learning Speed by Using Custom CUDA Kernel

1. Motivation
   
    Replace pytorch's standard modules with Custom CUDA Kernel
    - Advantages
      1. Speed
      2. Customization
    - Disadvantages
      1. Need CUDA Language skill
      2. Reusability low

2. Enviroment
   - HardWare
     - intel i7-9700k
     - Nvidia RTX2070
   - Software
     - Ubuntu 20.04
     - Pytorch 1.6.0
     - CUDA
       - Toolkit 10.2
       - Driver 440.100

3. Implementation
4. Experiment
   - Only Using Python & Pytorch
   - CPU vs GPU
   - Implement C++
   - Implement CUDA
5. Discussion
   - Use pyprof or nvprof to check How much kernel created
   - Profile number of kernel, memory usage, execution time, Occupancy