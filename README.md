# Improving Pytorch Learning Speed by Using Custom CUDA Kernel

1. Motivation
   
    Replace pytorch's standard modules with Custom CUDA Kernel
    - Advantages
      1. Speed
      2. Customization
         1. Make models not supported by Pytorch
         2. Variable Kernel size(Not 1 by 1 method)
    - Disadvantages
      1. Need CUDA Language skill

2. Enviroment
   - HardWare
     - intel i7-9700k
     - Nvidia RTX2070
   - Software
     - Ubuntu 20.04
     - Pytorch 1.6.0
     - GCC 9.3.0
     - Python 3.8.2
     - CUDA
       - Toolkit 10.2
       - Driver 440.100
     
3. Implementation
   - MNIST
   - Image Classification?
   - NLP?

4. Experiment
   - Only Using Python & Pytorch
   - CPU vs GPU
   - Implement C++
   - Implement CUDA

5. Discussion
   - Use pyprof or nvprof to check How much kernel created
   - Profile number of kernel, memory usage, execution time, Occupancy
   - Good for performance
   - Good for Cusomization