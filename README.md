
# Data Challenge

To run our experiments, execute

```
python3 experiments.py
```

in this folder. Note, that only a representative sample expirment is submitted. However, any expirement we did throughout the challenge can be expressed in a succint manner using the versatile `util.cross_validation` function. It takes as input three arrays representing differnet models, kernels and training data respectively. `util.cross_valdiation` outputs a scores object that can be plotted using the `util.plot_cross_val` function. To classify test data use `util.save_predictions`.

For a usage sample see `experiments.py`. The repository contains cached kernel matrices for the mismtach kernel with k=10 and k=12.


### Build C++ libary

For the weighted degree kernel we implemented the computation of kernel function in C++. To compile the c++ code, such that they can be called form our pyhton code execute (in project root):
```
g++ -c -fPIC Code/c-functions/kernel-functions.cpp -Wextra -Wall -o Code/c-functions/fun.o && g++ -shared -Wl,-soname,Code/c-functions/kernel-functions.so -o Code/c-functions/kernel-functions.so Code/c-functions/fun.o
```
