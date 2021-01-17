## Things to Code

- (slow) implement mismatch kernel (hope for performance)
- (maybe) change float64 to float16 to reduce storage


## Things to Analyse

- sort dataset by labels and compare (kernel matrix should be sorted, too)
- do PPCA with different kernels and find main components
- check the alphas from our SVM implementation


## Things to Plot

- performance of different kernels on same SVM with given dataset (cross val in C)


### Build C++ libary

execute (in project root):
```
g++ -c -fPIC Code/c-functions/kernel-functions.cpp -Wextra -Wall -o Code/c-functions/fun.o && g++ -shared -Wl,-soname,Code/c-functions/kernel-functions.so -o Code/c-functions/kernel-functions.so Code/c-functions/fun.o
```
