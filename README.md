Methods contains only SVM for now

Data is the same for kernels

each kernel has different folder where the kernel matrix will be computed and we do cross validation etc.

## Build C++ libary

execute (in project root):
```
g++ -c -fPIC Code/c-functions/kernel-functions.cpp -Wextra -Wall -o Code/c-functions/fun.o && g++ -shared -Wl,-soname,Code/c-functions/kernel-functions.so -o Code/c-functions/kernel-functions.so Code/c-functions/fun.o
```