#include <bits/stdc++.h>
using namespace std;

extern "C" {
    double wd_kernel(int* x1, int* x2, int k, int L) {
        double sum = 0.0;
        int mask = (1 << (2 * k)) - 1;
        int u1 = 0, u2 = 0;
        for (int i = 0; i < L; ++i) {
            u1 = ((u1 << 2) & mask) | x1[i];
            u2 = ((u2 << 2) & mask) | x2[i];
            if (i + 1 >= k) { //u1 and u2 already have length k?
                sum += (u1 == u2);
            }
        }
        return sum;
    }

}
