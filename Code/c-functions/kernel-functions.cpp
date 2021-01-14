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

    double wd_shifted(int* x1, int* x2, int k, int L, int S) {
        double sum = 0.0;
        int mask = (1 << (2 * k)) - 1;
        int u1 = 0, u2 = 0, u1_shifted = 0, u2_shifted = 0;
        for (int s = 0; s < S; ++s) {
            double delta_s = 1.0 / (2 * (s + 1));
            for (int i = 0; i < L - s; ++i) {
                u1 = ((u1 << 2) & mask) | x1[i];
                u2 = ((u2 << 2) & mask) | x2[i];
                u1_shifted = ((u1_shifted << 2) & mask) | x1[i + s];
                u2_shifted = ((u2_shifted << 2) & mask) | x1[i + s];
                if (i + 1 >= k) { //u1 and u2 already have length k?
                    sum += delta_s * ((u1 == u2_shifted) + (u2 == u1_shifted));
                }
            }
        }
        return sum;
    }
}
