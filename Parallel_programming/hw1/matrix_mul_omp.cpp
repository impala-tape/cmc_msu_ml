
/*
  Matrix multiplication (naive) with optional OpenMP parallelization.
  Usage: ./matrix_mul_omp M N K P
    A is MxN, B is NxK, C is MxK, P is number of threads.
  Prints only the elapsed time T (seconds) excluding memory allocation,
  as required.
*/
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <stdint.h>

static inline void init_matrix(double* A, size_t rows, size_t cols, unsigned seed) {
    // Deterministic but quick initialization: A[i,j] = ((i*1315423911u + j*2654435761u + seed) % 100) / 10.0
    // Avoids heavy RNG cost, keeps values bounded.
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            uint32_t v = (uint32_t)(i * 1315423911u) ^ (uint32_t)(j * 2654435761u) ^ seed;
            v ^= v << 13; v ^= v >> 17; v ^= v << 5;
            A[i*cols + j] = (double)(v % 100) / 10.0;
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "Usage: %s M N K P\n", argv[0]);
        return 1;
    }
    const long M = std::strtol(argv[1], nullptr, 10);
    const long N = std::strtol(argv[2], nullptr, 10);
    const long K = std::strtol(argv[3], nullptr, 10);
    const long P = std::strtol(argv[4], nullptr, 10);
    if (M <= 0 || N <= 0 || K <= 0 || P <= 0) {
        std::fprintf(stderr, "All arguments must be positive integers.\n");
        return 1;
    }

    // Allocate (excluded from timing)
    double* A = (double*) std::malloc(sizeof(double) * (size_t)M * (size_t)N);
    double* B = (double*) std::malloc(sizeof(double) * (size_t)N * (size_t)K);
    double* C = (double*) std::malloc(sizeof(double) * (size_t)M * (size_t)K);
    if (!A || !B || !C) {
        std::fprintf(stderr, "Allocation failed. Consider smaller sizes.\n");
        std::free(A); std::free(B); std::free(C);
        return 1;
    }

    // Initialize inputs (excluded from timing)
    init_matrix(A, (size_t)M, (size_t)N, 123u);
    init_matrix(B, (size_t)N, (size_t)K, 321u);

    // Configure OpenMP
    omp_set_num_threads((int)P);

    // Time only the compute kernel
    const double t0 = omp_get_wtime();

    if (P == 1) {
        // Sequential version (no OpenMP parallel region)
        for (long i = 0; i < M; ++i) {
            for (long j = 0; j < K; ++j) {
                double sum = 0.0;
                for (long k = 0; k < N; ++k) {
                    sum += A[i*N + k] * B[k*K + j];
                }
                C[i*K + j] = sum;
            }
        }
    } else {
        // Parallel version: parallelize the outer loop over i
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < M; ++i) {
            for (long j = 0; j < K; ++j) {
                double sum = 0.0;
                for (long k = 0; k < N; ++k) {
                    sum += A[i*N + k] * B[k*K + j];
                }
                C[i*K + j] = sum;
            }
        }
    }

    const double t1 = omp_get_wtime();
    const double T = t1 - t0;

    // Output only time in seconds
    std::printf("%.6f\n", T);

    // Optional: lightweight checksum to discourage dead-code elimination (excluded from output)
    // (Not printed, just prevents the compiler from optimizing away C writes in edge builds.)
    volatile double sink = C[(size_t)(M>0?M-1:0)*(size_t)K + (size_t)(K>0?K-1:0)];
    (void)sink;

    std::free(A); std::free(B); std::free(C);
    return 0;
}
