#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <omp.h>

static inline void init_matrix(double* A, size_t rows, size_t cols, unsigned seed) {
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j) {
            uint32_t v = (uint32_t)(i * 1315423911u) ^ (uint32_t)(j * 2654435761u) ^ seed;
            v ^= v << 13; v ^= v >> 17; v ^= v << 5;
            A[i*cols + j] = (double)(v % 100) / 10.0;
        }
}

int main(int argc, char** argv) {
    if (argc != 5) { std::fprintf(stderr, "Usage: %s M N K P\n", argv[0]); return 1; }
    const long M_in = std::strtol(argv[1], nullptr, 10);
    const long N_in = std::strtol(argv[2], nullptr, 10);
    const long K_in = std::strtol(argv[3], nullptr, 10);
    const long P_in = std::strtol(argv[4], nullptr, 10);
    if (M_in <= 0 || N_in <= 0 || K_in <= 0 || P_in <= 0) return 1;

    const size_t M = (size_t)M_in, N = (size_t)N_in, K = (size_t)K_in;
    const int P = (int)P_in;

    const size_t bytesA = M * N * sizeof(double);
    const size_t bytesB = N * K * sizeof(double);
    const size_t bytesC = M * K * sizeof(double);

    double* A = (double*)std::malloc(bytesA);
    double* B = (double*)std::malloc(bytesB);
    double* C = (double*)std::malloc(bytesC);
    if (!A || !B || !C) { std::free(A); std::free(B); std::free(C); return 1; }

    init_matrix(A, M, N, 123u);
    init_matrix(B, N, K, 321u);

    omp_set_num_threads(P);
    const double t0 = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < (long)M; ++i) {
        for (size_t j = 0; j < K; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k)
                sum += A[(size_t)i*N + k] * B[k*K + j];
            C[(size_t)i*K + j] = sum;
        }
    }
    const double t1 = omp_get_wtime();
    std::printf("%.6f\n", t1 - t0);

    volatile double sink = C[(M?M-1:0)*K + (K?K-1:0)];
    (void)sink;
    std::free(A); std::free(B); std::free(C);
    return 0;
}
