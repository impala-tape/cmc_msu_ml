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

int main() {
    const int M_in = 10000;
    const int N_in = 10000;
    const int P    = 8;

    const size_t M = (size_t)M_in, N = (size_t)N_in;

    double* A = (double*)std::malloc(M * N * sizeof(double));
    if (!A) {
        std::free(A);
        printf("Memory allocation failed\n");
        return 1;
    }
    init_matrix(A, M, N, 123u);

    omp_set_num_threads(P);

    const double t0 = omp_get_wtime();

    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (long i = 0; i < (long)M; ++i) {
        double row_sum = 0.0;
        const size_t base = (size_t)i * N;
        for (size_t j = 0; j < N; ++j)
            row_sum += A[base + j];
        sum += row_sum;
    }

    const double t1 = omp_get_wtime();

    std::printf("%.6f\n", t1 - t0);
    std::printf("Matrix size %dx%d\n", M_in, N_in);
    std::printf("Sum %f\n", sum);
    std::printf("Threads %d\n", P);

    std::free(A);
    return 0;
}
