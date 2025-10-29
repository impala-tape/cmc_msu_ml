#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 100

// Возврат текущего времени в миллисекундах (монотонные часы)
static inline double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

// Чтение матрицы A из CSV-файла (N строк по N чисел, разделитель — запятая)
void read_matrix_csv(const char* filename, double A[N][N]) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        perror("Не удалось открыть CSV-файл");
        exit(1);
    }
    char line[1 << 16];
    for (int i = 0; i < N; ++i) {
        if (!fgets(line, sizeof(line), f)) {
            fprintf(stderr, "Недостаточно строк в CSV (ожидалось %d)\n", N);
            exit(1);
        }
        char* p = line;
        for (int j = 0; j < N; ++j) {
            // пропуск разделителей
            while (*p == ' ' || *p == '\t' || *p == ',') ++p;
            char* endp;
            double v = strtod(p, &endp);
            if (p == endp) {
                fprintf(stderr, "Ошибка парсинга CSV: строка %d, столбец %d\n", i + 1, j + 1);
                exit(1);
            }
            A[i][j] = v;
            p = endp;
            // пропуск завершающих разделителей
            while (*p == ' ' || *p == '\t' || *p == ',') ++p;
        }
    }
    fclose(f);
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "SLAU_var_5.csv";

    double A[N][N], A_orig[N][N];
    double x_true[N], x_calc[N];
    double F[N], F_orig[N];

    // 1) Чтение A из CSV
    read_matrix_csv(path, A);

    // 2) Генерация случайного x_true в [-1, 1]
    srand(42);
    for (int i = 0; i < N; ++i) {
        x_true[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    // 3) Вычисление F = A * x_true
    for (int i = 0; i < N; ++i) {
        double s = 0.0;
        for (int j = 0; j < N; ++j) s += A[i][j] * x_true[j];
        F[i] = s;
    }

    // Копии для проверки
    for (int i = 0; i < N; ++i) {
        F_orig[i] = F[i];
        for (int j = 0; j < N; ++j) A_orig[i][j] = A[i][j];
    }

    // 4) Метод Гивенса (последовательно)
    double t0 = now_ms();
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            if (A[j][i] == 0.0) continue;
            double a = A[i][i];
            double b = A[j][i];
            double r = hypot(a, b);
            double c = a / r;
            double s = b / r;
            // Поворот строк i и j для столбцов k >= i
            for (int k = i; k < N; ++k) {
                double aik = A[i][k];
                double ajk = A[j][k];
                A[i][k] = c * aik + s * ajk;
                A[j][k] = -s * aik + c * ajk;
            }
            // Поворот правой части
            double Fi = F[i], Fj = F[j];
            F[i] = c * Fi + s * Fj;
            F[j] = -s * Fi + c * Fj;
        }
    }

    // 5) Обратный ход
    for (int i = N - 1; i >= 0; --i) {
        double s = 0.0;
        for (int j = i + 1; j < N; ++j) s += A[i][j] * x_calc[j];
        double diag = A[i][i];
        if (fabs(diag) < 1e-15) {
            fprintf(stderr, "Нулевой диагональный элемент в позиции (%d,%d)\n", i, i);
            exit(2);
        }
        x_calc[i] = (F[i] - s) / diag;
    }
    double t1 = now_ms();

    // 6) Невязка и погрешность
    double max_res = 0.0, max_err = 0.0;
    for (int i = 0; i < N; ++i) {
        double Ax = 0.0;
        for (int j = 0; j < N; ++j) Ax += A_orig[i][j] * x_calc[j];
        double res = fabs(Ax - F_orig[i]);
        if (res > max_res) max_res = res;
        double err = fabs(x_calc[i] - x_true[i]);
        if (err > max_err) max_err = err;
    }

    printf("max_residual   = %.3e\n", max_res);
    printf("max_error      = %.3e\n", max_err);
    printf("solve_time_ms  = %.3f\n", (t1 - t0));

    return 0;
}