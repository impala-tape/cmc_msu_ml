#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_ITER 512
#define EPS 1e-14

/* чтение квадратной матрицы из CSV, возвращает плоский массив A (строчно),
   размер n записывает в *n_out */
double* read_matrix_csv(const char* filename, int *n_out)
{
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(1);
    }

    char buffer[10000];

    /* первая строка — считаем число столбцов по запятым */
    if (!fgets(buffer, sizeof(buffer), f)) {
        fprintf(stderr, "Error reading first line of %s\n", filename);
        fclose(f);
        exit(1);
    }

    int count_commas = 0;
    for (char *p = buffer; *p; ++p) {
        if (*p == ',') count_commas++;
        if (*p == '\n') break;
    }
    int n = count_commas + 1;

    double *A = (double*)malloc(n * n * sizeof(double));
    if (!A) {
        fprintf(stderr, "Memory allocation error\n");
        fclose(f);
        exit(1);
    }

    /* парсим первую строку */
    char *token = strtok(buffer, ",\n");
    for (int j = 0; j < n; ++j) {
        if (!token) {
            fprintf(stderr, "Parse error in first line\n");
            free(A);
            fclose(f);
            exit(1);
        }
        A[0 * n + j] = atof(token);
        token = strtok(NULL, ",\n");
    }

    /* остальные строки */
    for (int i = 1; i < n; ++i) {
        if (!fgets(buffer, sizeof(buffer), f)) {
            fprintf(stderr, "Unexpected end of file at line %d\n", i+1);
            free(A);
            fclose(f);
            exit(1);
        }
        char *tok = strtok(buffer, ",\n");
        for (int j = 0; j < n; ++j) {
            if (!tok) {
                fprintf(stderr, "Parse error at line %d\n", i+1);
                free(A);
                fclose(f);
                exit(1);
            }
            A[i * n + j] = atof(tok);
            tok = strtok(NULL, ",\n");
        }
    }

    fclose(f);
    *n_out = n;
    return A;
}

/* произведение Bx, где B = I + A */
void matvec_B(const double *A, const double *x, double *y, int n)
{
    for (int i = 0; i < n; ++i) {
        double sum = x[i]; /* единичная матрица */
        const double *Ai = A + i * n;
        for (int j = 0; j < n; ++j) {
            sum += Ai[j] * x[j];
        }
        y[i] = sum;
    }
}

/* евклидова норма вектора */
double norm2(const double *v, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        s += v[i] * v[i];
    }
    return sqrt(s);
}

/* одношаговый метод Чебышёва для Bx = F,
 * B = I + A, спектр B в [m, M], N итераций.
 * логируем относительную ошибку в файл log (iter, rel_err).
 */
void one_step_chebyshev(const double *A, const double *F, double *x,
                        int n, double m, double M, int N,
                        const double *x_true, double norm_true,
                        FILE *log)
{
    double *r  = (double*)malloc(n * sizeof(double));
    double *Bx = (double*)malloc(n * sizeof(double));
    double *alpha = (double*)malloc((N + 1) * sizeof(double));
    if (!r || !Bx || !alpha) {
        fprintf(stderr, "Memory allocation error in one_step_chebyshev\n");
        exit(1);
    }

    /* параметры Чебышёва: alpha_k = 2 / ((M+m) - (M-m) cos(theta_k)) */
    for (int k = 1; k <= N; ++k) {
        double theta = (2.0 * k - 1.0) * M_PI / (2.0 * N);
        alpha[k] = 2.0 / ((M + m) - (M - m) * cos(theta));
    }

    /* x уже должен быть нулевым снаружи */

    for (int k = 1; k <= N; ++k) {
        matvec_B(A, x, Bx, n);
        for (int i = 0; i < n; ++i) {
            r[i] = F[i] - Bx[i];
        }
        for (int i = 0; i < n; ++i) {
            x[i] += alpha[k] * r[i];
        }

        if (log) {
            double diff_norm = 0.0;
            for (int i = 0; i < n; ++i) {
                double d = x[i] - x_true[i];
                diff_norm += d * d;
            }
            diff_norm = sqrt(diff_norm);
            double rel = diff_norm / norm_true;
            fprintf(log, "%d,%.16e\n", k, rel);
        }
    }

    free(r);
    free(Bx);
    free(alpha);
}

/* двухшаговый (трёхслойный) метод Чебышёва для Bx = F,
 * реализация через рекурренты по delta_k, alpha_k, beta_k.
 * логируем относительную ошибку по итерациям.
 */
void two_step_chebyshev(const double *A, const double *F, double *x,
                        int n, double m, double M, int N,
                        const double *x_true, double norm_true,
                        FILE *log)
{
    double *x_prev = (double*)calloc(n, sizeof(double));
    double *r      = (double*)malloc(n * sizeof(double));
    double *Bx     = (double*)malloc(n * sizeof(double));
    long double *delta = (long double*)malloc((N + 2) * sizeof(long double));
    double *alpha  = (double*)malloc((N + 2) * sizeof(double));
    double *beta   = (double*)malloc((N + 2) * sizeof(double));

    if (!x_prev || !r || !Bx || !delta || !alpha || !beta) {
        fprintf(stderr, "Memory allocation error in two_step_chebyshev\n");
        exit(1);
    }

    long double theta = ((long double)M + (long double)m) /
                        ((long double)M - (long double)m);

    delta[0] = 0.0L;
    delta[1] = 1.0L / theta;
    for (int k = 1; k < N; ++k) {
        delta[k + 1] = 1.0L / (2.0L * theta - delta[k]);
    }

    for (int k = 0; k < N; ++k) {
        alpha[k + 1] = (double)(4.0L * delta[k + 1] /
                                ((long double)(M - m)));
    }

    beta[1] = 0.0;
    for (int k = 1; k < N; ++k) {
        beta[k + 1] = (double)(- delta[k] * delta[k + 1]);
    }

    /* начальный остаток r^0 = F - Bx^0 = F, т.к. x^0 = 0 */
    for (int i = 0; i < n; ++i) {
        r[i] = F[i];
    }

    /* первая итерация: x^1 = x^0 + alpha_1 r^0 */
    for (int i = 0; i < n; ++i) {
        x[i] = alpha[1] * r[i];
    }

    if (log) {
        double diff_norm = 0.0;
        for (int i = 0; i < n; ++i) {
            double d = x[i] - x_true[i];
            diff_norm += d * d;
        }
        diff_norm = sqrt(diff_norm);
        double rel = diff_norm / norm_true;
        fprintf(log, "%d,%.16e\n", 1, rel);
    }

    for (int k = 1; k < N; ++k) {
        /* r^k = F - B x^k */
        matvec_B(A, x, Bx, n);
        for (int i = 0; i < n; ++i) {
            r[i] = F[i] - Bx[i];
        }

        double a = alpha[k + 1];
        double b = beta[k + 1];

        for (int i = 0; i < n; ++i) {
            double x_i = x[i];
            double new_x = x_i + a * r[i] + b * (x_i - x_prev[i]);
            x_prev[i] = x_i;
            x[i] = new_x;
        }

        if (log) {
            double diff_norm = 0.0;
            for (int i = 0; i < n; ++i) {
                double d = x[i] - x_true[i];
                diff_norm += d * d;
            }
            diff_norm = sqrt(diff_norm);
            double rel = diff_norm / norm_true;
            fprintf(log, "%d,%.16e\n", k + 1, rel);
        }
    }

    free(x_prev);
    free(r);
    free(Bx);
    free(delta);
    free(alpha);
    free(beta);
}

int main(void)
{
    const char *filename = "SLAU_var_5.csv";
    int n = 0;
    double *A = read_matrix_csv(filename, &n);

    double *x_true = (double*)malloc(n * sizeof(double));
    double *F      = (double*)malloc(n * sizeof(double));
    double *x1     = (double*)calloc(n, sizeof(double));
    double *x2     = (double*)calloc(n, sizeof(double));

    if (!A || !x_true || !F || !x1 || !x2) {
        fprintf(stderr, "Memory allocation error in main\n");
        exit(1);
    }

    srand(0);
    for (int i = 0; i < n; ++i) {
        double r = (double)rand() / (double)RAND_MAX;
        x_true[i] = 2.0 * r - 1.0;
    }

    /* F = (I + A) x_true */
    matvec_B(A, x_true, F, n);

    double norm_true = norm2(x_true, n);

    /* спектр B = I + A: заранее посчитанные оценки для этого варианта */
    double lambda_min_B = 49.5797;
    double lambda_max_B = 86.1512;

    /* одношаговый метод: пишем кривую ошибки */
    FILE *f1 = fopen("one_step.csv", "w");
    if (!f1) {
        fprintf(stderr, "Cannot open one_step.csv for writing\n");
        exit(1);
    }
    fprintf(f1, "iter,rel_err\n");

    one_step_chebyshev(A, F, x1, n,
                       lambda_min_B, lambda_max_B,
                       MAX_ITER,
                       x_true, norm_true, f1);

    fclose(f1);

    /* двухшаговый метод: пишем кривую ошибки */
    FILE *f2 = fopen("two_step.csv", "w");
    if (!f2) {
        fprintf(stderr, "Cannot open two_step.csv for writing\n");
        exit(1);
    }
    fprintf(f2, "iter,rel_err\n");

    two_step_chebyshev(A, F, x2, n,
                       lambda_min_B, lambda_max_B,
                       MAX_ITER,
                       x_true, norm_true, f2);

    fclose(f2);

    /* финальные ошибки и норма разности решений */
    double diff1 = 0.0, diff2 = 0.0, diff12 = 0.0;
    for (int i = 0; i < n; ++i) {
        double d1 = x1[i] - x_true[i];
        double d2 = x2[i] - x_true[i];
        double d12 = x1[i] - x2[i];
        diff1  += d1 * d1;
        diff2  += d2 * d2;
        diff12 += d12 * d12;
    }
    diff1  = sqrt(diff1)  / norm_true;
    diff2  = sqrt(di  ff2)  / norm_true;
    diff12 = sqrt(diff12);

    printf("Final rel error (one-step):  %.3e\n", diff1);
    printf("Final rel error (two-step):  %.3e\n", diff2);
    printf("||x_one - x_two||_2:         %.3e\n", diff12);

    free(A);
    free(x_true);
    free(F);
    free(x1);
    free(x2);

    return 0;
}
