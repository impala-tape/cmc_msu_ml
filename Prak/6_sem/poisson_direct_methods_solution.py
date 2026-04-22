"""Решение двумерного уравнения Пуассона прямым методом.

Задача:
    Δu = -f(x, y),   0 < x < 1, 0 < y < 1

Граничные условия:
    u(0, y) = φ1(y)
    u(1, y) = φ2(y)
    u(x, 0) = 0
    u(x, 1) = 0

Идея прямого метода:
1. По переменной y используем дискретный синусный базис,
   который является базисом собственных функций для одномерного
   разностного оператора с нулевыми значениями на границах y = 0 и y = 1.
2. Разлагаем правую часть и граничные данные по этому базису.
3. Для каждого номера гармоники k получаем отдельную одномерную
   трёхдиагональную задачу по x.
4. Каждую такую задачу решаем методом прогонки.
5. После этого восстанавливаем двумерное решение суммой по синусным модам.

Для учебной прозрачной реализации быстрый DST здесь не используется.
Коэффициенты по синусному базису считаются напрямую через дискретные суммы.
Это медленнее, чем FFT/DST, но полностью отражает математику метода
и остаётся удобным для умеренных сеток 16, 32, 64, 128.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DOMAIN_LENGTH_X = 1.0
DOMAIN_LENGTH_Y = 1.0
CONVERGENCE_GRID_SIZES = (16, 32, 64, 128)
VISUALIZATION_GRID_SIZE = 64
CONTOUR_LEVELS = 30
PLOT_DPI = 150


def exact_solution(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Возвращает аналитическое решение u(x, y) = exp(x) * sin(pi * y)."""

    return np.exp(x) * np.sin(np.pi * y)


def rhs_function(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Возвращает правую часть f(x, y) для тестовой задачи.

    Для выбранного точного решения:
        u(x, y) = exp(x) * sin(pi * y)

    имеем:
        u_xx = exp(x) * sin(pi * y)
        u_yy = -pi^2 * exp(x) * sin(pi * y)
        Δu = (1 - pi^2) * exp(x) * sin(pi * y)

    Так как в задаче требуется Δu = -f, получаем:
        f(x, y) = (pi^2 - 1) * exp(x) * sin(pi * y)
    """

    return (np.pi**2 - 1.0) * np.exp(x) * np.sin(np.pi * y)


def phi_left(y: np.ndarray) -> np.ndarray:
    """Левая граница: φ1(y) = u(0, y) = sin(pi * y)."""

    return np.sin(np.pi * y)


def phi_right(y: np.ndarray) -> np.ndarray:
    """Правая граница: φ2(y) = u(1, y) = e * sin(pi * y)."""

    return np.e * np.sin(np.pi * y)


def build_grid(n: int, m: int) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray]:
    """Строит равномерную сетку и возвращает узлы и двумерные массивы координат."""

    if n < 2 or m < 2:
        raise ValueError("Для двумерной задачи нужны хотя бы две ячейки по каждому направлению.")

    hx = DOMAIN_LENGTH_X / n
    hy = DOMAIN_LENGTH_Y / m

    x = np.linspace(0.0, DOMAIN_LENGTH_X, n + 1)
    y = np.linspace(0.0, DOMAIN_LENGTH_Y, m + 1)
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    return x, y, hx, hy, x_grid, y_grid


def build_sine_basis(m: int) -> np.ndarray:
    """Строит матрицу дискретного синусного базиса по y.

    Индексация:
        j = 1, ..., M - 1  - внутренние узлы по y
        k = 1, ..., M - 1  - номера мод

    Базисная функция на сетке:
        μ_k(j) = sin(pi * k * j / M)

    Возвращаемая матрица basis имеет размер (M - 1, M - 1),
    где basis[j - 1, k - 1] = μ_k(j).
    """

    if m < 2:
        raise ValueError("Число разбиений M должно быть не меньше 2.")

    j_indices = np.arange(1, m, dtype=np.float64)
    k_indices = np.arange(1, m, dtype=np.float64)
    return np.sin(np.pi * np.outer(j_indices, k_indices) / m)


def compute_discrete_eigenvalues(m: int, hy: float) -> np.ndarray:
    """Вычисляет собственные значения дискретного оператора по y.

    Для каждой моды k = 1, ..., M - 1:
        λ_k = (4 / hy^2) * sin^2(pi * k * hy / 2)

    Так как DOMAIN_LENGTH_Y = 1 и hy = 1 / M, формула совпадает с
    классическим выражением для дискретного оператора Лапласа
    с нулевыми граничными условиями.
    """

    k_indices = np.arange(1, m, dtype=np.float64)
    return (4.0 / hy**2) * np.sin(np.pi * k_indices * hy / 2.0) ** 2


def forward_sine_transform(values: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Вычисляет коэффициенты по дискретному синусному базису.

    Формула для одномерного набора значений g_j, j = 1, ..., M - 1:
        g_hat_k = (2 / M) * sum_{j=1}^{M-1} g_j * sin(pi * k * j / M)

    Здесь используется точная дискретная ортогональность синусов.

    Аргументы:
        values :
            либо вектор длины M - 1,
            либо матрица размера (N - 1, M - 1), где каждая строка
            соответствует фиксированному x_i и содержит значения по y.
        basis :
            матрица размера (M - 1, M - 1), построенная функцией build_sine_basis.
    """

    array = np.asarray(values, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[0] != basis.shape[1]:
        raise ValueError("Матрица базиса должна быть квадратной.")

    m_minus_one = basis.shape[0]
    normalization = 2.0 / (m_minus_one + 1)

    if array.ndim == 1:
        if array.size != m_minus_one:
            raise ValueError("Размер вектора values не совпадает с числом внутренних узлов по y.")
        return normalization * (array @ basis)

    if array.ndim == 2:
        if array.shape[1] != m_minus_one:
            raise ValueError("У матрицы values число столбцов должно быть равно M - 1.")
        return normalization * (array @ basis)

    raise ValueError("values должен быть либо вектором, либо матрицей.")


def inverse_sine_transform(coefficients: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Восстанавливает значения по коэффициентам синусного разложения.

    Если коэффициенты g_hat_k уже найдены, то восстановление выполняется как
        g_j = sum_{k=1}^{M-1} g_hat_k * sin(pi * k * j / M)

    Для матрицы коэффициентов восстановление выполняется построчно.
    """

    array = np.asarray(coefficients, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[0] != basis.shape[1]:
        raise ValueError("Матрица базиса должна быть квадратной.")

    m_minus_one = basis.shape[0]

    if array.ndim == 1:
        if array.size != m_minus_one:
            raise ValueError("Размер вектора coefficients не совпадает с числом мод.")
        return basis @ array

    if array.ndim == 2:
        if array.shape[1] != m_minus_one:
            raise ValueError("У матрицы coefficients число столбцов должно быть равно M - 1.")
        return array @ basis.T

    raise ValueError("coefficients должен быть либо вектором, либо матрицей.")


def thomas_algorithm(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Решает трёхдиагональную систему методом прогонки.

    Система имеет вид:
        a_i * x_{i-1} + b_i * x_i + c_i * x_{i+1} = d_i

    Здесь:
        a[0] = 0,
        c[n - 1] = 0,
        длины всех массивов одинаковы и равны n.
    """

    lower = np.asarray(a, dtype=np.float64)
    diagonal = np.asarray(b, dtype=np.float64)
    upper = np.asarray(c, dtype=np.float64)
    rhs = np.asarray(d, dtype=np.float64)

    if lower.ndim != 1 or diagonal.ndim != 1 or upper.ndim != 1 or rhs.ndim != 1:
        raise ValueError("Все аргументы метода прогонки должны быть одномерными массивами.")

    n = rhs.size
    if n == 0:
        raise ValueError("Пустую систему методом прогонки решать нельзя.")
    if not (lower.size == diagonal.size == upper.size == n):
        raise ValueError("Массивы a, b, c, d должны иметь одинаковую длину.")
    if abs(lower[0]) > 1e-14:
        raise ValueError("Для метода прогонки требуется a[0] = 0.")
    if abs(upper[-1]) > 1e-14:
        raise ValueError("Для метода прогонки требуется c[-1] = 0.")

    modified_upper = np.zeros(n, dtype=np.float64)
    modified_rhs = np.zeros(n, dtype=np.float64)

    if abs(diagonal[0]) < 1e-14:
        raise ZeroDivisionError("Нулевой главный элемент в начале прямого хода.")

    modified_upper[0] = upper[0] / diagonal[0] if n > 1 else 0.0
    modified_rhs[0] = rhs[0] / diagonal[0]

    for i in range(1, n):
        denominator = diagonal[i] - lower[i] * modified_upper[i - 1]
        if abs(denominator) < 1e-14:
            raise ZeroDivisionError(f"На шаге {i} возник нулевой знаменатель в методе прогонки.")

        modified_upper[i] = upper[i] / denominator if i < n - 1 else 0.0
        modified_rhs[i] = (rhs[i] - lower[i] * modified_rhs[i - 1]) / denominator

    solution = np.zeros(n, dtype=np.float64)
    solution[-1] = modified_rhs[-1]

    for i in range(n - 2, -1, -1):
        solution[i] = modified_rhs[i] - modified_upper[i] * solution[i + 1]

    return solution


def solve_mode_problems(
    n: int,
    hx: float,
    rhs_hat: np.ndarray,
    phi_left_hat: np.ndarray,
    phi_right_hat: np.ndarray,
    eigenvalues: np.ndarray,
) -> np.ndarray:
    """Решает все одномерные задачи по x для коэффициентов Фурье."""

    if rhs_hat.shape != (n - 1, eigenvalues.size):
        raise ValueError("Размер rhs_hat должен быть равен (N - 1, M - 1).")
    if phi_left_hat.shape != eigenvalues.shape or phi_right_hat.shape != eigenvalues.shape:
        raise ValueError("Размеры коэффициентов границ должны совпадать с числом мод.")

    number_of_modes = eigenvalues.size
    solution_hat = np.zeros((n + 1, number_of_modes), dtype=np.float64)
    solution_hat[0, :] = phi_left_hat
    solution_hat[-1, :] = phi_right_hat

    alpha = 1.0 / hx**2
    interior_size = n - 1

    for mode_index, lambda_k in enumerate(eigenvalues):
        lower = np.full(interior_size, alpha, dtype=np.float64)
        diagonal = np.full(interior_size, -2.0 * alpha - lambda_k, dtype=np.float64)
        upper = np.full(interior_size, alpha, dtype=np.float64)
        rhs = -rhs_hat[:, mode_index].astype(np.float64, copy=True)

        # Для первого и последнего внутреннего узла коэффициенты при
        # известных граничных значениях переносятся в правую часть.
        lower[0] = 0.0
        upper[-1] = 0.0
        rhs[0] -= alpha * phi_left_hat[mode_index]
        rhs[-1] -= alpha * phi_right_hat[mode_index]

        solution_hat[1:-1, mode_index] = thomas_algorithm(lower, diagonal, upper, rhs)

    return solution_hat


def reconstruct_solution(
    solution_hat: np.ndarray,
    basis: np.ndarray,
    phi_left_values: np.ndarray,
    phi_right_values: np.ndarray,
) -> np.ndarray:
    """Восстанавливает двумерное решение на полной сетке."""

    if solution_hat.ndim != 2:
        raise ValueError("solution_hat должен быть матрицей размера (N + 1, M - 1).")
    if basis.ndim != 2 or basis.shape[0] != basis.shape[1]:
        raise ValueError("basis должен быть квадратной матрицей.")

    n_plus_one = solution_hat.shape[0]
    m_minus_one = basis.shape[0]
    m_plus_one = m_minus_one + 2

    if phi_left_values.size != m_plus_one or phi_right_values.size != m_plus_one:
        raise ValueError("Граничные значения по y должны задаваться на полной сетке из M + 1 узлов.")

    solution = np.zeros((n_plus_one, m_plus_one), dtype=np.float64)
    solution[:, 1:-1] = inverse_sine_transform(solution_hat, basis)

    # Явно задаём граничные условия на полной сетке.
    solution[0, :] = phi_left_values
    solution[-1, :] = phi_right_values
    solution[:, 0] = 0.0
    solution[:, -1] = 0.0
    return solution


def solve_poisson_direct_method(n: int, m: int) -> dict[str, np.ndarray | float | int]:
    """Полностью решает разностную задачу прямым методом."""

    x, y, hx, hy, x_grid, y_grid = build_grid(n, m)

    basis = build_sine_basis(m)
    eigenvalues = compute_discrete_eigenvalues(m, hy)

    exact = exact_solution(x_grid, y_grid)
    rhs = rhs_function(x_grid, y_grid)

    phi_left_values = phi_left(y)
    phi_right_values = phi_right(y)

    # В разложения по синусам входят только внутренние узлы j = 1, ..., M - 1.
    phi_left_hat = forward_sine_transform(phi_left_values[1:-1], basis)
    phi_right_hat = forward_sine_transform(phi_right_values[1:-1], basis)
    rhs_hat = forward_sine_transform(rhs[1:-1, 1:-1], basis)

    solution_hat = solve_mode_problems(
        n=n,
        hx=hx,
        rhs_hat=rhs_hat,
        phi_left_hat=phi_left_hat,
        phi_right_hat=phi_right_hat,
        eigenvalues=eigenvalues,
    )

    numerical = reconstruct_solution(
        solution_hat=solution_hat,
        basis=basis,
        phi_left_values=phi_left_values,
        phi_right_values=phi_right_values,
    )

    error = np.abs(numerical - exact)
    delta = float(np.max(error))

    return {
        "N": n,
        "M": m,
        "x": x,
        "y": y,
        "hx": hx,
        "hy": hy,
        "X": x_grid,
        "Y": y_grid,
        "exact": exact,
        "numerical": numerical,
        "error": error,
        "delta": delta,
        "basis": basis,
        "eigenvalues": eigenvalues,
        "solution_hat": solution_hat,
    }


def build_convergence_table(grid_sizes: tuple[int, ...]) -> list[dict[str, float | int | None]]:
    """Считает ошибки на нескольких сетках и оценивает порядок сходимости."""

    rows: list[dict[str, float | int | None]] = []
    previous_delta: float | None = None
    previous_h: float | None = None

    for grid_size in grid_sizes:
        result = solve_poisson_direct_method(grid_size, grid_size)
        current_h = float(result["hx"])
        current_delta = float(result["delta"])

        order: float | None = None
        if previous_delta is not None and previous_h is not None:
            order = np.log(previous_delta / current_delta) / np.log(previous_h / current_h)

        rows.append(
            {
                "N": grid_size,
                "M": grid_size,
                "h": current_h,
                "delta": current_delta,
                "order": order,
            }
        )

        previous_delta = current_delta
        previous_h = current_h

    return rows


def print_convergence_table(rows: list[dict[str, float | int | None]]) -> None:
    """Печатает таблицу ошибок и оценок порядка сходимости."""

    print("\n=== Сходимость на последовательности сеток ===")
    print(f"{'N=M':>8} {'h':>12} {'delta':>18} {'порядок':>12}")
    print("-" * 56)

    for row in rows:
        order_text = "-" if row["order"] is None else f"{float(row['order']):.4f}"
        print(
            f"{int(row['N']):>8d} "
            f"{float(row['h']):>12.6e} "
            f"{float(row['delta']):>18.6e} "
            f"{order_text:>12}"
        )


def get_output_directory() -> Path:
    """Возвращает директорию, куда сохраняются рисунки."""

    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd()


def plot_solution_fields(result: dict[str, np.ndarray | float | int], output_dir: Path) -> Path:
    """Строит три карты: точное решение, численное решение и абсолютную ошибку."""

    x_grid = np.asarray(result["X"])
    y_grid = np.asarray(result["Y"])
    exact = np.asarray(result["exact"])
    numerical = np.asarray(result["numerical"])
    error = np.asarray(result["error"])

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.4), constrained_layout=True)
    fields = (
        ("Аналитическое решение", exact, "viridis"),
        ("Численное решение", numerical, "viridis"),
        ("Абсолютная ошибка", error, "magma"),
    )

    for axis, (title, field, cmap) in zip(axes, fields):
        contour = axis.contourf(x_grid, y_grid, field, levels=CONTOUR_LEVELS, cmap=cmap)
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        figure.colorbar(contour, ax=axis)

    output_path = output_dir / "poisson_direct_fields.png"
    figure.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    return output_path


def plot_convergence(rows: list[dict[str, float | int | None]], output_dir: Path) -> Path:
    """Строит график зависимости ошибки от шага сетки."""

    h_values = np.array([float(row["h"]) for row in rows], dtype=np.float64)
    delta_values = np.array([float(row["delta"]) for row in rows], dtype=np.float64)

    figure, axis = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    axis.plot(h_values, delta_values, marker="o", linewidth=1.8, label="Численная ошибка")

    reference_constant = delta_values[0] / (h_values[0] ** 2)
    reference_curve = reference_constant * h_values**2
    axis.plot(h_values, reference_curve, linestyle="--", linewidth=1.5, label="Опорная кривая O(h^2)")

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Шаг сетки h")
    axis.set_ylabel("delta = max |u_num - u_exact|")
    axis.set_title("Сходимость прямого метода")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()

    output_path = output_dir / "poisson_direct_convergence.png"
    figure.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    return output_path


def show_plots_if_possible() -> None:
    """Показывает графики только на интерактивном backend."""

    backend_name = plt.get_backend().lower()
    if "agg" in backend_name:
        print("Используется неинтерактивный backend matplotlib, поэтому графики сохранены только в файлы.")
        plt.close("all")
        return

    plt.show()


def main() -> None:
    """Запускает решение тестовой задачи, строит графики и печатает таблицы."""

    output_dir = get_output_directory()

    print("Решение двумерного уравнения Пуассона прямым методом")
    print("Тестовая задача выбрана так, чтобы точное решение было известно явно:")
    print("u_exact(x, y) = exp(x) * sin(pi * y)")
    print("f(x, y) = (pi^2 - 1) * exp(x) * sin(pi * y)")
    print("Левая граница:  phi1(y) = sin(pi * y)")
    print("Правая граница: phi2(y) = e * sin(pi * y)")
    print(
        "Коэффициенты по синусному базису считаются вручную по дискретной формуле "
        "ортогонального разложения, без использования готового DST."
    )

    visualization_result = solve_poisson_direct_method(VISUALIZATION_GRID_SIZE, VISUALIZATION_GRID_SIZE)
    print(
        f"\nСетка для визуализации: N = M = {VISUALIZATION_GRID_SIZE}, "
        f"delta = {float(visualization_result['delta']):.6e}"
    )

    convergence_rows = build_convergence_table(CONVERGENCE_GRID_SIZES)
    print_convergence_table(convergence_rows)

    fields_plot_path = plot_solution_fields(visualization_result, output_dir)
    convergence_plot_path = plot_convergence(convergence_rows, output_dir)

    print("\n=== Сохранённые графики ===")
    print(fields_plot_path)
    print(convergence_plot_path)

    if convergence_rows[-1]["order"] is not None:
        print(f"\nОценка порядка сходимости на последнем шаге: {float(convergence_rows[-1]['order']):.4f}")

    print(
        "Величина delta = max_{i,j} |u_num(i,j) - u_exact(x_i, y_j)| "
        "показывает максимальную абсолютную ошибку численного решения на всей сетке."
    )
    print(
        "Метод называется прямым, потому что после разложения по собственным функциям "
        "двумерная задача сразу сводится к набору независимых одномерных задач, "
        "которые решаются без итерационного процесса."
    )

    show_plots_if_possible()


if __name__ == "__main__":
    main()
