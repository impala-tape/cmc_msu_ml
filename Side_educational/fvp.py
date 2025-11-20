import numpy as np
import matplotlib.pyplot as plt
import os

# Создание папки для сохранения графиков
os.makedirs('Side_educational/plots', exist_ok=True)

def generate_pulse(a, tau, T, num_points=1000):
    """
    Генерация непрерывного прямоугольного импульса:
    амплитуда a, длительность tau, общая длина сигнала T.
    Центрируем импульс в середине интервала [-T/2, T/2].
    """
    t = np.linspace(-T/2, T/2, num_points, endpoint=False)
    xi = np.where(np.abs(t) < tau/2, a, 0.0)
    return t, xi

def sample_pulse(a, tau, T, N):
    """
    Дискретизация прямоугольного импульса:
    N — количество точек дискретизации на интервале [-T/2, T/2).
    Возвращает массив времён и значений сигнала.
    """
    h = T / N
    t_k = np.arange(N) * h - T/2
    xi_k = np.where(np.abs(t_k) < tau/2, a, 0.0)
    return t_k, xi_k

def compute_dft(xk, N0):
    """
    Вычисление дискретного преобразования Фурье (ДПФ)
    с нулевым дополнением до длины N0.
    Возвращает массив частот и комплексный спектр.
    """
    # Прямое ДПФ с нулевым дополнением во времени
    X = np.fft.fft(xk, n=N0)
    freqs = np.fft.fftfreq(N0, d=1.0 / N0)
    # Сдвигаем нулевую частоту в центр для красивого графика
    X_shifted = np.fft.fftshift(X)
    freqs_shifted = np.fft.fftshift(freqs)
    return freqs_shifted, X_shifted

def reconstruct_signal(xk, factor, T):
    """
    Восстановление/аппроксимация сигнала с более частым шагом по времени.
    Здесь мы не используем спектр, а просто интерполируем исходные
    дискретные отсчёты по времени.

    factor — во сколько раз уменьшить шаг (h' = h / factor).
    """
    N = len(xk)
    h = T / N

    # "грубая" временная сетка (как при дискретизации)
    t_coarse = np.arange(N) * h - T / 2

    # более частая временная сетка
    M = N * factor
    t_fine = np.arange(M) * (T / M) - T / 2

    # линейная интерполяция значений сигнала на более частой сетке
    x_fine = np.interp(t_fine, t_coarse, xk, left=0.0, right=0.0)

    return t_fine, x_fine

def plot_signal_and_recon(t_cont, xi_cont, reconstructions, N):
    """
    Построение графика исходного сигнала и его аппроксимаций
    с разными шагами по времени.

    Для наглядности:
    - исходный сигнал рисуем сплошной линией,
    - аппроксимации — только точками (без линии),
      чтобы было видно разную плотность (разный h').
    """
    plt.figure(figsize=(6, 4))

    # Исходный непрерывный прямоугольный импульс
    plt.plot(
        t_cont,
        xi_cont,
        label='Исходный сигнал $\\xi(t)$',
        color='blue',
        linewidth=2,
    )

    # Цвета для разных аппроксимаций
    colors = ['tab:orange', 'tab:green', 'tab:red']

    for idx, (t_rec, xi_rec) in enumerate(reconstructions):
        h_rec = t_rec[1] - t_rec[0]
        label = f'Аппроксимация (h={h_rec:.3f})'
        plt.plot(
            t_rec,
            xi_rec,
            linestyle='none',   # только маркеры, без линии
            marker='o',
            markersize=3,
            color=colors[idx % len(colors)],
            label=label,
        )

    plt.grid(True)
    plt.xlabel('Время t')
    plt.ylabel('Амплитуда')
    plt.title(f'Импульс и аппроксимация (N={N})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'Side_educational/plots/signal_recovery_N{N}.png')
    plt.close()


def plot_spectrum(freqs_norm, X, N):
    """
    Построение спектра сигнала:
    действительная, мнимая части и модуль спектра.
    """
    plt.figure(figsize=(7, 5))

    plt.subplot(3, 1, 1)
    plt.stem(freqs_norm, X.real, linefmt='C0-', markerfmt='C0.', basefmt='C0-')
    plt.grid(True)
    plt.title(f'Спектр сигнала при N={N} (норм. частота $f\tau$)')
    plt.ylabel('Re')

    plt.subplot(3, 1, 2)
    plt.stem(freqs_norm, X.imag, linefmt='C1-', markerfmt='C1.', basefmt='C1-')
    plt.grid(True)
    plt.ylabel('Im')

    plt.subplot(3, 1, 3)
    plt.stem(freqs_norm, np.abs(X), linefmt='C2-', markerfmt='C2.', basefmt='C2-')
    plt.grid(True)
    plt.xlabel('Норм. частота $f\tau$')
    plt.ylabel('|Спектр|')

    plt.tight_layout()
    plt.savefig(f'Side_educational/plots/spectrum_N{N}.png')
    plt.close()

def main():
    # Параметры сигнала
    a = 1.0        # Амплитуда импульса
    T = 1.0        # Общая длительность
    tau = T / 4.0  # Ширина импульса

    # Генерация идеального непрерывного импульса для графика
    t_cont, xi_cont = generate_pulse(a, tau, T)

    for N in [5, 11]:
        print(f"\nАнализ для N = {N} точек дискретизации")

        # Шаг дискретизации
        h = T / N

        # Дискретизация сигнала
        t_k, xi_k = sample_pulse(a, tau, T, N)

        # Длина спектра с нулевым дополнением (не менее 6N)
        N0 = 32 if N == 5 else 128

        # Вычисление спектра
        freqs, X = compute_dft(xi_k, N0)
        freqs_norm = freqs * tau  # нормируем на τ

        # Построение спектра
        plot_spectrum(freqs_norm, X, N)

        # Восстановление сигнала с более плотной сеткой
        print("  Восстановление с n = 2 и n = 4...")
        rec_n2 = reconstruct_signal(xi_k, factor=2, T=T)
        rec_n4 = reconstruct_signal(xi_k, factor=4, T=T)

        # Построение графика сигналов
        plot_signal_and_recon(t_cont, xi_cont, [rec_n2, rec_n4], N)

if __name__ == '__main__':
    main()
