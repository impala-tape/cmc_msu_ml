import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Sequence

# --------------------------- СИГНАЛ ------------------------------------------
def rect_pulse(t: np.ndarray, a: float, tau: float) -> np.ndarray:
    """
    Прямоугольный импульс xi(t):
        a,  |t| < tau/2
        0,  |t| > tau/2
    Почему strict '<': чтобы избежать неоднозначности на точной границе |t| = tau/2 в дискретной сетке.
    """
    return np.where(np.abs(t) < (tau / 2.0), a, 0.0)


def centered_time_grid(T: float, N: int) -> Tuple[np.ndarray, float]:
    """
    Центрированная равномерная сетка по времени для окна наблюдения T:
        шаг h = T/N,
        t_m = (m - N/2) * h, m = 0..N-1  (т.е. [-T/2, T/2-h])
    Зачем центрировать: симметрия картинок и спектра, удобно для задания прямоугольника вокруг нуля.
    """
    h = T / N
    m = np.arange(N)
    t = (m - N / 2) * h
    return t, h


# --------------------------- FFT-ПОМОЩНИКИ -----------------------------------
def next_pow2(n: int) -> int:
    """Ближайшая степень двойки >= n (быстрая и «гладкая» длина БПФ)."""
    p = 1
    while p < n:
        p <<= 1
    return p


def dense_spectrum_for_display(x: np.ndarray, h: float, N0: int):
    """
    "Гладкий" спектр для отрисовки: считаем FFT с нулевой доп. в КОНЦЕ временного ряда до длины N0.
    Зачем: N0 >= 6N повышает частотное разрешение сетки (больше точек на том же непрерывном спектре),
    физика не меняется — просто плотнее сетка частот.
      Возвращаем:
        S0_shift : смещённый спектр (DC в центре) для красивых симметричных графиков
        f_shift  : частоты в Гц (так же со сдвигом)
    """
    S0 = np.fft.fft(x, n=N0)
    f = np.fft.fftfreq(N0, d=h)       # частотная сетка в Гц (fs = 1/h)
    return np.fft.fftshift(S0), np.fft.fftshift(f)


def upsample_by_freq_zeropad(x: np.ndarray, N: int, n: int) -> np.ndarray:
    """
    Восстановление на более частой сетке h' = h/n (N' = n*N, T тот же) через НУЛЕВУЮ ВСТАВКУ В СПЕКТРЕ:
      1) X = FFT_N{x}
      2) fftshift(X): DC в центр
      3) симметрично вставляем нули слева/справа до длины N' = n*N (обнуляем "высокие" частоты)
      4) ifftshift -> IFFT длины N'
      5) умножаем на n (= N'/N), чтобы компенсировать нормировку NumPy (IFFT имеет фактор 1/N')
    Почему это «правильно»: это ровно идеальная bandlimited-интерполяция (sinc) в рамках DFT-модели
    с тем же окном наблюдения T. Нулевая вставка в спектре -> интерполяция во времени.
    """
    Np = n * N
    # 1) спектр исходного ряда длины N
    X = np.fft.fft(x, n=N)
    # 2) DC в центр для удобной симметричной вставки нулей
    Xc = np.fft.fftshift(X)
    # 3) вставляем нули по краям
    pad_total = Np - N
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    Xc_pad = np.pad(Xc, (pad_left, pad_right), mode="constant")
    # 4) возвращаем DC на нулевой индекс и делаем IFFT длины N'
    Xp = np.fft.ifftshift(Xc_pad)
    # 5) масштаб n = N'/N — сохраняем амплитуду исходного сигнала
    x_up = np.fft.ifft(Xp) * n
    return x_up


# --------------------------- ДЕМОНСТРАЦИЯ ------------------------------------
def run_demo(
    T: float = 1.0,                # фиксированное окно наблюдения (по заданию менять НЕ надо)
    a: float = 1.0,                # амплитуда импульса
    taus: Sequence[float] = (0.25, 0.5),  # несколько tau, чтобы увидеть зависимость спектра от ширины
    Ns: Sequence[int] = (5, 11),          # N из задания
    n_factors: Sequence[int] = (2, 4),    # факторы уточнения сетки
):
    for N in Ns:
        t, h = centered_time_grid(T, N)   # базовая сетка (шаг h = T/N)

        for tau in taus:
            x = rect_pulse(t, a, tau)     # дискретизация прямоугольника на БАЗОВОЙ сетке

            # --- СПЕКТР ДЛЯ ОТРИСОВКИ: N0 >= 6N (нулевая доп. во времени) ---
            N0 = next_pow2(6 * N)         # «гладкая» длина БПФ (часто 64/128 и т.п.)
            S0, fgrid = dense_spectrum_for_display(x, h, N0)

            # Безразмерная ось: u = f * tau (пропорционально tau^{-1} из формулировки задания)
            # Почему удобно: для прямоугольника теоретическая форма |S| ~ |sinc(pi f tau)| как функция f*tau.
            u = fgrid * tau

            # --- Графики спектра: отдельно Re, Im и |S| (без наложения кривых) ---
            plt.figure()
            plt.plot(u, S0.real)
            plt.title(f"Re(S) vs u=f·tau  (N={N}, tau={tau}, N0={N0})")
            plt.xlabel("u = f · tau")
            plt.ylabel("Re(S)")
            plt.grid(True)

            plt.figure()
            plt.plot(u, S0.imag)
            plt.title(f"Im(S) vs u=f·tau  (N={N}, tau={tau}, N0={N0})")
            plt.xlabel("u = f · tau")
            plt.ylabel("Im(S)")
            plt.grid(True)

            plt.figure()
            plt.plot(u, np.abs(S0))
            plt.title(f"|S| vs u=f·tau  (N={N}, tau={tau}, N0={N0})")
            plt.xlabel("u = f · tau")
            plt.ylabel("|S|")
            plt.grid(True)

            # --- ВОССТАНОВЛЕНИЕ НА БОЛЕЕ ЧАСТОЙ СЕТКЕ h' = h/n (N' = nN) ---
            for n in n_factors:
                x_up = upsample_by_freq_zeropad(x, N, n)   # идеальная интерполяция в рамках DFT-модели
                Np = n * N
                hp = T / Np
                mp = np.arange(Np)
                t_up = (mp - Np / 2) * hp  # центрированная «тонкая» сетка

                # Отдельный график на каждый (N, tau, n), чтобы кривые не прятались друг за друга
                plt.figure()
                # исходные дискретные точки — только маркеры
                plt.plot(t, x.real, 'o', label='исходные N точек')
                # восстановленная последовательность — линия
                plt.plot(t_up, x_up.real, label=f"восстановлено n={n} (h\'=h/{n})")
                plt.title(f"Временная область: T={T}, N={N}, tau={tau}, n={n}")
                plt.xlabel("t (с)")
                plt.ylabel("xi(t)")
                plt.grid(True)
                plt.legend()

    plt.show()


if __name__ == "__main__":
    # ВАЖНО: T фиксируем (так и нужно по заданию).
    # Меняются только N (грубая сетка) и n (во сколько раз уточняем).
    run_demo(
        T=1.0,
        a=1.0,
        taus=(0.25, 0.5),   # можно добавить свои tau, например 0.1, 0.33 и т.п.
        Ns=(5, 11),
        n_factors=(2, 4),
    )
