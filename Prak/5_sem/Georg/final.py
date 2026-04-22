import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# ПАРАМЕТРЫ
# ============================================================================
L = 1.0
Nx = 20
h = L / Nx
x = np.linspace(0, L, Nx+1)

# Параметры неоднородного стержня
rho0, rho1 = 1.0, 0.3
k0, k1 = 1.0, 0.3

def rho_func(x):
    return rho0 + rho1 / (1 + x)

def k_func(x):
    return k0 + k1 / (1 + x)

# Средние значения для однородного случая
rho_vals = rho_func(x)
k_vals = k_func(x)
P0 = (rho_func(0) + rho_func(L)) / 2
s0 = (k_func(0) + k_func(L)) / 2
a = np.sqrt(s0 / P0)  # скорость для однородного
print(f"P0 = {P0:.4f}, s0 = {s0:.4f}, a = {a:.4f}")

# Время T
T = 5.0 / (2.0 * a)
print(f"T = {T:.4f}")

# Начальные условия
def u0(x):
    return np.sin(np.pi * x)

def v0(x):
    return 0.0 * x

# ============================================================================
# 1. ТОЧНОЕ РЕШЕНИЕ для однородного стержня
# ============================================================================
def exact_homogeneous(x, t):
    """Точное решение: u_tt = a^2 u_xx, u(x,0)=sin(pi*x), u_t(x,0)=0"""
    return np.sin(np.pi * x) * np.cos(np.pi * a * t)

# ============================================================================
# 2. ЧИСЛЕННОЕ РЕШЕНИЕ для ОДНОРОДНОГО стержня (ПРОСТАЯ СХЕМА)
# ============================================================================
print("\n" + "="*60)
print("ОДНОРОДНЫЙ СТЕРЖЕНЬ (численное vs точное)")
print("="*60)

# Для однородного: u_tt = a^2 u_xx
# Стабильность: tau <= h/a (условие Куранта)
tau_h = 0.95 * h / a  # запас 5% для стабильности
Nt_h = int(T / tau_h) + 1
tau_h = T / Nt_h  # корректируем для точного попадания в T
print(f"tau_h = {tau_h:.6f}, Nt_h = {Nt_h}")

# Прямоугольная сетка
u_prev = u0(x)
u_curr = np.zeros(Nx+1)

# Первый шаг по времени (разложение Тейлора)
for i in range(1, Nx):
    u_xx = (u_prev[i+1] - 2*u_prev[i] + u_prev[i-1]) / h**2
    u_curr[i] = u_prev[i] + tau_h * v0(x[i]) + 0.5 * (tau_h * a)**2 * u_xx

u_curr[0] = 0.0
u_curr[Nx] = 0.0

# Трёхслойная схема для волнового уравнения
solutions_homog = [u_prev.copy()]
times_homog = [0.0]

for n in range(1, Nt_h + 1):
    u_next = np.zeros(Nx+1)
    
    for i in range(1, Nx):
        u_xx = (u_curr[i+1] - 2*u_curr[i] + u_curr[i-1]) / h**2
        u_next[i] = 2*u_curr[i] - u_prev[i] + (tau_h * a)**2 * u_xx
    
    # Граничные условия
    u_next[0] = 0.0
    u_next[Nx] = 0.0
    
    solutions_homog.append(u_next.copy())
    times_homog.append(n * tau_h)
    
    u_prev, u_curr = u_curr, u_next

solutions_homog = np.array(solutions_homog)
times_homog = np.array(times_homog)

# ============================================================================
# 3. ЧИСЛЕННОЕ РЕШЕНИЕ для НЕОДНОРОДНОГО стержня
# ============================================================================
print("\n" + "="*60)
print("НЕОДНОРОДНЫЙ СТЕРЖЕНЬ")
print("="*60)

# Максимальная скорость волны в неоднородном стержне
c_max = np.sqrt(np.max(k_vals) / np.min(rho_vals))
tau_nh = 0.95 * h / c_max
Nt_nh = int(T / tau_nh) + 1
tau_nh = T / Nt_nh
print(f"c_max = {c_max:.4f}, tau_nh = {tau_nh:.6f}, Nt_nh = {Nt_nh}")

# Начальные значения
u_prev = u0(x)
u_curr = np.zeros(Nx+1)

# Первый шаг для неоднородного
for i in range(1, Nx):
    # Для i=1..Nx-1 используем центральные разности
    # Производная k(x)
    if i == 1:
        k_der = (k_vals[2] - k_vals[0]) / (2*h)
    elif i == Nx-1:
        k_der = (k_vals[Nx] - k_vals[Nx-2]) / (2*h)
    else:
        k_der = (k_vals[i+1] - k_vals[i-1]) / (2*h)
    
    u_x = (u_prev[i+1] - u_prev[i-1]) / (2*h)
    u_xx = (u_prev[i+1] - 2*u_prev[i] + u_prev[i-1]) / h**2
    
    # Правая часть: (k*u_x)_x = k*u_xx + k_x*u_x
    rhs = (k_vals[i] * u_xx + k_der * u_x) / rho_vals[i]
    u_curr[i] = u_prev[i] + tau_nh * v0(x[i]) + 0.5 * tau_nh**2 * rhs

u_curr[0] = 0.0
u_curr[Nx] = 0.0

# Трёхслойная схема
solutions_nhomog = [u_prev.copy()]
times_nhomog = [0.0]

for n in range(1, Nt_nh + 1):
    u_next = np.zeros(Nx+1)
    
    for i in range(1, Nx):
        # Производная k(x)
        if i == 1:
            k_der = (k_vals[2] - k_vals[0]) / (2*h)
        elif i == Nx-1:
            k_der = (k_vals[Nx] - k_vals[Nx-2]) / (2*h)
        else:
            k_der = (k_vals[i+1] - k_vals[i-1]) / (2*h)
        
        u_x = (u_curr[i+1] - u_curr[i-1]) / (2*h)
        u_xx = (u_curr[i+1] - 2*u_curr[i] + u_curr[i-1]) / h**2
        
        rhs = (k_vals[i] * u_xx + k_der * u_x) / rho_vals[i]
        u_next[i] = 2*u_curr[i] - u_prev[i] + tau_nh**2 * rhs
    
    u_next[0] = 0.0
    u_next[Nx] = 0.0
    
    solutions_nhomog.append(u_next.copy())
    times_nhomog.append(n * tau_nh)
    
    u_prev, u_curr = u_curr, u_next

solutions_nhomog = np.array(solutions_nhomog)
times_nhomog = np.array(times_nhomog)

# ============================================================================
# 4. ВИЗУАЛИЗАЦИЯ и АНАЛИЗ
# ============================================================================
plt.figure(figsize=(15, 10))

# 1. Сравнение в момент T (однородный случай)
plt.subplot(2, 3, 1)
u_num_h = solutions_homog[-1, :]
u_exact = exact_homogeneous(x, T)
plt.plot(x, u_num_h, 'b-', linewidth=2, label='Численное')
plt.plot(x, u_exact, 'r--', linewidth=2, label='Точное')
plt.xlabel('x')
plt.ylabel('u(x, T)')
plt.title(f'ОДНОРОДНЫЙ: сравнение в T={T:.4f}')
plt.legend()
plt.grid(True)

# 2. Погрешность однородного
plt.subplot(2, 3, 2)
error = np.abs(u_num_h - u_exact)
plt.plot(x, error, 'g-', linewidth=2)
plt.xlabel('x')
plt.ylabel('|u_num - u_exact|')
plt.title(f'Погрешность (max={error.max():.2e})')
plt.grid(True)
plt.yscale('log' if error.max() > 0 else 'linear')

# 3. Неоднородный в момент T
plt.subplot(2, 3, 3)
u_num_nh = solutions_nhomog[-1, :]
plt.plot(x, u_num_nh, 'b-', linewidth=2, label='Численное')
plt.plot(x, u_exact, 'r--', linewidth=2, label='Однородное точное', alpha=0.5)
plt.xlabel('x')
plt.ylabel('u(x, T)')
plt.title(f'НЕОДНОРОДНЫЙ в T={T:.4f}')
plt.legend()
plt.grid(True)

# 4. Движение точки x=0.5 (однородное точное vs численное)
plt.subplot(2, 3, 4)
x0_idx = Nx // 2
# Однородное точное
t_fine = np.linspace(0, T, 500)
u_exact_x0 = exact_homogeneous(0.5, t_fine)
plt.plot(t_fine, u_exact_x0, 'k-', linewidth=1, label='Точное', alpha=0.7)

# Однородное численное
u_num_x0_h = solutions_homog[:, x0_idx]
plt.plot(times_homog, u_num_x0_h, 'bo', markersize=3, label='Числ. однор.')

# Неоднородное численное (редуцируем для визуализации)
step = max(1, len(times_nhomog)//100)
u_num_x0_nh = solutions_nhomog[::step, x0_idx]
t_nh_vis = times_nhomog[::step]
plt.plot(t_nh_vis, u_num_x0_nh, 'r.', markersize=4, label='Числ. неоднор.')

plt.axhline(0, color='gray', linestyle=':', alpha=0.5)
plt.axvline(T, color='green', linestyle='--', alpha=0.7, label=f'T={T:.3f}')
plt.xlabel('t')
plt.ylabel('u(0.5, t)')
plt.title('Движение средней точки')
plt.legend(loc='upper right')
plt.grid(True)

# 5. Распределение параметров
plt.subplot(2, 3, 5)
plt.plot(x, rho_vals, 'b-', linewidth=2, label=f'ρ(x)={rho0}+{rho1}/(1+x)')
plt.axhline(P0, color='b', linestyle='--', label=f'P0={P0:.3f}')
plt.xlabel('x')
plt.ylabel('Плотность ρ(x)')
plt.title('Распределение плотности')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 6)
plt.plot(x, k_vals, 'r-', linewidth=2, label=f'k(x)={k0}+{k1}/(1+x)')
plt.axhline(s0, color='r', linestyle='--', label=f's0={s0:.3f}')
plt.xlabel('x')
plt.ylabel('Жёсткость k(x)')
plt.title('Распределение жёсткости')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ============================================================================
# 5. ЭНЕРГЕТИЧЕСКИЙ АНАЛИЗ
# ============================================================================
def compute_energy(solutions, times, tau, rho_vals, k_vals):
    """Вычисление кинетической и потенциальной энергии"""
    Nt = len(times)
    Nx = len(x) - 1
    h = L / Nx
    
    K = np.zeros(Nt)
    P = np.zeros(Nt)
    
    for idx in range(Nt):
        u = solutions[idx, :]
        
        # Скорость (кроме первого шага)
        if idx == 0:
            v = v0(x)
        else:
            v = (solutions[idx, :] - solutions[idx-1, :]) / tau
        
        # Кинетическая энергия
        K[idx] = 0.5 * np.sum(rho_vals * v**2) * h
        
        # Потенциальная энергия (du/dx)
        u_x = np.zeros(len(x))
        # Внутренние точки
        for i in range(1, len(x)-1):
            u_x[i] = (u[i+1] - u[i-1]) / (2*h)
        # Границы
        u_x[0] = (u[1] - u[0]) / h
        u_x[-1] = (u[-1] - u[-2]) / h
        
        P[idx] = 0.5 * np.sum(k_vals * u_x**2) * h
    
    return K, P

# Энергии для однородного
rho_homog = np.full_like(x, P0)
k_homog = np.full_like(x, s0)
K_h, P_h = compute_energy(solutions_homog, times_homog, tau_h, rho_homog, k_homog)

# Энергии для неоднородного
K_nh, P_nh = compute_energy(solutions_nhomog, times_nhomog, tau_nh, rho_vals, k_vals)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(times_homog, K_h, 'r-', label='Кинетическая', alpha=0.7)
plt.plot(times_homog, P_h, 'b-', label='Потенциальная', alpha=0.7)
plt.plot(times_homog, K_h+P_h, 'k-', label='Полная', linewidth=2)
plt.xlabel('t')
plt.ylabel('Энергия')
plt.title('Энергии ОДНОРОДНОГО стержня')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(times_nhomog, K_nh, 'r-', label='Кинетическая', alpha=0.7)
plt.plot(times_nhomog, P_nh, 'b-', label='Потенциальная', alpha=0.7)
plt.plot(times_nhomog, K_nh+P_nh, 'k-', label='Полная', linewidth=2)
plt.xlabel('t')
plt.ylabel('Энергия')
plt.title('Энергии НЕОДНОРОДНОГО стержня')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ============================================================================
# ВЫВОД РЕЗУЛЬТАТОВ
# ============================================================================
print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ АНАЛИЗА:")
print("="*60)
print(f"1. Параметры: ρ(x)={rho0}+{rho1}/(1+x), k(x)={k0}+{k1}/(1+x)")
print(f"2. Средние: P0={P0:.4f}, s0={s0:.4f}, a={a:.4f}")
print(f"3. Время T (третий проход x=0.5 через 0): {T:.4f}")
print(f"\n4. ОДНОРОДНЫЙ стержень:")
print(f"   - Численное решение в T: u(0.5,T) = {u_num_h[x0_idx]:.6e}")
print(f"   - Точное решение в T:     u(0.5,T) = {u_exact[x0_idx]:.6e}")
print(f"   - Погрешность в точке x=0.5: {abs(u_num_h[x0_idx] - u_exact[x0_idx]):.2e}")
print(f"   - Макс. погрешность по x: {error.max():.2e}")
print(f"\n5. НЕОДНОРОДНЫЙ стержень:")
print(f"   - Численное решение в T: u(0.5,T) = {u_num_nh[x0_idx]:.6e}")
print(f"   - Отличие от однородного: {abs(u_num_nh[x0_idx] - u_exact[x0_idx]):.2e}")
print(f"\n6. СОХРАНЕНИЕ ЭНЕРГИИ:")
print(f"   Однородный: начальная={K_h[0]+P_h[0]:.6f}, конечная={K_h[-1]+P_h[-1]:.6f}")
print(f"   Изменение: {abs(K_h[-1]+P_h[-1]-K_h[0]-P_h[0])/(K_h[0]+P_h[0]):.2e}")
print(f"   Неоднородный: начальная={K_nh[0]+P_nh[0]:.6f}, конечная={K_nh[-1]+P_nh[-1]:.6f}")
print(f"   Изменение: {abs(K_nh[-1]+P_nh[-1]-K_nh[0]-P_nh[0])/(K_nh[0]+P_nh[0]):.2e}")