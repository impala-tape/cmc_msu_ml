from typing import Any
from math import pi, sin, e, cos
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
def run_through(n: int, 
                A: [float, ..., float],  
                B: [float, ..., float], 
                C: [float, ..., float],
                F: [float, ..., float], 
                x1: float, 
                x2: float,
                m1: float,
                m2: float) -> [float, ..., float]:
    a = [x1]
    b = [m1]
    y = []
    for i in range(n - 1):
        a.append(B[i] / (C[i] - A[i] * a[i]))
        b.append((F[i] + A[i] * b[i]) / (C[i] - A[i] * a[i]))
        
    y.append((m2 + x2 * b[n - 1]) / (1 - x2 * a[n - 1]))
    
    for i in reversed(range(n)):
        y.append(a[i] * y[n - 1 - i] + b[i])
    y.reverse()
    return y
def F(x = 0.0, t = 0.0):
    return 0

def fi(x = 0.0):
    return cos(pi * x)

def m1(x = 0.0):
    return cos(pi * x)

def m2(x): 
    return -cos(pi * x)

def psi(x: float): 
    return 0
  
def analit_reshen(x: float, t: float) -> float: 
    return cos(pi * x) * cos(pi * t)
def oscillation(n: int, 
                last: list[float],
                pres: list[float],
                F: Any,
                m1: Any,
                m2: Any,
                sig: float,
                tay: float,
                h: float,
                t: int,
                per: int):
    y = []
#     for el in last:
#         print(el, end=' ') 
    f = []
    a = [sig * (tay ** 2) / (h ** 2)]
    a[1:] = [a[0] for i in range(n - 1)]
    b = [a[0] for i in range(n)]
    c = [(2 * a[0] + 1) for i in range(n)]
    for i in range(0, t):
        for j in range(n - 1):
            f.append(2 * pres[j + 1] - last[j + 1] + tay * tay * F(h * (j + 1), (i - 1) * tay) + sig / (h * h) * tay * tay * (last[j] - 2 * last[j + 1] + last[j + 2]) + (1 - 2 * sig) / (h * h) * tay * tay * (pres[j] - 2 * pres[j + 1] + pres[j + 2]))       
        
        y.append(pres)
        last = pres
        pres = run_through(n, a, b, c, f, 0.0, 0.0, m1(i * tay), m2(i * tay))
        f = []

    return(y)
n = 100
t = 3
per = 100
sig = 0
h = 0.01
tay = 0.1
my_max = 0

last = []
pres = [0 for i in range(n + 1)]  # !!!!!!!!!!!

for i in range(n + 1):
    last.append(fi(i * h))
    
pres[0] = m1(tay)
pres[n] = m2(tay)
for i in range(1, n):
    pres[i] = tay * (psi(i * h) + 0.5 * tay / (h ** 2) * (last[i - 1] - 2 * last[i] \
                                                           + last[i + 1]) + F(i * h, 0)) + last[i]

x = list(map(lambda i: i/100, list(range(0, 101))))
y = oscillation(n, last, pres, F, m1, m2, sig, tay, h, t*100, per)
# y = list(map(lambda i: i/100, list(range(0, 301))))

#print(x)

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(xlim=(0, 1), ylim=(-1, 1))
line, = ax.plot([], [], lw=3)

def init():
    line.set_data([], [])
    return line,

def animate(i):
    #x = [0.01 * a for a in range(0, 101)]
    #z = list(map(lambda k: analit_reshen(k, i/1000), x))
    z = y[i]
    line.set_data(x, z)
    return line,

anim = FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=100, blit=True)

plt.title('sigma = 0, шаг по времени = 0.1')
plt.show()


n = 100
t = 3
per = 100
sig = 0.15
h = 0.01
tay = 0.03
my_max = 0

last = []
pres = [0 for i in range(n + 1)]  # !!!!!!!!!!!

for i in range(n + 1):
    last.append(fi(i * h))
    
pres[0] = m1(tay)
pres[n] = m2(tay)
for i in range(1, n):
    pres[i] = tay * (psi(i * h) + 0.5 * tay / (h ** 2) * (last[i - 1] - 2 * last[i] \
                                                           + last[i + 1]) + F(i * h, 0)) + last[i]

x = list(map(lambda i: i/100, list(range(0, 101))))
y = oscillation(n, last, pres, F, m1, m2, sig, tay, h, t*100, per)
# y = list(map(lambda i: i/100, list(range(0, 301))))

#print(x)

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(xlim=(0, 1), ylim=(-1, 1))
line, = ax.plot([], [], lw=3)

def init():
    line.set_data([], [])
    return line,

def animate(i):
    #x = [0.01 * a for a in range(0, 101)]
    #z = list(map(lambda k: analit_reshen(k, i/1000), x))
    z = y[i]
    line.set_data(x, z)
    return line,

anim = FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=10, blit=True)

plt.title('sigma = 0.15, шаг по времени = 0.03')
plt.show()
n = 100
t = 50
per = 100
sig = 0
h = 0.01
tay = 0.001
my_max = 0

last = []
pres = [0 for i in range(n + 1)]  # !!!!!!!!!!!

for i in range(n + 1):
    last.append(fi(i * h))
    
pres[0] = m1(tay)
pres[n] = m2(tay)
for i in range(1, n):
    pres[i] = tay * (psi(i * h) + 0.5 * tay / (h ** 2) * (last[i - 1] - 2 * last[i] \
                                                           + last[i + 1]) + F(i * h, 0)) + last[i]

x = list(map(lambda i: i/100, list(range(0, 101))))
y = oscillation(n, last, pres, F, m1, m2, sig, tay, h, t*100, per)
# y = list(map(lambda i: i/100, list(range(0, 301))))

#print(x)

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(xlim=(0, 1), ylim=(-1, 1))
line, = ax.plot([], [], lw=3)

def init():
    line.set_data([], [])
    return line,

def animate(i):
    #x = [0.01 * a for a in range(0, 101)]
    #z = list(map(lambda k: analit_reshen(k, i/1000), x))
    z = y[i]
    line.set_data(x, z)
    return line,

anim = FuncAnimation(fig, animate, init_func=init,
                               frames=2000, interval=2, blit=True)

plt.title('sigma = 0, шаг по времени = 0.001')
plt.show()
n = 100
t = 3
per = 100
sig = 0.2
h = 0.01
tay = 0.01
my_max = 0

last = []
pres = [0 for i in range(n + 1)]  # !!!!!!!!!!!

for i in range(n + 1):
    last.append(fi(i * h))
    
pres[0] = m1(tay)
pres[n] = m2(tay)
for i in range(1, n):
    pres[i] = tay * (psi(i * h) + 0.5 * tay / (h ** 2) * (last[i - 1] - 2 * last[i] \
                                                           + last[i + 1]) + F(i * h, 0)) + last[i]

x = list(map(lambda i: i/100, list(range(0, 101))))
y = oscillation(n, last, pres, F, m1, m2, sig, tay, h, t*100, per)
# y = list(map(lambda i: i/100, list(range(0, 301))))

#print(x)

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(xlim=(0, 1), ylim=(-1, 1))
line, = ax.plot([], [], lw=3)

def init():
    line.set_data([], [])
    return line,

def animate(i):
    #x = [0.01 * a for a in range(0, 101)]
    #z = list(map(lambda k: analit_reshen(k, i/1000), x))
    z = y[i]
    line.set_data(x, z)
    return line,

anim = FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=10, blit=True)

plt.title('sigma = 0.2, шаг по времени = 0.01')
plt.show()
n = 100
t = 3
per = 100
sig = 0.75
h = 0.01
tay = 0.01
my_max = 0

last = []
pres = [0 for i in range(n + 1)]  # !!!!!!!!!!!

for i in range(n + 1):
    last.append(fi(i * h))
    
pres[0] = m1(tay)
pres[n] = m2(tay)
for i in range(1, n):
    pres[i] = tay * (psi(i * h) + 0.5 * tay / (h ** 2) * (last[i - 1] - 2 * last[i] \
                                                           + last[i + 1]) + F(i * h, 0)) + last[i]

x = list(map(lambda i: i/100, list(range(0, 101))))
y = oscillation(n, last, pres, F, m1, m2, sig, tay, h, t*100, per)
# y = list(map(lambda i: i/100, list(range(0, 301))))

y_real = []
for i in range(300):
    y_real.append(list(map(lambda k: analit_reshen(k, i/100), x)))
print(len(y[0]))

delta = 0
for i in range(101):
    if (abs(y_real[0][i] - y[0][i]) > delta):
        delta = abs(y_real[0][i] - y[0][i])
print("Погрешность: ", delta)

delta = 0
for i in range(101):
    if (abs(y_real[100][i] - y[100][i]) > delta):
        delta = abs(y_real[100][i] - y[100][i])
print("Погрешность: ", delta)

for i in range(101):
    if (abs(y_real[299][i] - y[299][i]) > delta):
        delta = abs(y_real[299][i] - y[299][i])
print("Погрешность: ", delta)

print(y[0])
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(xlim=(0, 1), ylim=(-1, 1))
line, = ax.plot([], [], lw=3)

def init():
    line.set_data([], [])
    return line,

def animate(i):
    x = [0.01 * a for a in range(0, 101)]
    z = list(map(lambda k: analit_reshen(k, i/1000), x))
    z = y[i]
    line.set_data(x, z)
    return line,

anim = FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=10, blit=True)

plt.title('sigma = 0.75, шаг по времени = 0.01')
plt.show()
