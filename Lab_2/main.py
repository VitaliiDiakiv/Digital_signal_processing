import numpy as np
import matplotlib.pyplot as plt
import time

# функція для обчислення коефіцієнтів A_k, B_k, C_k
def dft_coeff(k, x):
    N = len(x)
    n = np.arange(N)
    A_k = np.sum(x * np.cos(2 * np.pi * k * n / N)) / N
    B_k = np.sum(x * np.sin(2 * np.pi * k * n / N)) / (-N)
    C_k = A_k + 1j * B_k
    return C_k

# функція для обчислення спектру амплітуд і фаз
def dft_amp_phase(x):
    N = len(x)
    amp = np.zeros(N)
    phase = np.zeros(N)
    for k in range(N):
        coeff = dft_coeff(k, x)
        A_k = coeff.real
        B_k = coeff.imag
        amp[k] = np.sqrt(A_k**2 + B_k**2)
        phase[k] = np.arctan2(B_k, A_k)
    return amp, phase


# створення випадкового вхідного вектору довжиною N=17
N = 100
x = np.random.rand(N)

# обчислення коефіцієнтів ряду Фур’є та спектру амплітуд і фаз для вхідного вектору
t1 = time.time()
coeff = []
for k in range(N):
    C_k = dft_coeff(k, x)
    coeff.append(dft_coeff(k, x))
    print(f"C_{k}: {C_k:.8f}")
coeff = np.array(coeff)
amp, phase = dft_amp_phase(x)
t2 = time.time()


# вивід часу обчислення
print(f"\nЧас виконання: {t2 - t1:.8f} секунд")

# обрахунок кількості операцій
num_operations = N*((5*N+5*N+2)+1)

# вивід кількості операцій
print(f"Кількість операцій: {num_operations}")


# побудова графіку спектру амплітуд і фаз
fig, axs = plt.subplots(2, 1, figsize=(8, 8))
amp = amp[1:]
axs[0].stem(amp)
axs[0].set_title('Спектр амплітуд')
axs[0].set_xlabel('Частота (k)')
axs[0].set_ylabel('Амплітуда')
axs[1].stem(phase)
axs[1].set_title('Cпектр фаз')
axs[1].set_xlabel('Частота (k)')
axs[1].set_ylabel('Фази (радіани)')
plt.tight_layout()
plt.show()
