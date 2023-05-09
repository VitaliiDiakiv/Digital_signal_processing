import numpy as np
import time
import matplotlib.pyplot as plt

#Функція для обчислення ШПФ для вхідного сигналу x
def fft(x):
    n = len(x)
    if n == 1:
        return x
    else:
        # Доповнюємо вхідний сигнал нулями до довжини, що є степенем двійки
        m = int(n/2)
        even = fft(x[::2])
        odd = fft(x[1::2])
        W = np.exp(-2j * np.pi * np.arange(m) / n)
        return np.concatenate([even + W * odd, even - W * odd])

# Генеруємо випадковий сигнал довжиною 32
N = 32
x = np.random.rand(N)

# Доповнюємо вхідний сигнал нулями до степеня 2
M = 2**int(np.ceil(np.log2(N)))
x = np.concatenate([x, np.zeros(M-N)])


t1 = time.time()
# Обчислюємо ШПФ
X = fft(x)
t2 = time.time()

# Виводимо результат
for i, val in enumerate(X):
    print(f"C_{i}: {val}")

# вивід часу обчислення
print(f"\nЧас виконання: {t2 - t1:} секунд")

# обрахунок кількості операцій
num_plus=N
num_mult=4*N
num_operations =num_plus+num_mult

print(f"\nКількість операцій множення: {num_mult}")
print(f"Кількість операцій додавання: {num_plus}")
print(f"Кількість операцій: {num_operations}")

# обчислення спектру амплітуд і фаз для вхідного вектору
amp = np.abs(X)
phase = np.angle(X)

# побудова графіку спектру амплітуд і фаз
plt.figure()
plt.stem(amp)
plt.title("Амплітудний спектр")
plt.xlabel("Частота")
plt.ylabel("Амплітуда")
plt.figure()
plt.stem(phase)
plt.title("Фазовий спектр")
plt.xlabel("Частота")
plt.ylabel("Фаза")
plt.show()