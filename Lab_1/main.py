import math
from scipy.integrate import quad
import matplotlib.pyplot as plt
import numpy as np
import csv

#"------------Пункт 1------------"
def plot_function():
    fig, ax = plt.subplots()

    # Задання обмеження для x
    x = np.linspace(-np.pi, 2*np.pi, 1000)
    y= 7 * np.sin(7*x*math.pi)

    # Створеня графіка заданої функції
    plt.plot(x,y)
    ax.set_xlabel('Вісь x')
    ax.set_ylabel('Вісь y')
    ax.set_title('Графік функції 7*sin(7*x*pi) в межах (-pi ,2*pi)')
    ax.grid()
    plt.show()

#"------------Пункт 2------------"
# Обчислення  коефіцієнтів ряду Фур'є
def compute_a_k_and_b_k(N):
    print("Обчислення значень a_k та b_k :")
    l = 3 * math.pi

    def integrand(x):
        return 7 * np.sin(7 * x * math.pi)

    result, _ = quad(integrand, -2 / l, 2 / l)
    a_k = (2 / l) * result
    print("a_k:", a_k)

    def integrand(x, k):
        return 7 * np.sin(7 * x * math.pi) * np.sin(k * x * math.pi)

    b_k = np.zeros(N)
    for k in range(1, N+1):
        result, _ = quad(integrand, -2 / l, 2 / l, args=(k,))
        b_k[k - 1] = (2 / l) * result
        print("b_{}: {}".format(k, b_k[k - 1]))

    return a_k, b_k


#"------------Пункт 3------------"
# Обчислення  наближення функції f(x) рядом Фур'є з точністю до порядку N
def fourier_series(f, l, N):
    def b(k):
        integrand = lambda x: f(x) * np.sin(2 * k * np.pi * x / l)
        result, _ = quad(integrand, -l / 2, l / 2)
        return (2 / l) * result

    def fourier_sum(x):
        sum = 0
        for k in range(1, N + 1):
            b_k = b(k)
            sum += b_k * np.sin(2 * k * np.pi * x / l)
        return sum

    return fourier_sum


def f(x):
    return 7 * np.sin(7 * x * np.pi)

# Перевірка, що наближення працює коректно
def check_approximation_plot(N):
    def b(k):
        integrand = lambda x: f(x) * np.sin(k * x)
        result, _ = quad(integrand, -np.pi, np.pi)
        return result / np.pi

    fig, ax = plt.subplots()
    ax.set_xlabel('Вісь x')
    ax.set_ylabel('Вісь y')
    ax.set_title('Графік апроксимації в межах (-pi ,2*pi)')

    for n in range(1, N + 1):
        approximation = fourier_series(f, 3*np.pi, n)
        x = np.linspace(-np.pi, 2 * np.pi, 1000)
        y = approximation(x)
        plt.plot(x, y, label=f"N={n}")

    plt.legend(loc='upper center', ncol=3, fontsize='small')
    ax.grid()
    plt.show()
#"------------Пункт 4------------"
def plot_fourier_coefficients(N):
    fig, axs = plt.subplots(2, figsize=(8, 6))

    l = 3*math.pi

    def integrand(x):
        return 7 * np.sin(7 * x * math.pi)

    result, _ = quad(integrand, -2 / l, 2 / l)
    a_0 = (1 / l) * result

    a = [0] * N
    b = [0] * N
    for k in range(1, N+1):
        def integrand(x):
            return 7 * np.sin(7 * x * math.pi) * np.cos(k * x * math.pi)
        result, _ = quad(integrand, -2 / l, 2 / l)
        a_k = (2 / l) * result
        a[k - 1] = a_k

        def integrand(x):
            return 7 * np.sin(7 * x * math.pi) * np.sin(k * x * math.pi)
        result, _ = quad(integrand, -2 / l, 2 / l)
        b_k = (2 / l) * result
        b[k - 1] = b_k
    axs[0].stem([0] + a)
    axs[0].set_xlabel('k')
    axs[0].set_ylabel('a(k)')
    axs[0].set_title('Графік гармонік для (a(k))')
    axs[1].stem(range(1, N+1), b)
    axs[1].set_xlabel('k')
    axs[1].set_ylabel('b(k)')
    axs[1].set_title('Графік гармонік для (b(k))')
    axs[1].set_xticks(range(1, N+1))
    axs[1].set_xticklabels([f'{i}' for i in range(1, N+1)])
    plt.tight_layout()
    plt.grid()
    plt.show()

#"------------Пункт 5------------"

def compute_approximation_errors(f, l, N):
    # Обчислення коефіцієнтів ряду Фур'є
    def b(k):
        return (2 / l) * np.trapz(f(x) * np.sin(2 * k * np.pi * x / l), x)

    x = np.linspace(-np.pi, 2 * np.pi, N)

    # Обчислення наближення функції f(x) рядом Фур'є з точністю до порядку N
    def fourier_sum(x):
        sum = 0
        for k in range(1, N + 1):
            b_k = b(k)
            sum += b_k * np.sin(2 * k * np.pi * x / l)
        return sum

# Обчислення відносної похибки в кожній точці
    approximations = [fourier_sum(x_i) for x_i in x]
    exact_values = f(x)
    eps = 1e-10
    errors = np.abs(approximations - exact_values) / (np.abs(exact_values) + eps)

    # Побудова графіка відносних похибок
    fig, ax = plt.subplots()
    ax.plot(x, errors)
    ax.set_xlabel('x')
    ax.set_ylabel('Відносна похибка')
    ax.set_title(f'Відносна похибка для N={N}')
    ax.grid()
    return errors

#"------------Пункт 6------------"
def save_to_file(N, filename):
    def f(x):
        return 7 * np.sin(7 * x * np.pi)

    l = 3 * np.pi
    # Обчислення значень a_k та b_k
    a_k, b_k = compute_a_k_and_b_k(N)

    # Обчислення відносної похибки
    errors = compute_approximation_errors(f, l, N)

    # Збереження у файл
    with open(filename, mode='w') as file:
        writer = csv.writer(file)
        file.write("N:" + str(N))
        file.write("\nОбчислені коефіцієнти a_k:" + str(a_k))
        file.write("\nОбчислені коефіцієнти b_k:" + str(b_k))
        file.write("\nОбчислена відносна похибка:" + str(errors))

N =20
#"------------Пункт 1------------"
plot_function()
print("------------Пункт 2-----------")
save_to_file(N, 'lab1')
#"------------Пункт 3------------"
check_approximation_plot(N)
#"------------Пункт 4------------"
plot_fourier_coefficients(N)
#"------------Пункт 5------------"
def f(x):
    return 7 * np.sin(7 * x * np.pi)
l = 3 * np.pi
errors = compute_approximation_errors(f, l, N)
print(f"Відносні похибки отриманого наближення для N = {N}: {errors}")