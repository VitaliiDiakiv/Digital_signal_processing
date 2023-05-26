import numpy as np
import matplotlib.pyplot as plt

# Параметри
A = 1.0  # Амплітуда синусоїди
n = 7
phi = np.pi/4  # Зсув по фазі
N = n * 100  # Кількість значень у послідовності

# Функція для генерації тестової послідовності з випадковими спотвореннями
def generate_test_sequence():
    x = np.linspace(0, 3, N)  # Інтервал x
    y_exact = A * np.sin(n*x + phi)  # Точне значення без спотворення
    deviation = np.random.uniform(-0.05*A, 0.05*A, N)  # Випадкове спотворення де похибка не перевищує 5%
    y = y_exact + deviation
    return x, y, y_exact

# Функції для обчислення середніх значень
def arithmetic_mean(sequence):
    return np.mean(sequence)

def harmonic_mean(sequence):
    return len(sequence) / np.sum(1/sequence)

def geometric_mean(sequence):
    sequence = np.where(sequence <= 0, np.nan, sequence)  # Заміна недопустимих значень на NaN
    return np.nanprod(sequence) ** (1/np.sum(~np.isnan(sequence)))

# Функція для виводу результату на екран у вигляді графіку функції
def plot_sequence(x, y, y_exact):
    plt.plot(x, y,color='red', label='Спотворена послідовність')
    plt.plot(x, y_exact, label='Точна послідовність')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper right')
    plt.show()

# Функція для виводу результату на екран у вигляді графіку функції
def plot_sequences(x, y, y_exact):
    plt.subplot(2, 1, 2)
    plt.plot(x, y, color='red', label='Спотворена послідовність')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper right')
    plt.subplot(2, 1, 1)
    plt.plot(x, y_exact, label='Точна послідовність')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# Функція для обчислення точного значення
def exact_value(x, A, n, phi):
    return A * np.sin(n*x + phi)

# Функція для порівняння наближеного значення з точним
def compare_values(approx_value, exact_value):
    return np.abs(approx_value - exact_value)

# Генерація тестової послідовності
x, y, y_exact = generate_test_sequence()

# Обчислення середніх значень
arithmetic_mean_value = arithmetic_mean(y)
harmonic_mean_value = harmonic_mean(y)
geometric_mean_value = geometric_mean(y)

# Обчислення точного значення та порівняння наближеного значення з точним
exact_values = exact_value(x, A, n, phi)
comparison = compare_values(y, exact_values)

# Обчислення максимумів і мінімумів абсолютних і відносних похибок
max_absolute_error = np.max(comparison)
min_absolute_error = np.min(comparison)
max_relative_error = np.max(comparison / np.abs(exact_values))
min_relative_error = np.min(comparison / np.abs(exact_values))

# Виведення результату
print("Середні значення:")
print("Арифметичне середнє: ", arithmetic_mean_value)
print("Гармонійне середнє: ", harmonic_mean_value)
print("Геометричне середнє: ", geometric_mean_value)

print("\nПохибки:")
print("Максимальна абсолютна похибка: ", max_absolute_error)
print("Мінімальна абсолютна похибка: ", min_absolute_error)
print("Максимальна відносна похибка: ", max_relative_error)
print("Мінімальна відносна похибка: ", min_relative_error)

# Виведення графіків
plot_sequences(x, y, y_exact)

# Виведення графіку
plot_sequence(x, y, y_exact)

