import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, Normalizer

# Image 1 - Asymptotic Behavior
x = np.linspace(-5, 5, 100)
f1 = -1 / (x - 2)
f2 = np.full_like(x, 4)
f3 = 1 / (x + 2)

plt.figure(figsize=(8, 6))
plt.plot(x, f1, label=r'$\lim_{x\to 2^-} f(x) = -\infty$')
plt.plot(x, f2, label=r'$\lim_{x\to 2^+} f(x) = 4$')
plt.plot(x, f3, label=r'$\lim_{x\to -2} f(x) = \infty$')
plt.axvline(x=2, color='k', linestyle='--', label='x = 2')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Asymptotic Behavior')
plt.legend()
plt.grid()
plt.show()

# Image 2 - Horizontal and Vertical Asymptotes
x = np.linspace(-5, 5, 100)
f = (np.sqrt(2*x**2 + 1)) / (3*x - 5)

plt.figure(figsize=(8, 6))
plt.plot(x, f)
plt.axhline(y=2/3, color='r', linestyle='--', label='Horizontal Asymptote')
plt.axvline(x=5/3, color='g', linestyle='--', label='Vertical Asymptote')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Horizontal and Vertical Asymptotes')
plt.legend()
plt.grid()
plt.show()

# Image 3 - Exponential Function and Asymptotes
x = np.linspace(-5, 5, 100)
y = np.exp(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.axhline(y=0, color='r', linestyle='--', label='Horizontal Asymptote')
plt.axhline(y=1, color='g', linestyle='--', label='Horizontal Asymptote')
plt.xlabel('x')
plt.ylabel('y = e^x')
plt.title('Exponential Function and Asymptotes')
plt.legend()
plt.grid()
plt.show()

# Generate synthetic dataset
np.random.seed(42)
data = np.random.randint(1, 100, size=(100, 1))

# Normalization Methods
scalers = {
    "Min-Max Scaling": MinMaxScaler(),
    "Z-Score Normalization": StandardScaler(),
    "Max-Abs Scaling": MaxAbsScaler(),
    "Robust Scaling": RobustScaler(),
    "L2 Normalization": Normalizer(norm='l2')
}

# Apply each normalization and store results
normalized_data = {}
for method, scaler in scalers.items():
    normalized_data[method] = scaler.fit_transform(data)

# Plot original and normalized data
plt.figure(figsize=(12, 8))

# Original Data Plot
plt.subplot(3, 2, 1)
plt.scatter(range(len(data)), data, color='blue', label='Original Data')
plt.title("Original Data")
plt.xlabel("Index")
plt.ylabel("Value")
plt.grid(True)
plt.legend()

# Normalized Data Plots
for i, (method, values) in enumerate(normalized_data.items(), start=2):
    plt.subplot(3, 2, i)
    plt.scatter(range(len(values)), values, label=method, alpha=0.8)
    plt.title(method)
    plt.xlabel("Index")
    plt.ylabel("Normalized Value")
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()