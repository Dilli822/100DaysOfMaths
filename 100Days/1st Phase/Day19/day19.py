import numpy as np
import matplotlib.pyplot as plt

# Generate x values for the plots
x = np.linspace(-10, 10, 500)
x_positive = np.linspace(0.1, 10, 500)  # Avoid log(0) and divide by zero

# Composite functions
y_add = x + 2
y_sub = x - 2
y_mul = x * 2
y_div = x / (x + 2)  # Rational function
y_log = np.log(x_positive)
y_cubic = x**3

# Create the plot
plt.figure(figsize=(14, 10))

# Plot each function
plt.subplot(2, 3, 1)
plt.plot(x, y_add, label="y = x + 2", color="blue")
plt.title("Addition Function")
plt.grid()
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(x, y_sub, label="y = x - 2", color="green")
plt.title("Subtraction Function")
plt.grid()
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(x, y_mul, label="y = x * 2", color="red")
plt.title("Multiplication Function")
plt.grid()
plt.legend()

plt.subplot(2, 3, 4)
plt.plot(x, y_div, label="y = x / (x + 2)", color="purple")
plt.title("Rational Function")
plt.grid()
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(x_positive, y_log, label="y = log(x)", color="orange")
plt.title("Logarithmic Function")
plt.grid()
plt.legend()

plt.subplot(2, 3, 6)
plt.plot(x, y_cubic, label="y = x^3", color="cyan")
plt.title("Cubic Function")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
