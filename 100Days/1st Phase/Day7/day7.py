import numpy as np
import matplotlib.pyplot as plt

# Define x values (positive values since log(x) is undefined for x <= 0)
x = np.linspace(0.1, 10, 500)

# Define logarithmic functions for different bases
log_base2 = np.log2(x)  # log base 2
log_base10 = np.log10(x)  # log base 10
log_natural = np.log(x)  # natural log (base e)

# Plot logarithmic functions
plt.figure(figsize=(12, 6))
plt.plot(x, log_base2, label=r'$\log_2(x)$', color='blue')
plt.plot(x, log_base10, label=r'$\log_{10}(x)$', color='red')
plt.plot(x, log_natural, label=r'$\ln(x)$ (natural log)', color='green')

plt.title('Logarithmic Functions for Different Bases', fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('Logarithm Value', fontsize=14)
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.show()


# Demonstrate Product, Quotient, and Power Rules
# Define two functions to multiply, divide, and raise to power
y1 = x  # first function
y2 = np.sqrt(x)  # second function

# Product Rule: log_b(x * y) = log_b(x) + log_b(y)
product = np.log(y1 * y2)
sum_logs = np.log(y1) + np.log(y2)

# Quotient Rule: log_b(x / y) = log_b(x) - log_b(y)
quotient = np.log(y1 / y2)
diff_logs = np.log(y1) - np.log(y2)

# Power Rule: log_b(x^n) = n * log_b(x)
power = np.log(y1**2)
scaled_log = 2 * np.log(y1)

# Plot all these relationships
plt.figure(figsize=(12, 8))

# Product Rule
plt.subplot(3, 1, 1)
plt.plot(x, product, label=r'$\log(xy)$', color='blue')
plt.plot(x, sum_logs, label=r'$\log(x) + \log(y)$', color='orange', linestyle='--')
plt.title('Product Rule', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

# Quotient Rule
plt.subplot(3, 1, 2)
plt.plot(x, quotient, label=r'$\log(x/y)$', color='green')
plt.plot(x, diff_logs, label=r'$\log(x) - \log(y)$', color='red', linestyle='--')
plt.title('Quotient Rule', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

# Power Rule
plt.subplot(3, 1, 3)
plt.plot(x, power, label=r'$\log(x^2)$', color='purple')
plt.plot(x, scaled_log, label=r'$2\cdot\log(x)$', color='brown', linestyle='--')
plt.title('Power Rule', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Define x values for exponential functions (negative and positive values)
x_exp = np.linspace(-2, 2, 500)

# Define exponential functions
exp_growth = np.exp(x_exp)  # e^x (growth)
exp_decay = np.exp(-x_exp)  # e^(-x) (decay)

# Plot exponential growth and decay
plt.figure(figsize=(10, 6))
plt.plot(x_exp, exp_growth, label=r'$e^x$ (Exponential Growth)', color='blue')
plt.plot(x_exp, exp_decay, label=r'$e^{-x}$ (Exponential Decay)', color='red')

plt.title('Exponential Growth and Decay', fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('Value', fontsize=14)
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.show()

# Define a broader range of x values to demonstrate inverse relationships
x_inverse = np.linspace(0.1, 10, 500)
y_inverse = np.linspace(-2, 2, 500)

# Logarithmic and exponential inverse functions
log_exp = np.exp(np.log(x_inverse))  # log followed by exp
exp_log = np.log(np.exp(y_inverse))  # exp followed by log

# Plot inverse relationships
plt.figure(figsize=(10, 6))

# Plot log(exp(x)) and exp(log(x))
plt.plot(x_inverse, log_exp, label=r'$\exp(\log(x))$', color='blue')
plt.plot(y_inverse, exp_log, label=r'$\log(\exp(x))$', color='red')

# Add identity lines for reference
plt.plot(x_inverse, x_inverse, label='Identity Line (y=x)', color='green', linestyle='--')

plt.title('Inverse Relationship of Logarithmic and Exponential Functions', fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('Value', fontsize=14)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.show()
