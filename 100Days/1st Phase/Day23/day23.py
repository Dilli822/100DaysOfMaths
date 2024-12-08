import numpy as np
import matplotlib.pyplot as plt

# Define x values for each function's valid range
x_inv_trig = np.linspace(-0.99, 0.99, 500)  # For arcsin, arccos, arctan, etc.
x_sec_csc = np.linspace(1.01, 5, 500)      # For arcsec, arccsc (|x| > 1)
x_log = np.linspace(0.01, 10, 500)         # Positive x values for logarithmic functions

# Derivatives of inverse trigonometric functions
sin_inv_deriv = 1 / np.sqrt(1 - x_inv_trig**2)
cos_inv_deriv = -1 / np.sqrt(1 - x_inv_trig**2)
tan_inv_deriv = 1 / (1 + x_inv_trig**2)
cot_inv_deriv = -1 / (1 + x_inv_trig**2)
sec_inv_deriv = 1 / (np.abs(x_sec_csc) * np.sqrt(x_sec_csc**2 - 1))
csc_inv_deriv = -1 / (np.abs(x_sec_csc) * np.sqrt(x_sec_csc**2 - 1))

# Derivatives of logarithmic functions
log_deriv = 1 / x_log  # Derivative of ln(x)
log_base2_deriv = 1 / (x_log * np.log(2))  # Derivative of log2(x)
log_base10_deriv = 1 / (x_log * np.log(10))  # Derivative of log10(x)

# Create a figure with a clean layout
plt.figure(figsize=(14, 12))

# Plot derivatives of inverse circular functions
plt.subplot(2, 1, 1)
plt.plot(x_inv_trig, sin_inv_deriv, label=r"$\frac{d}{dx}(\sin^{-1}(x))$", color="red", linewidth=2)
plt.plot(x_inv_trig, cos_inv_deriv, label=r"$\frac{d}{dx}(\cos^{-1}(x))$", color="blue", linewidth=2)
plt.plot(x_inv_trig, tan_inv_deriv, label=r"$\frac{d}{dx}(\tan^{-1}(x))$", color="green", linewidth=2)
plt.plot(x_inv_trig, cot_inv_deriv, label=r"$\frac{d}{dx}(\cot^{-1}(x))$", color="purple", linewidth=2)
plt.plot(x_sec_csc, sec_inv_deriv, label=r"$\frac{d}{dx}(\sec^{-1}(x))$", color="orange", linewidth=2)
plt.plot(x_sec_csc, csc_inv_deriv, label=r"$\frac{d}{dx}(\csc^{-1}(x))$", color="brown", linewidth=2)
plt.title("Derivatives of Inverse Circular Functions", fontsize=16, fontweight="bold")
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.ylim(-10, 10)
plt.xlabel("x", fontsize=14)
plt.ylabel("Derivative", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

# Plot derivatives of logarithmic functions
plt.subplot(2, 1, 2)
plt.plot(x_log, log_deriv, label=r"$\frac{d}{dx}(\ln(x))$", color="red", linewidth=2)
plt.plot(x_log, log_base2_deriv, label=r"$\frac{d}{dx}(\log_2(x))$", color="blue", linewidth=2)
plt.plot(x_log, log_base10_deriv, label=r"$\frac{d}{dx}(\log_{10}(x))$", color="green", linewidth=2)
plt.title("Derivatives of Logarithmic Functions", fontsize=16, fontweight="bold")
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.ylim(0, 5)
plt.xlabel("x", fontsize=14)
plt.ylabel("Derivative", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()



from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# Define x values for each function's valid range
x_inv_trig = np.linspace(-0.99, 0.99, 500)  # For arcsin, arccos, arctan, etc.
x_sec_csc = np.linspace(1.01, 5, 500)      # For arcsec, arccsc (|x| > 1)
x_log = np.linspace(0.01, 10, 500)         # Positive x values for logarithmic functions

# Derivatives of inverse trigonometric functions
sin_inv_deriv = 1 / np.sqrt(1 - x_inv_trig**2)
cos_inv_deriv = -1 / np.sqrt(1 - x_inv_trig**2)
tan_inv_deriv = 1 / (1 + x_inv_trig**2)
cot_inv_deriv = -1 / (1 + x_inv_trig**2)
sec_inv_deriv = 1 / (np.abs(x_sec_csc) * np.sqrt(x_sec_csc**2 - 1))
csc_inv_deriv = -1 / (np.abs(x_sec_csc) * np.sqrt(x_sec_csc**2 - 1))

# Derivatives of logarithmic functions
log_deriv = 1 / x_log  # Derivative of ln(x)
log_base2_deriv = 1 / (x_log * np.log(2))  # Derivative of log2(x)
log_base10_deriv = 1 / (x_log * np.log(10))  # Derivative of log10(x)

# Create a 3D figure
fig = plt.figure(figsize=(14, 10))

# 3D plot for inverse trigonometric derivatives
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(x_inv_trig, sin_inv_deriv, zs=1, label=r"$\frac{d}{dx}(\sin^{-1}(x))$", color="red", linewidth=1.5)
ax1.plot(x_inv_trig, cos_inv_deriv, zs=2, label=r"$\frac{d}{dx}(\cos^{-1}(x))$", color="blue", linewidth=1.5)
ax1.plot(x_inv_trig, tan_inv_deriv, zs=3, label=r"$\frac{d}{dx}(\tan^{-1}(x))$", color="green", linewidth=1.5)
ax1.plot(x_inv_trig, cot_inv_deriv, zs=4, label=r"$\frac{d}{dx}(\cot^{-1}(x))$", color="purple", linewidth=1.5)
ax1.plot(x_sec_csc, sec_inv_deriv, zs=5, label=r"$\frac{d}{dx}(\sec^{-1}(x))$", color="orange", linewidth=1.5)
ax1.plot(x_sec_csc, csc_inv_deriv, zs=6, label=r"$\frac{d}{dx}(\csc^{-1}(x))$", color="brown", linewidth=1.5)
ax1.set_title("3D Visualization of Inverse Trigonometric Derivatives", fontsize=14, fontweight="bold")
ax1.set_xlabel("x", fontsize=12)
ax1.set_ylabel("Derivative Value", fontsize=12)
ax1.set_zlabel("Function Index", fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# 3D plot for logarithmic derivatives
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(x_log, log_deriv, zs=1, label=r"$\frac{d}{dx}(\ln(x))$", color="red", linewidth=1.5)
ax2.plot(x_log, log_base2_deriv, zs=2, label=r"$\frac{d}{dx}(\log_2(x))$", color="blue", linewidth=1.5)
ax2.plot(x_log, log_base10_deriv, zs=3, label=r"$\frac{d}{dx}(\log_{10}(x))$", color="green", linewidth=1.5)
ax2.set_title("3D Visualization of Logarithmic Derivatives", fontsize=14, fontweight="bold")
ax2.set_xlabel("x", fontsize=12)
ax2.set_ylabel("Derivative Value", fontsize=12)
ax2.set_zlabel("Function Index", fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
