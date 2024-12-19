import numpy as np
import matplotlib.pyplot as plt

# 1. Visualizing consistent and inconsistent systems
def plot_linear_system():
    x = np.linspace(-10, 10, 200)

    # Consistent system (One solution)
    y1 = 2 * x + 3  # Equation 1: y = 2x + 3
    y2 = -x + 5     # Equation 2: y = -x + 5

    # Inconsistent system (No solution)
    y3 = 0.5 * x + 1  # Equation 3: y = 0.5x + 1
    y4 = 0.5 * x + 4  # Equation 4: y = 0.5x + 4 (parallel to Equation 3)

    # Infinite solutions (Same line)
    y5 = -x + 2      # Equation 5: y = -x + 2
    y6 = -x + 2      # Equation 6: y = -x + 2 (overlaps with Equation 5)

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Plot consistent system (One solution)
    ax[0].plot(x, y1, label='y = 2x + 3')
    ax[0].plot(x, y2, label='y = -x + 5')
    ax[0].legend()
    ax[0].grid(True)
    ax[0].set_title("Consistent System (One Solution)")
    ax[0].set_xlim(-10, 10)
    ax[0].set_ylim(-10, 10)

    # Plot inconsistent system (No solution)
    ax[1].plot(x, y3, label='y = 0.5x + 1')
    ax[1].plot(x, y4, label='y = 0.5x + 4')
    ax[1].legend()
    ax[1].grid(True)
    ax[1].set_title("Inconsistent System (No Solution)")
    ax[1].set_xlim(-10, 10)
    ax[1].set_ylim(-10, 10)

    # Plot infinite solutions
    ax[2].plot(x, y5, label='y = -x + 2')
    ax[2].plot(x, y6, '--', label='y = -x + 2 (Same Line)')
    ax[2].legend()
    ax[2].grid(True)
    ax[2].set_title("Consistent System (Infinite Solutions)")
    ax[2].set_xlim(-10, 10)
    ax[2].set_ylim(-10, 10)

    plt.show()

# 2. Matrix notation and solution visualization
def matrix_notation_and_solutions():
    # Example systems
    print("\n=== One Solution ===")
    A = np.array([[2, 1], [1, -1]])
    B = np.array([5, 1])
    print("Coefficient Matrix A:\n", A)
    print("Constant Matrix B:\n", B)
    solution = np.linalg.solve(A, B)
    print("Solution:\n", solution)

    print("\n=== No Solution (Inconsistent) ===")
    A = np.array([[1, -2], [2, -4]])
    B = np.array([3, 8])
    print("Coefficient Matrix A:\n", A)
    print("Constant Matrix B:\n", B)
    try:
        solution = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        print("No solution (rows are parallel or inconsistent).")

    print("\n=== Infinite Solutions ===")
    A = np.array([[1, -1], [2, -2]])
    B = np.array([2, 4])
    print("Coefficient Matrix A:\n", A)
    print("Constant Matrix B:\n", B)
    print("Infinite solutions (lines overlap). Use parameterized representation.")

# 3. Row elementary operations visualization
def row_operations_visualization():
    # Initial matrix
    matrix = np.array([[2, 1, 5], [1, -1, 1]])
    print("Initial Augmented Matrix:")
    print(matrix)

    # Applying row operations
    matrix[1] = matrix[1] - 0.5 * matrix[0]  # R2 -> R2 - 0.5*R1
    print("\nAfter R2 -> R2 - 0.5*R1:")
    print(matrix)

    matrix[0] = matrix[0] / 2  # R1 -> R1 / 2
    print("\nAfter R1 -> R1 / 2:")
    print(matrix)

    matrix[1] = matrix[1] / -1.5  # R2 -> R2 / -1.5
    print("\nAfter R2 -> R2 / -1.5:")
    print(matrix)

# Run the functions
plot_linear_system()
matrix_notation_and_solutions()
row_operations_visualization()


