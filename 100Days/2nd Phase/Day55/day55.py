import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -------------------------
# 1. Probability of Rolling Dice
# -------------------------
faces = [1, 2, 3, 4, 5, 6]
probabilities = [1/6] * 6

plt.figure(figsize=(10, 6))
plt.bar(faces, probabilities, color='skyblue', edgecolor='black')
plt.xlabel('Dice Faces')
plt.ylabel('Probability')
plt.title('Probability Distribution of Rolling a Fair Die')
plt.xticks(faces)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# -------------------------
# 2. Probability of Tossing Coins (2D)
# -------------------------
outcomes = ['HH', 'HT', 'TH', 'TT']
probabilities = [1/4, 1/4, 1/4, 1/4]

plt.figure(figsize=(10, 6))
plt.bar(outcomes, probabilities, color='lightgreen', edgecolor='black')
plt.xlabel('Coin Toss Outcomes')
plt.ylabel('Probability')
plt.title('Probability Distribution of Tossing Two Fair Coins')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# -------------------------
# 3. Mutually Exclusive and Non-Mutually Exclusive Events
# -------------------------
def mutually_exclusive():
    x = np.linspace(0, 1, 100)
    event_a = x
    event_b = 1 - x

    plt.figure(figsize=(10, 6))
    plt.plot(x, event_a, label='P(A)', color='blue')
    plt.plot(x, event_b, label='P(B)', color='orange')
    plt.fill_between(x, event_a, 0, color='blue', alpha=0.2)
    plt.fill_between(x, event_b, 0, color='orange', alpha=0.2)

    plt.title('Mutually Exclusive Events (P(A \u2229 B) = 0)')
    plt.xlabel('Event Space')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    plt.show()

mutually_exclusive()

# Non-Mutually Exclusive Events
def non_mutually_exclusive():
    x = np.linspace(0, 1, 100)
    event_a = np.sin(np.pi * x)
    event_b = np.cos(np.pi * x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, event_a, label='P(A)', color='blue')
    plt.plot(x, event_b, label='P(B)', color='orange')
    plt.plot(x, event_a * event_b, label='P(A \u2229 B)', color='green')
    plt.fill_between(x, event_a * event_b, 0, color='green', alpha=0.2)

    plt.title('Non-Mutually Exclusive Events (P(A \u2229 B) \u2260 0)')
    plt.xlabel('Event Space')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    plt.show()

non_mutually_exclusive()

# -------------------------
# 4. Birthday Problem (Probability of a Shared Birthday)
# -------------------------
def birthday_problem():
    num_people = np.arange(1, 101)
    probabilities = [1 - np.prod([(365 - i) / 365 for i in range(k)]) for k in num_people]

    plt.figure(figsize=(10, 6))
    plt.plot(num_people, probabilities, color='purple')
    plt.title('Birthday Problem: Probability of Shared Birthday')
    plt.xlabel('Number of People')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.show()

birthday_problem()

# -------------------------
# 5. Complement of Probability
# -------------------------
def complement_probability():
    probabilities = np.linspace(0, 1, 100)
    complements = 1 - probabilities

    plt.figure(figsize=(10, 6))
    plt.plot(probabilities, complements, label='1 - P(A)', color='red')
    plt.title('Complement of Probability')
    plt.xlabel('P(A)')
    plt.ylabel("P(A')")
    plt.grid(True)
    plt.legend()
    plt.show()

complement_probability()

# -------------------------
# 6. 3D Visualization (Joint Probability)
# -------------------------
from mpl_toolkits.mplot3d import Axes3D

def joint_probability():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    X, Y = np.meshgrid(x, y)
    Z = X * Y

    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_title('Joint Probability Distribution P(A \u2229 B)')
    ax.set_xlabel('P(A)')
    ax.set_ylabel('P(B)')
    ax.set_zlabel('P(A \u2229 B)')

    plt.show()

joint_probability()

# -------------------------
# 7. Animated Visualization (Dice Rolling Simulation)
# -------------------------
def animate_dice():
    fig, ax = plt.subplots(figsize=(8, 6))
    outcomes = [1, 2, 3, 4, 5, 6]
    probabilities = np.zeros(6)

    bars = ax.bar(outcomes, probabilities, color='skyblue', edgecolor='black')
    ax.set_ylim(0, 1)
    ax.set_title('Simulated Probability of Rolling a Die')
    ax.set_xlabel('Dice Faces')
    ax.set_ylabel('Probability')

    def update(frame):
        roll = np.random.randint(1, 7)
        probabilities[roll - 1] += 1
        probs = probabilities / probabilities.sum()

        for bar, prob in zip(bars, probs):
            bar.set_height(prob)

        return bars

    ani = FuncAnimation(fig, update, frames=100, interval=100, repeat=False)
    plt.show()

animate_dice()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib_venn import venn2, venn3

# 1. Rolling Dice Visualization (Set Representation)
def dice_probability():
    outcomes = [1, 2, 3, 4, 5, 6]  # Use a list instead of a set
    probabilities = [1/6] * 6

    plt.bar(outcomes, probabilities, color='skyblue', edgecolor='black')
    plt.title('Probability Distribution for Rolling a Die')
    plt.xlabel('Outcomes')
    plt.ylabel('Probability')
    plt.ylim(0, 0.2)
    plt.show()


dice_probability()

# 2. Coin Tossing (Set Visualization with Venn Diagram)
def coin_toss_venn():
    event_A = {"Head"}
    event_B = {"Tail"}
    venn2([event_A, event_B], ('Heads', 'Tails'))
    plt.title("Venn Diagram for Coin Toss Outcomes")
    plt.show()

coin_toss_venn()

# 3. Mutually Exclusive Events (Sets)
def mutually_exclusive():
    event_A = {"1", "3", "5"}  # Odd numbers
    event_B = {"2", "4", "6"}  # Even numbers

    venn2([event_A, event_B], ('Odd Numbers', 'Even Numbers'))
    plt.title('Mutually Exclusive Events')
    plt.show()

mutually_exclusive()

# 4. Non-Mutually Exclusive Events
def non_mutually_exclusive():
    event_A = {"A", "B", "C"}  # Group A
    event_B = {"B", "C", "D"}  # Group B

    venn2([event_A, event_B], ('Group A', 'Group B'))
    plt.title('Non-Mutually Exclusive Events')
    plt.show()

non_mutually_exclusive()

# 5. Complement of Probability (Set View)
def complement_probability():
    universal_set = set(range(1, 11))  # Universal set {1, 2, ..., 10}
    event_A = {1, 2, 3, 4, 5}  # Event A

    event_A_complement = universal_set - event_A

    venn2([event_A, event_A_complement], ('Event A', 'Complement of A'))
    plt.title("Complement of Probability")
    plt.show()

complement_probability()

# 6. Sum of Probabilities
def sum_of_probabilities():
    events = ["A", "B", "C", "D"]
    probabilities = [0.2, 0.3, 0.1, 0.4]

    plt.bar(events, probabilities, color='lightgreen', edgecolor='black')
    plt.title('Sum of Probabilities for Events')
    plt.xlabel('Events')
    plt.ylabel('Probability')
    plt.ylim(0, 0.5)

    total_prob = sum(probabilities)
    plt.text(1.5, 0.45, f"Total = {total_prob:.1f}", color='red', fontsize=12)
    plt.show()

sum_of_probabilities()

# 7. Birthday Problem (Line Plot)
def birthday_problem():
    n_people = np.arange(1, 101)
    prob_no_shared = np.array([np.prod([(365 - i) / 365 for i in range(n)]) for n in n_people])
    prob_shared = 1 - prob_no_shared

    plt.plot(n_people, prob_shared, color='purple')
    plt.title('Birthday Problem: Probability of Shared Birthdays')
    plt.xlabel('Number of People')
    plt.ylabel('Probability')
    plt.grid()
    plt.show()

birthday_problem()

# 8. Joint and Disjoint Probabilities (3D Surface Plot)
def joint_probability_3D():
    from mpl_toolkits.mplot3d import Axes3D

    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = X * Y

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_title('Joint Probability 3D Surface')
    ax.set_xlabel('P(A)')
    ax.set_ylabel('P(B)')
    ax.set_zlabel('P(A \u2229 B)')
    plt.show()

joint_probability_3D()

# 9. Animated Dice Rolling Simulation
def animate_dice():
    fig, ax = plt.subplots()
    outcomes = np.arange(1, 7)
    counts = np.zeros(6)

    def update(frame):
        roll = np.random.randint(1, 7)
        counts[roll - 1] += 1
        ax.clear()
        ax.bar(outcomes, counts / counts.sum(), color='orange', edgecolor='black')
        ax.set_title('Simulated Dice Rolls Over Time')
        ax.set_xlabel('Outcome')
        ax.set_ylabel('Probability')
        ax.set_ylim(0, 0.5)

    ani = animation.FuncAnimation(fig, update, frames=200, repeat=False, interval=100)
    plt.show()

animate_dice()
