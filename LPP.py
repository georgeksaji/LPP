
import numpy as np
import matplotlib.pyplot as plt


def get_coefficients(prompt):
    return list(map(float, input(prompt).split()))


problem_type = input("Is this a maximization or minimization problem? (Enter 'max' or 'min'): ").strip().lower()
if problem_type not in ['max', 'min']:
    print("Invalid input! Please enter 'max' or 'min'.")
    exit()

print("\nEnter coefficients for the objective function (Z = ax + by):")
obj_coeff = get_coefficients("Enter a and b (separated by space): ")
a, b = obj_coeff

num_constraints = int(input("\nEnter the number of constraints: "))

constraints = []
print("\nEnter constraints in the form (ax + by <= c):")
for i in range(num_constraints):
    constraint = get_coefficients(f"Enter a, b, and c for constraint {i+1} (separated by space): ")
    constraints.append(constraint)

x = np.linspace(0, 10, 400)


for i, (a_con, b_con, c_con) in enumerate(constraints):
    y = (c_con - a_con * x) / b_con
    plt.plot(x, y, label=f'Constraint {i+1}: {a_con}x + {b_con}y <= {c_con}')

# Plot the non-negativity constraints
plt.axhline(0, color='black', linestyle='--', label=r'$y \geq 0$')
plt.axvline(0, color='black', linestyle='--', label=r'$x \geq 0$')

# Shade the feasible region
y_min = np.minimum.reduce([(c_con - a_con * x) / b_con for (a_con, b_con, c_con) in constraints])
plt.fill_between(x, 0, y_min, where=(y_min >= 0), color='gray', alpha=0.5)

# Find corner points
corner_points = [(0, 0)]  # Origin

# Intersection of constraints
for i in range(len(constraints)):
    for j in range(i + 1, len(constraints)):
        A = np.array([[constraints[i][0], constraints[i][1]], [constraints[j][0], constraints[j][1]]])
        B = np.array([constraints[i][2], constraints[j][2]])
        try:
            intersection_point = np.linalg.solve(A, B)
            if intersection_point[0] >= 0 and intersection_point[1] >= 0:  # Check non-negativity
                corner_points.append(tuple(intersection_point))
        except np.linalg.LinAlgError:
            continue

# Intersection with axes
for (a_con, b_con, c_con) in constraints:
    if a_con != 0:
        corner_points.append((c_con / a_con, 0))  # Intersection with x-axis
    if b_con != 0:
        corner_points.append((0, c_con / b_con))  # Intersection with y-axis

# Remove duplicate corner points
corner_points = list(set(corner_points))

# Evaluate the objective function at each corner point
Z_values = [a * x + b * y for (x, y) in corner_points]

# Determine optimal point based on problem type
if problem_type == 'max':
    optimal_point = corner_points[np.argmax(Z_values)]
    optimal_value = np.max(Z_values)
else:
    optimal_point = corner_points[np.argmin(Z_values)]
    optimal_value = np.min(Z_values)

# Plot the optimal point
plt.scatter(*optimal_point, color='red', label=f'Optimal Point: {optimal_point}')
plt.annotate(f'Optimal Z = {optimal_value}', optimal_point, textcoords="offset points", xytext=(10, -10), ha='center')

# Set labels and title
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('Graphical Method for LPP')
plt.legend()
plt.grid(True)
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.show()

# Print results
print("\nCorner Points:", corner_points)
print("Optimal Point:", optimal_point)
print("Optimal Value:", optimal_value)
