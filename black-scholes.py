import numpy as np
import matplotlib.pyplot as plt

# Parameters
r = 0.05       # risk-free rate
sigma = 0.2    # volatility
K = 50         # strike price
S_min, S_max = 1, 100  # asset price range
N = 500        # number of grid points

# Grid
S = np.linspace(S_min, S_max, N)
h = S[1] - S[0]  # grid spacing

# Coefficients from ODE
a = 1 / h**2
b = (r / sigma**2 - 0.5) / (2 * h)
c = -r / sigma**2

# Tridiagonal matrix A and RHS vector
A = np.zeros((N, N))
rhs = np.zeros(N)

for i in range(1, N - 1):
    A[i, i - 1] = a - b
    A[i, i]     = -2 * a + c
    A[i, i + 1] = a + b

# Boundary conditions
V0 = 0
VN = S_max - K

rhs[0] = V0
rhs[-1] = VN

A[0, 0] = 1
A[-1, -1] = 1

# Solve
V = np.linalg.solve(A, rhs)

# Plot
plt.plot(S, V, label="Option Value V(S)")
plt.xlabel("Stock Price S")
plt.ylabel("Option Value V")
plt.title("European Call Option Value (ODE via Finite Difference)")
plt.axvline(K, color='gray', linestyle='--', label="Strike Price K")
plt.legend()
plt.grid(True)
plt.show()
