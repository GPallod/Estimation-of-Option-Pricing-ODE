import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
S = 50      # stock price
K = 50      # strike price
r = 0.05    # risk-free rate
T = 1.0     # time to maturity

# Define Black-Scholes call price as a function of sigma
def bs_call(sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Volatility grid
N = 300
sigma_min, sigma_max = 0.01, 1.0
sigma = np.linspace(sigma_min, sigma_max, N)
h = sigma[1] - sigma[0]

# Evaluate true V(sigma) from BS formula
V_true = bs_call(sigma)

# Finite difference ODE: V'' + c1 V' + c2 V = 0
# Choose arbitrary constants for illustration
c1 = -1.0
c2 = -10.0

# Build finite difference system
A = np.zeros((N, N))
rhs = np.zeros(N)

for i in range(1, N - 1):
    A[i, i - 1] = 1 / h**2 - c1 / (2 * h)
    A[i, i]     = -2 / h**2 + c2
    A[i, i + 1] = 1 / h**2 + c1 / (2 * h)

# Boundary conditions: use true values at endpoints
A[0, 0] = 1
rhs[0] = V_true[0]
A[-1, -1] = 1
rhs[-1] = V_true[-1]

# Solve
V_fd = np.linalg.solve(A, rhs)

# Plot
plt.plot(sigma, V_true, label='Exact Black-Scholes V(σ)', linewidth=2)
plt.plot(sigma, V_fd, '--', label='Numerical Solution (FD)', linewidth=2)
plt.xlabel('Volatility σ')
plt.ylabel('Option Price V')
plt.title('Option Price as a Function of Volatility')
plt.legend()
plt.grid(True)
plt.show()
