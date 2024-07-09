import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
import time

L = 5 # Number of lattice spins
d = 10 # Number of disorder realizations
b = 1  #interaction parameter for Hx

g = 0.9045
h = 0.8090
t = 0.8  # tau parameter
gamma_values = np.linspace(0.1, 0.9, 10)  # range of gamma values

# Function to calculate reduced density matrix for a single spin
def reduced_density_matrix_single_spin(wavefunction, site_index, L):
    psi = wavefunction.reshape([2] * L)
    psi = np.moveaxis(psi, site_index, 0)
    psi = psi.reshape(2, -1)
    rho_reduced = np.dot(psi, np.conjugate(psi.T))
    return rho_reduced


# the spin we are interested in
I = 2 
# number of evolution steps
n = 100 

# Dictionary to store coherence evolution for each gamma value
coherence_evolution = {}

# Storing the 2**L dimensional Hadamard matrix
H = hadamard(2 ** L) / np.sqrt(2 ** L)

# Action of the Z part of the Hamiltonian
z = np.array([1, -1])
Dz = np.zeros(2 ** L)
for i in range(L):
    id1 = np.ones(2 ** i)
    id2 = np.ones(2 ** (L - i - 1))
    Z = np.kron(id1, np.kron(z, id2))
    Dz = Z + Dz

# Action of the X part of the Hamiltonian for different disorder realization
x = np.array([1, -1, -1, 1])
Dx = np.zeros((d, 2 ** L), dtype=complex)
X = np.zeros(2 ** L, dtype=complex)

for i in range(L - 1):
    id1 = np.ones(2 ** i)
    id2 = np.ones(2 ** (L - i - 2))
    X += np.kron(id1, np.kron(x, id2))
X += np.kron(z, np.kron(np.ones(2 ** (L - 2)), z))  # PBC
Dx = b * np.tile(X, (d, 1))

# Main loop for different gamma values
for gamma in gamma_values:
    # Generate new disorder realizations for each gamma
    J = np.zeros((d, L))
    for i in range(d):
        J[i, :] = np.random.normal(0, 1, L)
    J = h + g * np.sqrt(1 - gamma ** 2) * J
    
    Dx_local = np.copy(Dx)
    for j in range(d):
        for i in range(L - 1):
            id1 = np.ones(2 ** i)
            id2 = np.ones(2 ** (L - i - 1))
            X = J[j, i] * np.kron(id1, np.kron(z, id2))
            Dx_local[j, :] = X + Dx_local[j, :]

    Dz_exp = np.exp(-1j * (g * gamma) * Dz * t / 2)
    Dx_exp = np.exp(-1j * Dx_local * t)
    C = np.zeros((d, n), dtype=complex)

    for j in range(d):
        v = np.zeros(2 ** L, dtype=complex)
        v[3] = 1  # particular basis state
        v = np.dot(H, v)

        for i in range(n):
            v = Dz_exp * v
            v = np.dot(H, v)
            v = Dx_exp[j, :] * v
            v = np.dot(H, v)
            v = Dz_exp * v
            u = reduced_density_matrix_single_spin(v, I, L)

            C[j, i] = 2 * np.abs(u[0, 1])

    Ci = np.sum(C, axis=0) / d
    coherence_evolution[gamma] = Ci

# Plotting coherence evolution for each gamma value
for gamma in gamma_values:
    plt.plot(np.arange(n), coherence_evolution[gamma], label=f'$\Gamma$ = {gamma:.2f}')

plt.xlabel("Time")
plt.ylabel("Coherence")
plt.title("Spin Coherence Evolution for Different Gamma Values")
plt.legend(loc='best')
plt.show()
