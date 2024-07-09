# calculating entanglement negativity evolution

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
from numpy.linalg import eigvals



L = 8 # Number of lattice spins
d = 400 # Number of disorder realizations
J = np.zeros((d, L)) #disorder realizations
b = 1  #interaction parameter for Hx

g = 0.9045
h = 0.8090
t = 0.8  # tau parameter
T = 0.1 # gamma paramter

for i in range(d):
    J[i,:] = np.random.normal(0,1,L)

J = h + g*np.sqrt(1 - T**2)*J

# Storing the 2**L dimensional Hadamard matrix
H = hadamard(2**L)/np.sqrt(2**L)


# Action of the Z part of the Hamiltonian
z = np.array([1,-1])
Dz = np.zeros(2**L)
for i in range(L):
    id1 = np.ones(2**(i))
    id2 = np.ones(2**(L- i -1))
    Z = np.kron(id1, np.kron(z, id2))
    Dz = Z + Dz
Dz = np.exp(-1j*(g*T)*Dz*t/2)


# Action of the X part of the Hamiltonian for different disorder realization
x = np.array([1, -1, -1, 1])
Dx = np.zeros((d, 2**L), dtype = complex)
X = np.zeros(2**L, dtype = complex)

for i in range(L-1):
    id1 = np.ones(2**(i))
    id2 = np.ones(2**(L- i -2))
    X += np.kron(id1, np.kron(x, id2))
#X += np.kron(z, np.kron(np.ones(2**(L-2)), z))   #PBC
Dx = b*np.tile(X, (d,1))

for j in range(d):
    for i in range(L-1):
        id1 = np.ones(2**(i))
        id2 = np.ones(2**(L- i -1))
        X = J[j,i]*np.kron(id1, np.kron(z, id2))
        Dx[j,:] = X + Dx[j,:]

Dx = np.exp(-1j*Dx*t)


# Function to calculate reduced density matrix for two spins
def reduced_density_matrix_two_spins(wavefunction, site_index1, site_index2, L):
    psi = wavefunction.reshape([2] * L)
    psi = np.moveaxis(psi, [site_index1, site_index2], [0, 1])    
    psi = psi.reshape(4, -1)    
    rho_reduced = np.dot(psi, np.conjugate(psi.T))
    
    return rho_reduced

# position of the entangled spins at I and I+R+1
I = 0
R = 3

n = 100 # nu7mber of evolution of steps

# # specifying the bell pair initial state 
# v1 = np.array([1, 0, 0, 1, 1])
# v2 = np.array([1, 1, 1, 1, 1])

# # convert bit string array into the decimal number
# k1 = np.dot(v1, 2**np.arange(v1.size)[::-1])
# k2 = np.dot(v2, 2**np.arange(v2.size)[::-1])
# # define into a 2**L dimensional vector
k2 = 2**(L-I-1) + 2**(L-I-R-2)
state = np.zeros(2**L, dtype = complex)
state[0] = 1
state[k2] = 1

# store the entanglement negativity
entanglement_negativity = np.zeros((d, n), dtype = complex)

for j in range(d):
    v = state/np.linalg.norm(state) 
    v = np.dot(H,v)
    
    for i in range(n):     
        v = Dz*v 
        v = np.dot(H,v)
        v = Dx[j,:]*v
        v = np.dot(H,v)
        v = Dz*v 
        rho = reduced_density_matrix_two_spins(v,I,I+R+1,L)
        rho = rho.reshape(2, 2, 2, 2).swapaxes(1, 3).reshape(4, 4)
        eigenvalues = eigvals(rho)
        
        entanglement_negativity[j,i] = np.sum(np.abs(eigenvalues[eigenvalues < 0]))

entanglement_negativity = np.sum(entanglement_negativity, axis = 0)/d
plt.plot(np.arange(n), entanglement_negativity)
plt.xlabel('Time steps')
plt.ylabel('Entanglement Negativity')
plt.title("Spin at site " +str(I)+", for L = " +str(L)+", for R = " +str(R) +", $\Gamma$ = " +str(T)+", realizations = " +str(d))
plt.show()



        
       


