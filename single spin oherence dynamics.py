# calculating single spin coherences

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard


# e = 0.8 # epsilon factor interaction strength for Hz
L = 8  # Number of lattice spins
d = 100 # Number of disorder realizations
# h = 0.5 # Disorder strength
J = np.zeros((d, L)) #disorder realizations
b = 1  #interaction parameter for Hx

g = 0.9045
h = 0.8090
t = 1  # tau parameter
T = 0.1  # gamma paramter

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
Dz = np.exp(-1j*(g*T)*Dz*t)


# Action of the X part of the Hamiltonian for different disorder realization
x = np.array([1, -1, -1, 1])
Dx = np.zeros((d, 2**L), dtype = complex)
X = np.zeros(2**L, dtype = complex)

for i in range(L-1):
    id1 = np.ones(2**(i))
    id2 = np.ones(2**(L- i -2))
    X += np.kron(id1, np.kron(x, id2))
X += np.kron(z, np.kron(np.ones(2**(L-2)), z))   #PBC
Dx = b*np.tile(X, (d,1))

for j in range(d):
    for i in range(L-1):
        id1 = np.ones(2**(i))
        id2 = np.ones(2**(L- i -1))
        X = J[j,i]*np.kron(id1, np.kron(z, id2))
        Dx[j,:] = X + Dx[j,:]

Dx = np.exp(-1j*Dx*t)



def reduced_density_matrix_single_spin(wavefunction, site_index, L):
    psi = wavefunction.reshape([2] * L)
    psi = np.moveaxis(psi, site_index, 0)
    psi = psi.reshape(2, -1)
    rho_reduced = np.dot(psi, np.conjugate(psi.T))
    
    return rho_reduced


def fwht(a):
    h = 1
    while h < len(a):
        for i in range(0, len(a), h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2
    return a / np.sqrt(len(a))


I =  7 # the spin we are interested in

n = 1000 # number of evolution of steps




# Evolution of the state and coherences
C = np.zeros((d, n+1), dtype = complex)
C[:,0] = 0

for j in range(d):
    v = np.zeros(2**L, dtype = complex)
    v[3] = 1 #particular basis state
    
    for i in range(n):
        v = np.dot(H,v)
        v = Dz*v 
        v = np.dot(H,v)
        v = Dx[j,:]*v
        u = reduced_density_matrix_single_spin(v,I,L)
        
        C[j,i+1] = 2*np.abs(u[0,1])
    print()  
        
        
Ci = np.sum(C, axis = 0 )/d
plt.plot(np.arange(n+1),Ci)
plt.xlabel("Time")
plt.axhline(np.average(Ci[10:]), color = 'r', linestyle = '--', label = "Avg = " + str(round(np.real(np.average(Ci[10:]) ),3)))
plt.legend(loc = 'best')   
plt.ylabel("Coherence")
plt.title("Spin at site " +str(I)+", for L = " +str(L) +", $\Gamma$ = " +str(T)+", realizations = " +str(d))
plt.savefig("Spin at site " +str(I)+", for L = " +str(L) +", Gamma = " +str(T)+", realizations = " +str(d)+"tau = "+str(t)+" with PBC.png")
# plt.title("Spin at site " +str(I)+", for L = " +str(L) +", h = " + str(h) +", $J_z$ = " +str(e) + ", $J_x$ = " +str(b)+ ", realizations = " +str(d))
# plt.savefig("Spin at site " +str(I)+", for L = " +str(L) +", h = " + str(h) +", Jz = " +str(e) + ", Jx = " +str(b)+ ", realizations = " +str(d)+".png")
plt.show()