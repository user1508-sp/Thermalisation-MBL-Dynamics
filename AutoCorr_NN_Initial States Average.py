#only averaging over all initial states
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard

e = 1e-2 # epsilon factor
L = 10 # Number of lattice spins
J = np.random.rand(L-1)*4
print(J)

# defining state of the system stored as a bit string of length L (representing x polarised basis states) and converting it 
# into a 2**L dimensional vector
state = np.array([1, 0, 1, 0, 1])



# convert bit string array into the decimal number
k = np.dot(state, 2**np.arange(state.size)[::-1])
# define into a 2**L dimensional vector
v = np.identity(2**L, dtype = complex)


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
Dz = np.exp(-1j*(np.pi+e)*Dz)
Hz = np.diag(Dz)

# Action of the X part of the Hamiltonian
x = np.array([1, -1, -1, 1])
Dx = np.zeros(2**L)
for i in range(L-1):
    id1 = np.ones(2**(i))
    id2 = np.ones(2**(L- i -2))
    X = J[i]*np.kron(id1, np.kron(x, id2))
    Dx = X + Dx

Dx = np.exp(-1j*Dx)
Hx = np.diag(Dx)

print("Dx", Dx.shape)
print("Dz", Dz.shape)

I = 3 # the spin we are interested in
id1 = np.ones(2**(I-1))
id2 = np.ones(2**(L- I ))
Xi = np.kron(id1, np.kron(z, id2))
Xi0 = Xi

n = 10 # number of evolution of steps

# Evolution of the state
Ci = np.zeros(n + 1, dtype = complex)
Ci[0] = 1
for i in range(n):
    v = (H@v.T).T
    v = Dz[np.newaxis,:]*v 
    v = (H@v.T).T
    v = Dx[np.newaxis,:]*v
    vf = Xi*v
#     if(i == 2):
#         plt.imshow(np.real(Xi0*np.conj(v)*vf))
#         plt.colorbar()
    Ci[i+1] = np.sum(Xi0[:,np.newaxis]*np.conj(v)@vf.T)/(2**L)
    

# Plotting the autocorrelation function
plt.plot(np.arange(n+1), np.real(Ci))
plt.xlabel('Time')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation time evolution')
plt.show()

print(np.real(Ci))

