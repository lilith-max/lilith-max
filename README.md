mport numpy as np
import matplotlib.pyplot as plt

# Constants
k_B = 1.380649e-23  # Boltzmann constant (J/K)
T = 298  # Temperature (K)
pi = 3.14159  # Pi
eta = 0.001  # Dynamic viscosity of water (Pa·s)
r_particle = 10e-9  # Radius of TiO2 nanoparticle (m)
k = 1e-3  # Reaction rate constant (s^-1)

# Initial concentration of the pollutant (mol/L)
C_initial = 1.0

# Calculate the diffusion coefficient D
D = k_B * T / (6 * pi * eta * r_particle)

# Define the spatial and temporal grid
Lx, Ly, Lz = 0.1, 0.1, 0.1  # Dimensions of the beaker (m)
Nx, Ny, Nz = 50, 50, 50  # Number of spatial points in each direction
Nt = 100  # Number of time steps
dx, dy, dz = Lx / (Nx - 1), Ly / (Ny - 1), Lz / (Nz - 1)  # Spatial step sizes
dt = 0.01  # Time step size

# Initialize the concentration array
C = np.ones((Nx, Ny, Nz)) * C_initial

# Function to update the concentration using the finite difference method
def update_concentration(C, D, k, dx, dy, dz, dt):
    C_new = np.copy(C)
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            for l in range(1, Nz-1):
                diffusion_term = D * (
                    (C[i+1, j, l] - 2*C[i, j, l] + C[i-1, j, l]) / dx**2 +
                    (C[i, j+1, l] - 2*C[i, j, l] + C[i, j-1, l]) / dy**2 +
                    (C[i, j, l+1] - 2*C[i, j, l] + C[i, j, l-1]) / dz**2
                )
                reaction_term = -k * C[i, j, l]
                C_new[i, j, l] += dt * (diffusion_term + reaction_term)
    return C_new

# Time evolution
for t in range(Nt):
    C = update_concentration(C, D, k, dx, dy, dz, dt)

# Plot the final concentration profile in a slice of the beaker
plt.imshow(C[:, :, Nz//2], cmap='viridis', origin='lower', extent=[0, Lx, 0, Ly])
plt.colorbar(label="Concentration (mol/L)")
plt.title("Concentration Profile of Pollutant in the Center Slice")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.show()

print(f"Final concentration profile slice at z={Lz/2}: {C[:, :, Nz//2]}")
