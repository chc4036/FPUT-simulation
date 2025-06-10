import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as ani

# Parameters
N = 32
dt = 0.3
alpha = 0.3
beta = 0.3
dtq = dt**2  # dt squared for second derivative terms

# Initial conditions
initial_y = [np.sin(np.pi * i / (N-1)) for i in range(N)]
prev_state = initial_y.copy()  # State at time t-1
curr_state = initial_y.copy()  # State at time t

# Initialize plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.set_title('FPUT node displacements')
ax2.set_title('Node Energy Distribution')
ln, = ax1.plot([], [], 'ro-')
ln2, = ax2.plot([], [], 'bo-')

# Time elapsed display
time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=12)

# Initialize axes
def init():
    ax1.set_xlim(0, N-1)
    ax1.set_ylim(-2.5, 2.5)
    ax2.set_xlim(0, N-1)
    ax2.set_ylim(0, 3) 
    time_text.set_text('t = 0.0')  # Initialize time display
    return ln, ln2, time_text

# Animation update function
def update(frame):
    global prev_state, curr_state
    
    next_state = curr_state.copy()  # Start with current state
    
    #positions for inner nodes (1 to N-2)
    for i in range(1, N-1):
        next_state[i] = 2*curr_state[i] - prev_state[i] + dtq * (
            curr_state[i-1] - 2*curr_state[i] + curr_state[i+1] +
            alpha * ((curr_state[i+1] - curr_state[i])**2 - (curr_state[i] - curr_state[i-1])**2) +
            beta * ((curr_state[i+1] - curr_state[i])**3 - (curr_state[i] - curr_state[i-1])**3)
        )
    
    #node energies at new state (time t+1)
    v = [(next_state[i] - curr_state[i]) / dt for i in range(N)]  # Velocities
    
    #spring energies (between nodes)
    V_springs = []
    for j in range(N-1):
        dy = next_state[j+1] - next_state[j]
        Vj = 0.5 * dy**2 + (alpha/3) * dy**3 + (beta/4) * dy**4
        V_springs.append(Vj)
    
    #energy per node
    U = np.zeros(N)  # Potential energy
    K = np.zeros(N)  # Kinetic energy
    
    # Endpoints (fixed)
    U[0] = 0.5 * V_springs[0] if N > 1 else 0
    U[N-1] = 0.5 * V_springs[-1] if N > 1 else 0
    
    # Internal nodes
    for i in range(1, N-1):
        U[i] = 0.5 * (V_springs[i-1] + V_springs[i])
    
    # Kinetic energy
    K = [0.5 * vi**2 for vi in v]
    
    # Total energy per node
    E = [K[i] + U[i] for i in range(N)]

    # Update states for next iteration
    prev_state = curr_state
    curr_state = next_state
    
    # Update plots
    ln.set_data(range(N), next_state)
    ln2.set_data(range(N), E)
    
    # Update time display
    elapsed_time = frame * dt
    time_text.set_text(f't = {elapsed_time:.1f}')
    
    # Adjust energy plot y-axis
    current_max_energy = max(E) * 1.1 if max(E) > 0 else 1
    ax2.set_ylim(0, current_max_energy)
    
    return ln, ln2, time_text

#animation
simulation = ani.FuncAnimation(
    fig, 
    update, 
    frames=10000, 
    init_func=init,
    blit=True, 
    interval=1
)

plt.tight_layout()
plt.show()