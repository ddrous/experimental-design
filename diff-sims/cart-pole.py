
#%%

r"""
Stochastic Cart-Pole Simulation with Diffrax and JAX
Based on "Nesting Particle Filters for Experimental Design in Dynamical Systems"

This script models the stochastic cart-pole dynamics using Euler-Maruyama integration,
allows you to customize the design space (the applied force schedule), and
renders a high-quality OpenAI-Gym-style animation.
"""

import jax
import jax.numpy as jnp
import diffrax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
from IPython.display import HTML, display

# ==========================================
# 1. System Parameters (From Section D.3)
# ==========================================
# True hidden parameters (length, pole mass, cart mass)
L = 1.0 
M_P = 0.1
M_C = 1.0
G = 9.81

# Damping and stiffness constants
K1 = K2 = D1 = D2 = 0.01

# ==========================================
# 2. Control Policy / Design Space
# ==========================================
def force_fn(t):
    r"""
    Control over the design space. 
    This function defines the force \xi_t applied to the cart at time t.
    The paper specifies limits \xi_t \in [-5, 5].
    
    Here we use a sine wave to induce a swinging motion, but you can 
    change this to a PRBS (Pseudo-Random Binary Signal) or any neural policy.
    """
    return 0.5 * jnp.sin(2.0 * jnp.pi * 0.4 * t)

# ==========================================
# 3. SDE Definition
# ==========================================
def drift(t, y, args):
    r"""
    Deterministic part of the SDE (h(x_t, \xi_t, \theta)).
    y = [s (position), q (angle), v (velocity), omega (angular velocity)]
    """
    s, q, v, omega = y
    xi = force_fn(t)  # Evaluate force schedule at time t

    sin_q = jnp.sin(q)
    cos_q = jnp.cos(q)
    
    # Denominator term common to both accelerations
    denom = M_C + M_P * sin_q**2

    # Cart acceleration (v_dot or \ddot{s})
    v_dot = (xi + M_P * sin_q * (L * omega**2 + G * cos_q) 
             - (K1 * s + D1 * v) 
             - (K2 * q + D2 * omega) * (cos_q / L)) / denom

    # Pole angular acceleration (omega_dot or \ddot{q})
    omega_dot = (-xi * cos_q - M_P * L * omega**2 * cos_q * sin_q 
                 - (M_C + M_P) * G * sin_q 
                 - (K1 * s + D1 * v) * cos_q 
                 - (K2 * q + D2 * omega) * (cos_q**2 / L)) / (L * denom)

    # Return the derivative of the state: [\dot{s}, \dot{q}, \ddot{s}, \ddot{q}]
    return jnp.array([v, omega, v_dot, omega_dot])


def diffusion(t, y, args):
    r"""
    Stochastic part of the SDE (L).
    The paper adds noise [0, 0, 0.1, 0]^T to the cart velocity.
    """
    # Multiplied by a 1D Brownian motion in the solver
    return jnp.array([0.0, 0.0, 0.1, 0.0])

# ==========================================
# 4. Simulation Execution using Diffrax
# ==========================================
def simulate_cartpole():
    print("Simulating stochastic cart-pole SDE...")
    
    t0, t1 = 0.0, 10.0  # 10 seconds of simulation
    dt = 0.05           # Discretization step size 
    saveat = diffrax.SaveAt(ts=jnp.arange(t0, t1, dt))
    
    # Generate the underlying Brownian motion for the stochastic noise
    key = jax.random.PRNGKey(42)
    bm = diffrax.VirtualBrownianTree(t0, t1, tol=1e-3, shape=(), key=key)
    
    # Construct the SDE terms
    drift_term = diffrax.ODETerm(drift)
    diff_term = diffrax.ControlTerm(diffusion, bm)
    terms = diffrax.MultiTerm(drift_term, diff_term)
    
    # Initial state: s=0, q=0 (hanging down), v=0, omega=0
    y0 = jnp.array([0.0, 0.0, 0.0, 0.0])
    
    # Setup the solver (Euler-Maruyama as specified in the paper)
    solver = diffrax.Euler()
    
    # Solve the SDE
    sol = diffrax.diffeqsolve(terms, solver, t0, t1, dt0=dt, y0=y0, saveat=saveat)
    
    return sol.ts, sol.ys

# ==========================================
# 5. High-Quality OpenAI-Gym Style Rendering
# ==========================================
def visualize(ts, ys):
    print("Rendering animation...")
    
    # Convert JAX arrays to NumPy for Matplotlib
    times = np.array(ts)
    s_vals = np.array(ys[:, 0])
    q_vals = np.array(ys[:, 1])
    
    # Setup Figure and Axes
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_facecolor('#f4f4f9')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Remove borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    
    # Graphical Elements
    cart_w, cart_h = 0.8, 0.4
    cart = Rectangle((-cart_w/2, -cart_h/2), cart_w, cart_h, fc='#2c3e50', ec='black', lw=1, zorder=3)
    ax.add_patch(cart)
    
    # The pole
    pole_line, = ax.plot([], [], '-', lw=6, color='#e74c3c', zorder=2)
    hinge = Circle((0, 0), 0.08, color='#f1c40f', zorder=4)
    ax.add_patch(hinge)
    
    # A motion trail for the tip of the pole
    trail_len = 15
    trail_x, trail_y = [], []
    trail_line, = ax.plot([], [], '-', lw=3, color='#e74c3c', alpha=0.3, zorder=1)
    
    # Text labels
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, family='monospace')
    force_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=12, family='monospace')
    
    # Force visual indicator (an arrow that scales with applied force)
    force_arrow = ax.quiver(0, 0, 0, 0, color='green', scale=20, width=0.005, zorder=5)

    def init():
        cart.set_xy((-cart_w/2, -cart_h/2))
        pole_line.set_data([], [])
        trail_line.set_data([], [])
        force_arrow.set_UVC(0, 0)
        return cart, pole_line, hinge, trail_line, time_text, force_text, force_arrow

    def update(frame):
        s = s_vals[frame]
        q = q_vals[frame]
        t = times[frame]
        force_val = float(force_fn(t))
        
        # 1. Update Cart
        cart.set_xy((s - cart_w/2, -cart_h/2))
        hinge.center = (s, 0)
        
        # 2. Update Pole (hanging down means q=0)
        # Therefore, tip position is based on sin(q) and -cos(q)
        tip_x = s + L * np.sin(q)
        tip_y = -L * np.cos(q)
        pole_line.set_data([s, tip_x], [0, tip_y])
        
        # 3. Update Trail
        trail_x.append(tip_x)
        trail_y.append(tip_y)
        if len(trail_x) > trail_len:
            trail_x.pop(0)
            trail_y.pop(0)
        trail_line.set_data(trail_x, trail_y)
        
        # 4. Update HUD
        time_text.set_text(f'Time  : {t:.2f}s')
        
        force_color = '#27ae60' if force_val > 0 else '#c0392b'
        force_text.set_text(f'Force : {force_val:+.2f} N')
        force_text.set_color(force_color)
        
        # 5. Update Force Arrow
        # Move arrow to cart center, scale length by force
        force_arrow.set_offsets(np.c_[s, 0])
        force_arrow.set_UVC(force_val, 0)
        force_arrow.set_color(force_color)

        return cart, pole_line, hinge, trail_line, time_text, force_text, force_arrow

    # Create Animation
    anim = animation.FuncAnimation(
        fig, update, frames=len(times), 
        init_func=init, blit=True, interval=50
    )
    
    plt.title("Stochastic Cart-Pole SDE Simulation", fontweight='bold')
    plt.tight_layout()
    
    # 1. Prevent the static image from rendering
    plt.close(fig) 
    
    # 2. Convert to an interactive HTML video player and return it
    return HTML(anim.to_jshtml())


#%%

if __name__ == "__main__":
    ts, ys = simulate_cartpole()
    
    # Capture the HTML animation and display it
    anim_html = visualize(ts, ys)
    display(anim_html)


# %%
