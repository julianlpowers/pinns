import torch
import numpy as np


def grad(outputs, inputs):
    """Computes the partial derivative of 
    an output with respect to an input.
    Args:
        outputs: (N, 1) tensor
        inputs: (N, D) tensor
    """
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True
    )

def hessian(outputs, inputs):
    """Computes the second derivative (Hessian) of 
    an output with respect to an input.
    Args:
        outputs: (N, 1) tensor
        inputs: (N, D) tensor
    Returns:
        (N, D) tensor of second derivatives
    """
    grad_outputs = torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True, retain_graph=True
    )[0]
    hess = []
    for i in range(inputs.shape[1]):
        grad2 = torch.autograd.grad(
            grad_outputs[:, i], inputs, grad_outputs=torch.ones_like(grad_outputs[:, i]), create_graph=True, retain_graph=True
        )[0][:, i]
        hess.append(grad2)
    return torch.stack(hess, dim=1)


def spring_mass_system(time, m, k, c, x0, v0):
    """
    Analytical solution for a damped spring-mass system.
    Args:
        time: array-like, time points
        m: mass
        k: spring constant
        c: damping constant
        x0: initial position
        v0: initial velocity
    Returns:
        x: displacement at each time point
    """
    omega0 = np.sqrt(k / m)
    zeta = c / (2 * np.sqrt(m * k))
    if zeta < 1:  # Underdamped
        omega_d = omega0 * np.sqrt(1 - zeta ** 2)
        A = x0
        B = (v0 + zeta * omega0 * x0) / omega_d
        x = np.exp(-zeta * omega0 * time) * (A * np.cos(omega_d * time) + B * np.sin(omega_d * time))
    elif zeta == 1:  # Critically damped
        A = x0
        B = v0 + omega0 * x0
        x = (A + B * time) * np.exp(-omega0 * time)
    else:  # Overdamped
        s1 = -omega0 * (zeta - np.sqrt(zeta ** 2 - 1))
        s2 = -omega0 * (zeta + np.sqrt(zeta ** 2 - 1))
        A = (v0 - s2 * x0) / (s1 - s2)
        B = (s1 * x0 - v0) / (s1 - s2)
        x = A * np.exp(s1 * time) + B * np.exp(s2 * time)
    return x


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Parameters
    m = 1.0      # mass (kg)
    k = 4.0      # spring constant (N/m)
    c = 0.5      # damping constant (N s/m)
    x0 = 1.0     # initial position (m)
    v0 = 0.0     # initial velocity (m/s)

    t = np.linspace(0, 10, 200)
    x = spring_mass_system(t, m, k, c, x0, v0)

    plt.plot(t, x)
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.title("Damped Spring-Mass System: x0=1, v0=0")
    plt.grid(True)
    plt.show()
