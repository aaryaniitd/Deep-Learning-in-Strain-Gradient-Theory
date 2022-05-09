import torch
from torch.autograd import grad
import matplotlib.pyplot as plt
#from sympy import *

dtype = torch.float
device = torch.device("cpu")

def generate_grid_1d(length, samples=50, initial_coordinate=0.0):
    x = torch.linspace(initial_coordinate, initial_coordinate + length, samples, requires_grad=True)
    return x.view(samples, 1)

def trapezoidal_integration_1d(y, x):
    dx = x[1] - x[0]
    result = torch.sum(y)
    result = result - (y[0] + y[-1]) / 2
    return result * dx

def get_derivative(y, x, n):
    #x = symbols('x')
    if n == 0:
        return y
    else:
        #dy_dx = Derivative(y,x).doit()
        dy_dx = grad(y, x, torch.ones(x.size()[0], 1, device=device), create_graph=True, retain_graph=True)[0]
        return get_derivative(dy_dx, x, n - 1)


def build_model(input_dimension, hidden_dimension, output_dimension):
    """Build a neural network of given dimensions."""

    modules=[]
    modules.append(torch.nn.Linear(input_dimension, hidden_dimension[0]))
    modules.append(torch.nn.Tanh())
    for i in range(len(hidden_dimension)-1):
        print('check',len(hidden_dimension))
        modules.append(torch.nn.Linear(hidden_dimension[i], hidden_dimension[i+1]))
        modules.append(torch.nn.Tanh())
    
    modules.append(torch.nn.Linear(hidden_dimension[-1], output_dimension))
    
    model = torch.nn.Sequential(*modules)

    return model

def plot_displacements_bar(x, u, u_analytic=None):
    fig, ax = plt.subplots(figsize=(4,3))
    ax.set_title("Displacements")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    if u_analytic != None:
        ax.plot(x.detach().numpy(), u_analytic(x.detach().numpy()),color='r', linewidth=2, label="u_analytic")
    ax.plot(x.detach().numpy(), u.detach().numpy(),color='k',linestyle=':',linewidth=5, label="u_pred")
    ax.legend()
    plt.show()
    fig.tight_layout()
    
def plot_stiffness_bar(x, EA, EA_analytic=None):
    fig, ax = plt.subplots(figsize=(4,3))
    ax.set_title("Stiffness")
    ax.set_xlabel("x")
    ax.set_ylabel("EA")
    ax.plot(x.detach().numpy(), EA.detach().numpy(),color='r', label="EA_pred")
    if EA_analytic != None:
        ax.plot(x.detach().numpy(), EA_analytic(x.detach().numpy()), label="EA_analytic")
    ax.legend()
    plt.show()
    fig.tight_layout()

