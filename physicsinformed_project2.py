import torch
import numpy as np
import matplotlib.pyplot as plt
import utilities
from torch.autograd import grad
import math

class PhysicsInformedBarModel:
    """A class used for the definition of Physics Informed Models for one dimensional bars."""

    def __init__(self, D1, D2, L, u0, dist_load):
        """Construct a PhysicsInformedBar model"""
        self.x0 = math.sqrt(D1)
        self.x = utilities.generate_grid_1d(L)
        self.u0 = u0
        self.dist_load = dist_load
        self.w0 = (D2 ** 2)/(D1 ** 3)
        self.model = utilities.build_model(1,[32,32,32,32,32],1)
        self.differential_equation_loss_history = None
        self.boundary_condition_loss_history = None
        self.total_loss_history = None
        self.optimizer = None

    def get_displacements(self, x):
        """Get displacements."""

        u = self.model(x)   # predict

        return u

    def costFunction(self, x):
        """Compute the cost function."""
        y = x.clone().detach().requires_grad_(True) / self.x0
        #print(x.view(50),y.view(50))
        # Differential equation loss (f)
        u = self.get_displacements(y)
        y_x = grad(u, y, torch.ones(y.size()[0], 1), create_graph= True, allow_unused = True)[0]

        y_x2 = grad(y_x, y, torch.ones(y.size()[0], 1), create_graph=True, allow_unused = True)[0]
        
        y_x3 = grad(y_x2, y, torch.ones(y.size()[0], 1), create_graph=True, allow_unused = True)[0]
        
        y_x4 = grad(y_x3, y, torch.ones(y.size()[0], 1), create_graph=True, allow_unused = True)[0]
        
        y_x5 = grad(y_x4, y, torch.ones(y.size()[0], 1), create_graph=True, allow_unused = True)[0]
        
        y_x6 = grad(y_x5, y, torch.ones(y.size()[0], 1), create_graph=True, allow_unused = True)[0]
        
        differential_equation_loss = y_x4 - y_x6 - self.dist_load(x)
        differential_equation_loss = torch.sum(differential_equation_loss ** 2).view(1)
        #print(y_x.view(50),y_x2.view(50),y_x3.view(50))
        
        #Define V,M,Q
        V = self.x0 * (y_x3 - y_x5)
        M = (self.x0 ** 2)*(y_x2 - y_x4)
        Q = (self.x0 ** 3)*y_x3

        # Boundary condition loss initialization
        boundary_condition_loss = 0

        # Sum over dirichlet boundary condition losses
        boundary_condition_loss += (u[0] - self.u0[0]) ** 2
        boundary_condition_loss += (y_x[0] - self.u0[1]) ** 2
        boundary_condition_loss += (y_x2[0] - self.u0[2]) ** 2
        # Sum over Neumann boundary condition losses
        boundary_condition_loss += (V[-1] - self.u0[3]) ** 2
        boundary_condition_loss += (M[-1] - self.u0[4]) ** 2
        boundary_condition_loss += (Q[-1] - self.u0[5]) ** 2
        

        return differential_equation_loss, boundary_condition_loss

    def closure(self):
        """Calculation of training error and gradient"""
        self.optimizer.zero_grad()
        u_pred = self.get_displacements(self.x)
        loss = self.costFunction(self.x)
        loss = loss[0] + loss[1]
        loss.backward(retain_graph=True)
        return loss

    def train(self, epochs, optimizer='Adam', **kwargs):
        """Train the model."""

        # Set optimizer
        if optimizer=='Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), **kwargs)
        
        elif optimizer=='LBFGS':
            self.optimizer = torch.optim.LBFGS(self.model.parameters(), **kwargs)

        # Initialize history arrays
        self.differential_equation_loss_history = np.zeros(epochs)
        self.boundary_condition_loss_history = np.zeros(epochs)
        self.total_loss_history = np.zeros(epochs)

        # Training loop
        for i in range(epochs):
            # Predict displacements
#             u_pred = self.get_displacements(self.x) 

            # Cost function calculation
            differential_equation_loss, boundary_condition_loss = self.costFunction(self.x)

            # Total loss
            total_loss = differential_equation_loss + boundary_condition_loss

            # Add energy values to history
            self.differential_equation_loss_history[i] += differential_equation_loss
            self.boundary_condition_loss_history[i] += boundary_condition_loss
            self.total_loss_history[i] += total_loss

            # Print training state
            self.print_training_state(i, epochs)
            
            # Update parameters (Neural network train)
            self.optimizer.step(self.closure)

    def print_training_state(self, epoch, epochs, print_every=100):
        """Print the loss values of the current epoch in a training loop."""

        if epoch == 0 or epoch == (epochs - 1) or epoch % print_every == 0 or print_every == 'all':
            # Prepare string
            string = "Epoch: {}/{}\t\tDifferential equation loss = {:.3e}\t\tBoundary condition loss = {:.3e}\t\tTotal loss = {:.3e}"

            # Format string and print
            print(string.format(epoch, epochs - 1, self.differential_equation_loss_history[epoch]*(10**7),
                                self.boundary_condition_loss_history[epoch]*(10**7), self.total_loss_history[epoch]*(10**7)))

    def plot_training_history(self, yscale='log'):
        """Plot the training history."""

        # Set up plot
        fig, ax = plt.subplots(figsize=(4,3))
        ax.set_title("Cost function history")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Cost function C")
        plt.yscale(yscale)

        # Plot data
        ax.plot(self.total_loss_history, 'k', linewidth=2, label="Total cost")
        ax.plot(self.differential_equation_loss_history, color='silver', linestyle='--', linewidth=2, label="Differential equation loss")
        ax.plot(self.boundary_condition_loss_history, color='r', linestyle='-.', linewidth=2, label="Boundary condition loss")
        

        ax.legend()
        fig.tight_layout()
        plt.show()   