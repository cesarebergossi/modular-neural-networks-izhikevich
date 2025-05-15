import numpy as np
import matplotlib.pyplot as plt
from iznetwork import IzNetwork

class ModularNetwork:
    """
    A class representing a modular network of neurons, with both excitatory and inhibitory neurons,
    and a modular structure where neurons are grouped into modules.
    """
    def __init__(self, p, num_modules=8, neurons_per_module=100, num_inh=200, Dmax=20):
        """
        Initializes the modular network with the given parameters.

        :param p: Probability of rewiring between modules
        :param num_modules: Number of modules in the network
        :param neurons_per_module: Number of neurons per module (excitatory)
        :param num_inh: Number of inhibitory neurons
        :param Dmax: Maximum delay between neurons
        """

        self.p = p
        self.num_modules = num_modules
        self.neurons_per_module = neurons_per_module
        self.num_exc = num_modules * neurons_per_module
        self.num_inh = num_inh
        self.total_neurons = self.num_exc + self.num_inh

        # Initialize IzNetwork
        self.network = IzNetwork(self.total_neurons, Dmax=Dmax)

        # Set neuron parameters and generate the network connectivity
        self.set_neuron_parameters()
        self.generate_modular_connectivity()

    def set_neuron_parameters(self):
        """
        Sets the neuron parameters for both excitatory and inhibitory neurons.
        """
        r = np.random.random(self.total_neurons)

        a = np.zeros(self.total_neurons)
        a[0:self.num_exc] = 0.02  # Excitatory neurons
        a[self.num_exc:] = 0.02 + 0.08 * r[self.num_exc:]  # Inhibitory neurons

        b = np.zeros(self.total_neurons)
        b[0:self.num_exc] = 0.2  
        b[self.num_exc:] = 0.25 - 0.05 * r[self.num_exc:]  
        
        c = np.zeros(self.total_neurons)
        c[0:self.num_exc] = -65 + 15 * r[0:self.num_exc] ** 2  
        c[self.num_exc:] = -65 

        d = np.zeros(self.total_neurons)
        d[0:self.num_exc] = 8 - 6 * r[0:self.num_exc] ** 2 
        d[self.num_exc:] = 2  
        
        self.network.setParameters(a, b, c, d)

    def generate_modular_connectivity(self, m=1000):
        """
        Generates the modular connectivity of the network, with excitatory and inhibitory connections.
        This method includes random rewiring between modules with probability p, and other intra- and
        inter-module connections.
        """

        p_rewire = self.p

        # Scaling factors
        exc_exc_scale = 17
        exc_inh_scale = 50
        inh_exc_scale = 2
        inh_inh_scale = 1

        # Initialize weight and delay matrices
        self.delay_matrix = np.ones((self.total_neurons, self.total_neurons), dtype=int)
        self.weight_matrix = np.zeros((self.total_neurons, self.total_neurons))
        
        # Create m random intra-module excitatory-excitatory edges for each module
        for module in range(self.num_modules):
            start_idx = module * self.neurons_per_module
            end_idx = start_idx + self.neurons_per_module
            
            # Assign m directed excitatory connections within the module
            for _ in range(m):
                src = np.random.randint(start_idx, end_idx)  # Random source neuron
                tgt = np.random.randint(start_idx, end_idx)  # Random target neuron
                
                while self.weight_matrix[src, tgt] > 0 or src == tgt:  # Avoid repeated edges and self-connections
                    src = np.random.randint(start_idx, end_idx) 
                    tgt = np.random.randint(start_idx, end_idx) 
                    
                self.weight_matrix[src, tgt] = exc_exc_scale  # Excitatory-to-excitatory weight
                self.delay_matrix[src, tgt] = np.random.randint(1, 21)  # Random delay between 1 and 20 ms
            
        # Rewire connections between modules with probability p_rewire
        edges = np.where(self.weight_matrix > 0)
        num_edges = len(edges[0])
        
        for i in range(num_edges):
            source, target = edges[0][i], edges[1][i]
            
            # Check if source and target are in the same module
            source_module = source // self.neurons_per_module
            target_module = target // self.neurons_per_module

            if source_module == target_module:
                if np.random.random() < p_rewire:
                    # Remove intra-community connection
                    self.weight_matrix[source, target] = 0

                    # Rewire to a different community
                    new_module = np.random.choice([m for m in range(self.num_modules) if m != source_module])
                    new_target = np.random.randint(new_module * self.neurons_per_module, (new_module + 1) * self.neurons_per_module)
                    
                    self.weight_matrix[source, new_target] = exc_exc_scale  # Directed inter-community connection
                    self.delay_matrix[source, new_target] = np.random.randint(1, 21)

        # Focal Excitatory-to-Inhibitory connections within the same module
        for inh_neuron in range(self.num_exc, self.total_neurons):
            module = inh_neuron % self.num_modules  # Each inhibitory neuron is assigned to a module
            start_idx = module * self.neurons_per_module
            excitatory_targets = np.random.choice(range(start_idx, start_idx + self.neurons_per_module), 4, replace=False)  # Randomly select 4 excitatory neurons
            for target in excitatory_targets:
                self.weight_matrix[target, inh_neuron] = np.random.uniform(0, 1) * exc_inh_scale  # Excitatory-to-inhibitory weight
                self.delay_matrix[target, inh_neuron] = 1  # Fixed delay of 1

        # Inhibitory-to-Excitatory (inh-exc) Diffuse Connections
        for inh_neuron in range(self.num_exc, self.total_neurons):
            for exc_neuron in range(self.num_exc):
                self.weight_matrix[inh_neuron, exc_neuron] = np.random.uniform(-1, 0) * inh_exc_scale  # Inhibitory-to-excitatory weight
                self.delay_matrix[inh_neuron, exc_neuron] = 1  # Fixed delay of 1

        # Inhibitory-to-Inhibitory (inh-inh) Diffuse Connections
        for inh_neuron in range(self.num_exc, self.total_neurons):
            for other_inh_neuron in range(self.num_exc, self.total_neurons):
                if inh_neuron != other_inh_neuron:
                    self.weight_matrix[inh_neuron, other_inh_neuron] = np.random.uniform(-1, 0) * inh_inh_scale  # Inhibitory-to-inhibitory weight
                    self.delay_matrix[inh_neuron, other_inh_neuron] = 1  # Fixed delay of 1

        # Ensure no self-connections
        np.fill_diagonal(self.weight_matrix, 0)

        self.network.setDelays(self.delay_matrix)
        self.network.setWeights(self.weight_matrix)

    def run_simulation(self, T):
        """
        Run the simulation for T time steps and record the firing rates and membrane potentials.

        :param T: Total time steps for the simulation
        :return: firing_matrix (recorded firing rates) and V (membrane potentials)
        """
        # Matrices to store firing rates per module and membrane potentials
        firing_matrix = np.zeros((T, self.num_modules))
        V = np.zeros((T, self.total_neurons))

        # Run simulation for T ms and record activity
        for t in range(T):
            poisson_samples = np.random.poisson(0.01, self.total_neurons)
            I = 15 * (poisson_samples > 0)

            self.network.setCurrent(I)

            # Update network state and record membrane potentials
            self.network.update()
            V[t, :], _ = self.network.getState()

            # Determine which neurons fired and record firing rates per module
            fired = V[t, :] > 29
            for i in range(self.num_modules):
                start_idx = i * self.neurons_per_module
                end_idx = start_idx + self.neurons_per_module
                firing_matrix[t, i] = np.sum(fired[start_idx:end_idx])
            
        return firing_matrix, V
    
    def connectivity_matrix(self):
        """
        Visualize the excitatory connectivity matrix.
        """
        # Indices of non-zero connections
        y, x = np.where(self.weight_matrix[:self.num_exc, :self.num_exc] > 0)

        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, s=1, color='blue')
        plt.title(f'Excitatory Connectivity Matrix (p = {self.p})')
        plt.xlabel('Target Neuron')
        plt.ylabel('Source Neuron')

        plt.ylim(self.num_exc, 0)
        plt.xlim(0, self.num_exc)
        plt.savefig(f'Connectivity_Matrix_p_{self.p}.pdf', format='pdf')
        plt.show()

    def plot_firing_and_mean_firing_rate(self, T, p):
        """
        Plot the raster plot of neuron firing and mean firing rate per module.

        :param T: Total time steps for the simulation
        """

        firing_matrix, V = self.run_simulation(T)

        plt.figure(figsize=(12, 6))

        # Raster plot of neuron firing
        plt.subplot(2, 1, 1)
        plt.suptitle(f'Firing Activity of Modular Network')
        plt.title(f'Neuron Firing Raster Plot (p = {p})')
        t, n = np.where(V > 29)  # Time and neuron index of spikes
        plt.scatter(t, n, s=3, color='black')
        plt.ylim(self.num_exc, 0)
        plt.xlim(0, T)
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron Index')

        # Mean firing rate per module
        plt.subplot(2, 1, 2)
        mean_firing_rates = np.zeros([50, self.num_modules])
        # Compute mean firing rate in non-overlapping windows of 50 ms, sliding every 20 ms
        for i in range(0, T, 20):
            window_index = i // 20
            mean_firing_rates[window_index, :] = np.mean(firing_matrix[i:i + 50, :], axis=0)
        time_points = np.arange(0, T, 20)
        for i in range(self.num_modules):
            plt.plot(time_points, mean_firing_rates[:, i], label=f"Module {i+1}")

        plt.title('Mean Firing Rate per Module')
        plt.xlabel('Time (ms)')
        plt.ylabel('Mean Firing Rate (spikes/ms)')
        plt.ylim(0)
        plt.xlim(0, T)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'Firing_Activity_p_{p}.pdf', format='pdf')
        plt.show()

def main():
    """
    Main function to run the simulation with different rewiring probabilities and generate plots.
    """
    
    p_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    T = 1000  # Simulation time in ms

    for p in p_values:
        network = ModularNetwork(p)
        network.connectivity_matrix()
        network.plot_firing_and_mean_firing_rate(T, p)

if __name__ == "__main__":
    main()