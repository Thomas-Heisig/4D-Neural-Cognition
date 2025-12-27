# Glossary

This glossary defines key terms and concepts used in the 4D Neural Cognition project.

## A

**Action Potential**  
Also called a "spike". A rapid rise and fall in membrane potential when a neuron fires.

**Aging**  
The process by which neurons increase in age with each simulation step, eventually leading to death.

**Area (Brain Area)**  
A defined region in the 4D lattice designated for a specific function (e.g., V1_like for vision, A1_like for audition).

**Axon**  
In biological neurons, the long projection that carries signals away from the cell body. In our simulation, represented implicitly through synapses.

## B

**Benchmark**  
A standardized test used to evaluate and compare performance of different network configurations.

**BrainModel**  
The main class that contains all neurons, synapses, and configuration for a neural network.

**Brain Area**  
See *Area*.

## C

**Callback**  
A user-defined function that is called during simulation for monitoring or intervention.

**Cell Lifecycle**  
The system managing neuron aging, health, death, and reproduction with mutation.

**Configuration**  
The JSON file or dictionary specifying all parameters of a brain model (lattice size, neuron parameters, areas, etc.).

**Connection Probability**  
The probability that any two neurons will be connected by a synapse during random initialization.

**Coordinate System**  
The 4D space (x, y, z, w) in which neurons are positioned.

## D

**Dataclass**  
A Python class decorator that provides automatic initialization and representation methods. Used for Neuron and Synapse classes.

**Death**  
When a neuron's health reaches zero or it exceeds maximum age, it is removed from the network.

**Delay (Synaptic)**  
The number of simulation steps before a spike propagates across a synapse (simulates axonal transmission time).

**Dendrite**  
In biological neurons, branched projections that receive signals. In our simulation, represented by synaptic connections.

**Density**  
The fraction of possible neuron positions that are actually filled with neurons (0.0 to 1.0).

**Digital Sense**  
A novel sensory modality for processing abstract data patterns (text, structured data, etc.).

## E

**Environment**  
In the tasks framework, an abstract representation of a task that the neural network interacts with.

**Excitatory**  
A synapse or neuron that increases the membrane potential of its target (makes it more likely to spike).

**External Input**  
Current injected into a neuron from outside the network (e.g., from sensory input).

## F

**Flask**  
The Python web framework used for the web interface.

**Fourth Dimension**  
See *W-dimension*.

## G

**Generation**  
A counter tracking how many reproduction cycles separate a neuron from the original population.

**GIL (Global Interpreter Lock)**  
Python's mechanism that prevents true parallel execution of threads, limiting performance.

## H

**HDF5**  
Hierarchical Data Format, used for efficient compressed storage of large neural networks.

**Health**  
A metric (0.0 to 1.0) representing a neuron's viability. Decays over time; when zero, neuron dies.

**Heatmap**  
A visualization showing neuron activity levels using color intensity.

**Hebbian Plasticity**  
Learning rule: "Cells that fire together, wire together." Synapses strengthen when pre- and post-synaptic neurons are both active.

## I

**Inhibitory**  
A synapse or neuron that decreases the membrane potential of its target. Not yet implemented (planned).

**Initialization**  
The process of creating neurons and synapses in a new network.

**Integrate-and-Fire**  
See *LIF Model*.

## J

**JSON (JavaScript Object Notation)**  
A text format used for configuration files and small model storage.

## K

**Knowledge Database**  
A SQLite database storing training examples that can be used for pre-training or fallback learning.

## L

**Lattice**  
The 4D grid structure in which neurons are positioned.

**Learning Rate**  
A parameter controlling how quickly synaptic weights change during plasticity.

**Leaky Integrate-and-Fire (LIF)**  
A simplified neuron model where membrane potential integrates inputs and leaks toward rest potential.

**LIF Model**  
See *Leaky Integrate-and-Fire*.

## M

**Membrane Potential**  
The electrical voltage across a neuron's membrane. When it reaches threshold, the neuron spikes.

**Modality**  
A type of sensory input (vision, audition, etc.).

**Mutation**  
Random changes to neuron parameters when a new neuron is created through reproduction.

## N

**Neuron**  
The basic computational unit of the neural network, modeled after biological neurons.

**Neuron ID**  
A unique integer identifier for each neuron.

**NumPy**  
Python library for numerical computing, used for array operations.

## O

**Observation**  
In the tasks framework, the sensory input provided to the network by an environment.

## P

**Plasticity**  
The ability of synapses to change strength based on activity (learning).

**Plasticity Tag**  
A value stored with each synapse for use by learning rules.

**Post-synaptic**  
The neuron receiving a signal across a synapse.

**Pre-synaptic**  
The neuron sending a signal across a synapse.

**Pre-training**  
Training a network on knowledge database examples before running tasks.

## R

**Reaction Time**  
The number of simulation steps from stimulus presentation to first output spike.

**Receptive Field**  
The region of sensory input that affects a particular neuron. Currently mapped directly, not learned.

**Refractory Period**  
A brief time after spiking when a neuron cannot spike again (simulates biological refractory period).

**Reproduction**  
The process by which active, healthy neurons create offspring neurons with slightly mutated parameters.

**Rest Potential (V_rest)**  
The membrane potential a neuron returns to when no inputs are present.

**Reward**  
In the tasks framework, a signal indicating how well the network performed on a step.

## S

**Seed (Random)**  
An integer used to initialize random number generators for reproducibility.

**Sense**  
A sensory modality (vision, audition, taste, smell, somatosensory, vestibular, digital).

**Sensory Input**  
External data fed to the neural network through one of the senses.

**Simulation**  
The class that orchestrates the execution of simulation steps.

**Socket.IO**  
A library for real-time bidirectional communication between web clients and servers.

**Soma**  
The cell body of a biological neuron. In our simulation, properties are part of the Neuron dataclass.

**Spike**  
See *Action Potential*.

**STDP (Spike-Timing-Dependent Plasticity)**  
An advanced learning rule where synapse changes depend on precise spike timing. Planned but not yet implemented.

**Synapse**  
A connection between two neurons that transmits spikes with a certain weight and delay.

**Synaptic Weight**  
The strength of a synaptic connection, determining how much effect a pre-synaptic spike has.

## T

**Task**  
In the evaluation framework, a standardized test that measures network performance.

**Tau (τ)**  
Time constant in the LIF model, determining how quickly membrane potential decays.

**Threshold (V_threshold)**  
The membrane potential level at which a neuron fires a spike.

## V

**Vestibular**  
Sensory modality related to balance, spatial orientation, and movement.

**Visualization**  
Graphical representation of neural activity, typically as heatmaps or plots.

**V_membrane**  
See *Membrane Potential*.

**V_reset**  
The membrane potential a neuron is set to immediately after spiking.

**V_rest**  
See *Rest Potential*.

**V_threshold**  
See *Threshold*.

## W

**W-dimension**  
The fourth spatial dimension in our 4D lattice. Flexible in interpretation (modality, time, hierarchy, etc.).

**Weight**  
See *Synaptic Weight*.

**Weight Clipping**  
Limiting synaptic weights to a minimum and maximum value to prevent overflow or negative weights.

**Weight Decay**  
Gradual reduction of synaptic weights over time to prevent runaway potentiation.

**WebSocket**  
A protocol for real-time bidirectional communication between browser and server.

## X

**X-coordinate**  
The first spatial dimension in the lattice.

## Y

**Y-coordinate**  
The second spatial dimension in the lattice.

## Z

**Z-coordinate**  
The third spatial dimension in the lattice.

---

## Acronyms and Abbreviations

- **API**: Application Programming Interface
- **BCM**: Bienenstock-Cooper-Munro (plasticity rule, not yet implemented)
- **CLI**: Command-Line Interface
- **CPU**: Central Processing Unit
- **CSV**: Comma-Separated Values
- **GABA**: Gamma-Aminobutyric Acid (inhibitory neurotransmitter)
- **GIL**: Global Interpreter Lock
- **GPU**: Graphics Processing Unit
- **GUI**: Graphical User Interface
- **HDF5**: Hierarchical Data Format version 5
- **HTTP**: Hypertext Transfer Protocol
- **IDE**: Integrated Development Environment
- **IEEE**: Institute of Electrical and Electronics Engineers
- **ISO**: International Organization for Standardization
- **JSON**: JavaScript Object Notation
- **LIF**: Leaky Integrate-and-Fire
- **PSTH**: Peri-Stimulus Time Histogram
- **REST**: Representational State Transfer
- **STDP**: Spike-Timing-Dependent Plasticity
- **UI**: User Interface
- **URL**: Uniform Resource Locator
- **WSL**: Windows Subsystem for Linux

---

## Mathematical Symbols

- **τ (tau)**: Time constant
- **Δt**: Time step duration
- **η (eta)**: Learning rate
- **ρ (rho)**: Density
- **σ (sigma)**: Standard deviation
- **w**: Synaptic weight
- **x, y, z, w**: 4D coordinates

---

## Related Documentation

- [README.md](../../README.md) - Project overview
- [API.md](../API.md) - API reference
- [ARCHITECTURE.md](../ARCHITECTURE.md) - Technical architecture
- [FAQ.md](FAQ.md) - Frequently Asked Questions

---

*Last Updated: December 2025*  
*Version: 1.0*

*Missing a term? Please suggest additions via [GitHub Issues](https://github.com/Thomas-Heisig/4D-Neural-Cognition/issues) with the `documentation` label.*
