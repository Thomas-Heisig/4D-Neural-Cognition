"""
Neuron Base Structure - Complete Neuron Components

This module provides the fundamental building blocks for neurons in the
4D neural cognition system, including soma, dendrites, and axons with
biological-inspired properties.

Basisstruktur für Neuronen mit vollständigen Unter-Komponenten.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import numpy as np


class NeuronType(Enum):
    """
    Enumeration of neuron types based on biological classification.
    
    Typen von Neuronen nach biologischer Klassifikation.
    """
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    MODULATORY = "modulatory"
    SENSORY = "sensory"
    MOTOR = "motor"
    INTERNEURON = "interneuron"


class CompartmentType(Enum):
    """
    Types of neuron compartments.
    
    Typen von Neuron-Kompartimenten.
    """
    SOMA = "soma"
    DENDRITE = "dendrite"
    AXON = "axon"
    SPINE = "spine"


@dataclass
class Soma:
    """
    Represents the soma (cell body) of a neuron.
    
    Contains the nucleus and most cellular machinery for protein synthesis.
    Soma integriert Signale von Dendriten und erzeugt Aktionspotentiale.
    
    Attributes:
        membrane_potential: Current membrane voltage (mV)
        threshold: Spike threshold voltage (mV)
        resting_potential: Resting membrane potential (mV)
        capacitance: Membrane capacitance (pF)
        diameter: Soma diameter (μm)
        ion_channels: Dictionary of ion channel densities
    """
    membrane_potential: float = -65.0  # mV
    threshold: float = -50.0  # mV
    resting_potential: float = -65.0  # mV
    reset_potential: float = -70.0  # mV
    capacitance: float = 100.0  # pF
    diameter: float = 20.0  # micrometers
    ion_channels: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default ion channels if not provided."""
        if not self.ion_channels:
            self.ion_channels = {
                'Na': 120.0,  # Sodium conductance (mS/cm²)
                'K': 36.0,    # Potassium conductance (mS/cm²)
                'Ca': 0.3,    # Calcium conductance (mS/cm²)
                'leak': 0.3,  # Leak conductance (mS/cm²)
            }
    
    def integrate_current(self, current: float, dt: float = 0.1) -> bool:
        """
        Integrate incoming current and update membrane potential.
        
        Args:
            current: Input current (pA)
            dt: Time step (ms)
            
        Returns:
            True if spike threshold is reached, False otherwise
        """
        # Leaky integrate-and-fire dynamics
        # Ensure leak conductance has minimum value to prevent division by zero
        leak_conductance = max(self.ion_channels['leak'], 0.01)
        tau_m = self.capacitance / leak_conductance
        dv = (-(self.membrane_potential - self.resting_potential) + current) / tau_m
        self.membrane_potential += dv * dt
        
        # Check for spike
        if self.membrane_potential >= self.threshold:
            self.membrane_potential = self.reset_potential
            return True
        return False


@dataclass
class Dendrite:
    """
    Represents a dendritic branch for receiving signals.
    
    Dendrites empfangen synaptische Eingänge und leiten diese zum Soma.
    Sie können lokale Verarbeitung durchführen und Plastizität zeigen.
    
    Attributes:
        length: Length of dendrite (μm)
        diameter: Average diameter (μm)
        branch_order: Order in dendritic tree (0 = primary)
        synapses: List of synapse locations and weights
        spine_density: Number of spines per μm
        active_conductances: Active dendritic conductances
    """
    length: float = 100.0  # micrometers
    diameter: float = 2.0  # micrometers
    branch_order: int = 0
    synapses: List[Dict[str, Any]] = field(default_factory=list)
    spine_density: float = 5.0  # spines per micrometer
    active_conductances: Dict[str, float] = field(default_factory=dict)
    local_potential: float = -65.0  # mV
    
    def __post_init__(self):
        """Initialize active conductances if not provided."""
        if not self.active_conductances:
            self.active_conductances = {
                'NMDA': 0.1,  # NMDA receptor conductance
                'AMPA': 0.5,  # AMPA receptor conductance
                'GABA': 0.3,  # GABA receptor conductance
            }
    
    def add_synapse(self, pre_neuron_id: int, weight: float, 
                    location: float, synapse_type: str = 'excitatory'):
        """
        Add a synaptic connection to this dendrite.
        
        Args:
            pre_neuron_id: ID of presynaptic neuron
            weight: Synaptic weight
            location: Position along dendrite (0.0 to 1.0)
            synapse_type: Type of synapse ('excitatory' or 'inhibitory')
        """
        synapse = {
            'pre_neuron_id': pre_neuron_id,
            'weight': weight,
            'location': location,
            'type': synapse_type,
            'plasticity_tag': 0.0,
        }
        self.synapses.append(synapse)
    
    def compute_dendritic_current(self) -> float:
        """
        Compute total current contribution from all synapses.
        
        Returns:
            Total synaptic current (pA)
        """
        total_current = 0.0
        for synapse in self.synapses:
            # Simple model: current proportional to weight
            # In real implementation, would include distance attenuation
            current = synapse['weight'] * 10.0  # Convert weight to pA
            if synapse['type'] == 'inhibitory':
                current = -current
            total_current += current
        return total_current


@dataclass
class Axon:
    """
    Represents the axon for signal transmission.
    
    Axone leiten Aktionspotentiale vom Soma zu den Synapsen.
    Sie können myelinisiert sein für schnellere Weiterleitung.
    
    Attributes:
        length: Length of axon (μm)
        diameter: Axon diameter (μm)
        conduction_velocity: Signal propagation speed (m/s)
        myelination: Degree of myelination (0.0 to 1.0)
        branches: Number of axonal branches
        terminal_synapses: List of output synapses
    """
    length: float = 500.0  # micrometers
    diameter: float = 1.0  # micrometers
    conduction_velocity: float = 1.0  # m/s
    myelination: float = 0.0  # 0.0 = unmyelinated, 1.0 = fully myelinated
    branches: int = 5
    terminal_synapses: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate conduction velocity based on myelination and diameter."""
        # Simple model: velocity increases with diameter and myelination
        base_velocity = 0.5 + self.diameter * 0.2  # m/s
        self.conduction_velocity = base_velocity * (1.0 + 5.0 * self.myelination)
    
    def add_terminal_synapse(self, post_neuron_id: int, weight: float,
                            synapse_type: str = 'excitatory'):
        """
        Add an output synapse to this axon.
        
        Args:
            post_neuron_id: ID of postsynaptic neuron
            weight: Synaptic weight
            synapse_type: Type of synapse
        """
        synapse = {
            'post_neuron_id': post_neuron_id,
            'weight': weight,
            'type': synapse_type,
            'delay': self.calculate_delay(),
        }
        self.terminal_synapses.append(synapse)
    
    def calculate_delay(self) -> float:
        """
        Calculate signal propagation delay.
        
        Returns:
            Delay in milliseconds
        """
        # Convert length from μm to m and calculate time
        length_m = self.length * 1e-6
        delay_s = length_m / self.conduction_velocity
        return delay_s * 1000.0  # Convert to ms


@dataclass
class NeuronBase:
    """
    Complete neuron base structure integrating all components.
    
    Vollständige Basisstruktur für Neuronen, die alle Komponenten integriert.
    Diese Klasse bildet die Grundlage für spezialisierte Neuronentypen.
    
    Attributes:
        neuron_id: Unique neuron identifier
        position_4d: Position in 4D lattice (x, y, z, w)
        neuron_type: Type classification of neuron
        soma: Soma component
        dendrites: List of dendritic branches
        axon: Axon component
        age: Age of neuron (simulation steps)
        health: Health status (0.0 to 1.0)
        generation: Generation number for inheritance tracking
        parent_id: ID of parent neuron (for reproduction)
        genetic_parameters: Reference to DNA bank parameters
    """
    neuron_id: int
    position_4d: Tuple[float, float, float, float]  # (x, y, z, w)
    neuron_type: NeuronType = NeuronType.EXCITATORY
    soma: Soma = field(default_factory=Soma)
    dendrites: List[Dendrite] = field(default_factory=list)
    axon: Axon = field(default_factory=Axon)
    age: int = 0
    health: float = 1.0
    generation: int = 0
    parent_id: int = -1
    genetic_parameters: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize neuron with default dendrites if none provided."""
        if not self.dendrites:
            # Create default dendritic tree
            # Primary dendrites (basal)
            for i in range(5):
                self.dendrites.append(Dendrite(branch_order=0, length=150.0))
            # Apical dendrite
            self.dendrites.append(Dendrite(branch_order=0, length=300.0, diameter=3.0))
    
    def update(self, dt: float = 0.1) -> bool:
        """
        Update neuron state for one time step.
        
        Args:
            dt: Time step (ms)
            
        Returns:
            True if neuron spikes, False otherwise
        """
        # Collect dendritic currents
        total_current = 0.0
        for dendrite in self.dendrites:
            total_current += dendrite.compute_dendritic_current()
        
        # Integrate current in soma
        spike = self.soma.integrate_current(total_current, dt)
        
        # Age neuron
        self.age += 1
        
        return spike
    
    def get_membrane_potential(self) -> float:
        """
        Get current membrane potential.
        
        Returns:
            Membrane potential in mV
        """
        return self.soma.membrane_potential
    
    def add_dendritic_synapse(self, pre_neuron_id: int, weight: float,
                             dendrite_index: int = 0, location: float = 0.5,
                             synapse_type: str = 'excitatory'):
        """
        Add a synapse to a specific dendrite.
        
        Args:
            pre_neuron_id: Presynaptic neuron ID
            weight: Synaptic weight
            dendrite_index: Index of target dendrite
            location: Position along dendrite (0.0 to 1.0)
            synapse_type: Type of synapse
        """
        if 0 <= dendrite_index < len(self.dendrites):
            self.dendrites[dendrite_index].add_synapse(
                pre_neuron_id, weight, location, synapse_type
            )
    
    def add_axonal_synapse(self, post_neuron_id: int, weight: float,
                          synapse_type: str = 'excitatory'):
        """
        Add an output synapse from this neuron's axon.
        
        Args:
            post_neuron_id: Postsynaptic neuron ID
            weight: Synaptic weight
            synapse_type: Type of synapse
        """
        self.axon.add_terminal_synapse(post_neuron_id, weight, synapse_type)
    
    def apply_health_decay(self, decay_rate: float = 0.0001):
        """
        Apply aging-related health decay.
        
        Args:
            decay_rate: Rate of health decay per time step
        """
        self.health = max(0.0, self.health - decay_rate)
    
    def is_alive(self) -> bool:
        """
        Check if neuron is alive.
        
        Returns:
            True if health > 0, False otherwise
        """
        return self.health > 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert neuron to dictionary representation.
        
        Returns:
            Dictionary with neuron data
        """
        return {
            'neuron_id': self.neuron_id,
            'position_4d': self.position_4d,
            'neuron_type': self.neuron_type.value,
            'age': self.age,
            'health': self.health,
            'generation': self.generation,
            'parent_id': self.parent_id,
            'membrane_potential': self.soma.membrane_potential,
            'num_dendrites': len(self.dendrites),
            'num_synapses': sum(len(d.synapses) for d in self.dendrites),
        }
