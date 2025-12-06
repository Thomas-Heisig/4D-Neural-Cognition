"""
Glia Cell Types - Supporting Cells for Neural Networks

This module implements various types of glial cells that support and modulate
neuronal function, including astrocytes, oligodendrocytes, and microglia.

Basisklassen und Typen für Gliazellen: Astrozyten, Oligodendrozyten, Mikroglia.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Set
from enum import Enum
import numpy as np


class GliaType(Enum):
    """
    Types of glial cells.
    
    Typen von Gliazellen im Nervensystem.
    """
    ASTROCYTE = "astrocyte"
    OLIGODENDROCYTE = "oligodendrocyte"
    MICROGLIA = "microglia"
    SCHWANN_CELL = "schwann_cell"
    EPENDYMAL = "ependymal"


class GliaState(Enum):
    """
    Activation states of glial cells.
    
    Aktivierungszustände von Gliazellen.
    """
    RESTING = "resting"
    ACTIVE = "active"
    REACTIVE = "reactive"
    PROLIFERATING = "proliferating"


@dataclass
class GliaCell:
    """
    Base class for all glial cells.
    
    Basisklasse für alle Gliazellen mit gemeinsamen Eigenschaften.
    
    Attributes:
        cell_id: Unique identifier
        position_4d: Position in 4D lattice
        glia_type: Type of glial cell
        state: Current activation state
        age: Age in simulation steps
        health: Health status (0.0 to 1.0)
        associated_neurons: Set of neuron IDs this cell supports
        metabolic_rate: Rate of metabolic activity
    """
    cell_id: int
    position_4d: tuple  # (x, y, z, w)
    glia_type: GliaType
    state: GliaState = GliaState.RESTING
    age: int = 0
    health: float = 1.0
    associated_neurons: Set[int] = field(default_factory=set)
    metabolic_rate: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update(self, dt: float = 0.1):
        """
        Update glial cell state.
        
        Args:
            dt: Time step (ms)
        """
        self.age += 1
        # Base aging effect
        self.health = max(0.0, self.health - 0.00001)
    
    def is_alive(self) -> bool:
        """Check if cell is alive."""
        return self.health > 0.0
    
    def associate_with_neuron(self, neuron_id: int):
        """
        Associate this glial cell with a neuron.
        
        Args:
            neuron_id: ID of neuron to associate with
        """
        self.associated_neurons.add(neuron_id)
    
    def dissociate_from_neuron(self, neuron_id: int):
        """
        Remove association with a neuron.
        
        Args:
            neuron_id: ID of neuron to dissociate from
        """
        self.associated_neurons.discard(neuron_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'cell_id': self.cell_id,
            'position_4d': self.position_4d,
            'glia_type': self.glia_type.value,
            'state': self.state.value,
            'age': self.age,
            'health': self.health,
            'num_associated_neurons': len(self.associated_neurons),
            'metabolic_rate': self.metabolic_rate,
        }


@dataclass
class Astrocyte(GliaCell):
    """
    Astrocyte glial cell - supports neurons and regulates synapses.
    
    Astrozyten regulieren die extrazelluläre Umgebung, unterstützen
    Synapsen und modulieren neuronale Aktivität.
    
    Attributes:
        coverage_radius: Spatial range of influence (μm)
        neurotransmitter_uptake: Rate of neurotransmitter clearance
        ion_buffering_capacity: Capacity to buffer extracellular ions
        gliotransmitter_release: Rate of gliotransmitter release
        synapses_modulated: Set of synapse IDs this astrocyte modulates
        calcium_level: Internal calcium concentration
    """
    coverage_radius: float = 50.0  # micrometers
    neurotransmitter_uptake: float = 0.8  # relative rate
    ion_buffering_capacity: float = 100.0  # arbitrary units
    gliotransmitter_release: float = 0.1  # relative rate
    synapses_modulated: Set[tuple] = field(default_factory=set)  # (pre_id, post_id)
    calcium_level: float = 0.1  # μM
    
    def __post_init__(self):
        """Initialize astrocyte-specific properties."""
        self.glia_type = GliaType.ASTROCYTE
        if not self.metadata:
            self.metadata = {
                'glutamate_transporters': 1.0,
                'gap_junction_coupling': 0.5,
                'water_channels': 1.0,
            }
    
    def update(self, dt: float = 0.1):
        """
        Update astrocyte state and perform regulatory functions.
        
        Args:
            dt: Time step (ms)
        """
        super().update(dt)
        
        # Calcium dynamics - simple decay model
        self.calcium_level *= 0.99
        
        # State transitions based on activity
        if self.calcium_level > 0.5:
            self.state = GliaState.ACTIVE
        elif self.calcium_level > 0.3:
            self.state = GliaState.REACTIVE
        else:
            self.state = GliaState.RESTING
    
    def modulate_synapse(self, pre_neuron_id: int, post_neuron_id: int):
        """
        Mark a synapse for modulation by this astrocyte.
        
        Astrozyten können synaptische Übertragung modulieren.
        
        Args:
            pre_neuron_id: Presynaptic neuron ID
            post_neuron_id: Postsynaptic neuron ID
        """
        self.synapses_modulated.add((pre_neuron_id, post_neuron_id))
    
    def uptake_neurotransmitter(self, amount: float) -> float:
        """
        Simulate neurotransmitter uptake from extracellular space.
        
        Args:
            amount: Amount of neurotransmitter present
            
        Returns:
            Amount taken up by astrocyte
        """
        uptake = amount * self.neurotransmitter_uptake * 0.1
        return min(uptake, amount)
    
    def release_gliotransmitter(self) -> float:
        """
        Release gliotransmitters that can modulate neuronal activity.
        
        Returns:
            Amount of gliotransmitter released
        """
        if self.state == GliaState.ACTIVE:
            return self.gliotransmitter_release * self.calcium_level
        return 0.0
    
    def buffer_potassium(self, k_concentration: float) -> float:
        """
        Buffer excess extracellular potassium.
        
        Astrozyten regulieren die Kaliumkonzentration im Extrazellularraum.
        
        Args:
            k_concentration: Current K+ concentration (mM)
            
        Returns:
            Buffered amount
        """
        if k_concentration > 3.5:  # Normal K+ is ~3.5 mM
            excess = k_concentration - 3.5
            buffered = min(excess, self.ion_buffering_capacity * 0.01)
            return buffered
        return 0.0


@dataclass
class Oligodendrocyte(GliaCell):
    """
    Oligodendrocyte - provides myelination for axons.
    
    Oligodendrozyten bilden Myelinscheiden um Axone zur Beschleunigung
    der Signalweiterleitung.
    
    Attributes:
        max_axons_myelinated: Maximum number of axons this cell can myelinate
        myelination_rate: Rate of myelin production
        myelin_thickness: Thickness of myelin sheath (μm)
        axons_myelinated: Set of axon IDs currently myelinated
        myelin_maintenance_cost: Metabolic cost of maintaining myelin
    """
    max_axons_myelinated: int = 40  # One oligodendrocyte can myelinate many axons
    myelination_rate: float = 0.1  # relative rate of myelin formation
    myelin_thickness: float = 0.5  # micrometers
    axons_myelinated: Set[int] = field(default_factory=set)
    myelin_maintenance_cost: float = 0.02
    
    def __post_init__(self):
        """Initialize oligodendrocyte-specific properties."""
        self.glia_type = GliaType.OLIGODENDROCYTE
        if not self.metadata:
            self.metadata = {
                'myelin_basic_protein': 1.0,
                'proteolipid_protein': 1.0,
            }
    
    def update(self, dt: float = 0.1):
        """
        Update oligodendrocyte state and maintain myelination.
        
        Args:
            dt: Time step (ms)
        """
        super().update(dt)
        
        # Metabolic cost of maintaining myelin
        maintenance_cost = len(self.axons_myelinated) * self.myelin_maintenance_cost
        self.metabolic_rate = 1.0 + maintenance_cost
    
    def can_myelinate_more(self) -> bool:
        """
        Check if oligodendrocyte can myelinate additional axons.
        
        Returns:
            True if capacity available, False otherwise
        """
        return len(self.axons_myelinated) < self.max_axons_myelinated
    
    def myelinate_axon(self, axon_id: int) -> bool:
        """
        Myelinate an axon.
        
        Fügt einem Axon eine Myelinscheide hinzu zur Beschleunigung der Leitung.
        
        Args:
            axon_id: ID of axon to myelinate
            
        Returns:
            True if successful, False if capacity reached
        """
        if self.can_myelinate_more():
            self.axons_myelinated.add(axon_id)
            return True
        return False
    
    def unmyelinate_axon(self, axon_id: int):
        """
        Remove myelination from an axon.
        
        Args:
            axon_id: ID of axon to unmyelinate
        """
        self.axons_myelinated.discard(axon_id)
    
    def get_myelination_boost(self) -> float:
        """
        Calculate conduction velocity boost from myelination.
        
        Returns:
            Multiplicative factor for conduction velocity
        """
        # Thicker myelin provides greater boost
        return 1.0 + (self.myelin_thickness * 2.0)


@dataclass
class Microglia(GliaCell):
    """
    Microglia - immune cells of the brain.
    
    Mikroglia sind die Immunzellen des Gehirns, die Schäden erkennen,
    tote Zellen entfernen und Entzündungsreaktionen vermitteln.
    
    Attributes:
        surveillance_radius: Range for monitoring neural health (μm)
        phagocytic_capacity: Capacity to clear debris
        activation_threshold: Threshold for activation
        cytokine_production: Rate of inflammatory cytokine production
        monitored_cells: Set of cell IDs being monitored
    """
    surveillance_radius: float = 70.0  # micrometers
    phagocytic_capacity: float = 1.0
    activation_threshold: float = 0.5
    cytokine_production: float = 0.0
    monitored_cells: Set[int] = field(default_factory=set)
    
    def __post_init__(self):
        """Initialize microglia-specific properties."""
        self.glia_type = GliaType.MICROGLIA
        if not self.metadata:
            self.metadata = {
                'ramification_index': 1.0,  # How branched the processes are
                'motility': 0.5,  # Movement capability
            }
    
    def update(self, dt: float = 0.1):
        """
        Update microglia state and perform surveillance.
        
        Args:
            dt: Time step (ms)
        """
        super().update(dt)
        
        # Cytokine production decays over time
        self.cytokine_production *= 0.95
        
        # State transitions
        if self.cytokine_production > 0.7:
            self.state = GliaState.REACTIVE
        elif self.cytokine_production > 0.3:
            self.state = GliaState.ACTIVE
        else:
            self.state = GliaState.RESTING
    
    def detect_damage(self, cell_health: float) -> bool:
        """
        Detect if a cell is damaged and needs attention.
        
        Args:
            cell_health: Health value of monitored cell
            
        Returns:
            True if damage detected, False otherwise
        """
        return cell_health < self.activation_threshold
    
    def activate_immune_response(self, threat_level: float):
        """
        Activate immune response to neural damage or pathogens.
        
        Aktiviert Immunantwort bei neuronalen Schäden.
        
        Args:
            threat_level: Severity of threat (0.0 to 1.0)
        """
        self.state = GliaState.REACTIVE
        self.cytokine_production = min(1.0, self.cytokine_production + threat_level)
    
    def phagocytose_debris(self, debris_amount: float) -> float:
        """
        Remove cellular debris through phagocytosis.
        
        Args:
            debris_amount: Amount of debris present
            
        Returns:
            Amount removed
        """
        if self.state in [GliaState.ACTIVE, GliaState.REACTIVE]:
            removed = min(debris_amount, self.phagocytic_capacity * 0.1)
            return removed
        return 0.0
    
    def monitor_cell(self, cell_id: int):
        """
        Add a cell to surveillance list.
        
        Args:
            cell_id: ID of cell to monitor
        """
        self.monitored_cells.add(cell_id)
    
    def stop_monitoring_cell(self, cell_id: int):
        """
        Remove a cell from surveillance list.
        
        Args:
            cell_id: ID of cell to stop monitoring
        """
        self.monitored_cells.discard(cell_id)
