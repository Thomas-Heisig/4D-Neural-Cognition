"""Developmental processes for 4D Neural Cognition.

This module implements neural development:
- Neurogenesis (proliferation, differentiation)
- Cell migration (radial, tangential)
- Circuit assembly (axon guidance, synapse specification)
- Critical periods for experience-dependent refinement
- Developmental apoptosis

References:
- Rakic, P. (2009). Evolution of the neocortex
- Marín, O., & Rubenstein, J.L. (2003). Cell migration in the forebrain
- Jessell, T.M., & Sanes, J.R. (2000). Development of the vertebrate neuromuscular junction
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import numpy as np


class CellFate(Enum):
    """Cell fate determination."""
    NEURAL_STEM = "neural_stem"
    PROGENITOR = "progenitor"
    NEURON = "neuron"
    ASTROCYTE = "astrocyte"
    OLIGODENDROCYTE = "oligodendrocyte"
    APOPTOTIC = "apoptotic"


class MigrationMode(Enum):
    """Cell migration modes."""
    RADIAL = "radial"  # Glial-guided migration
    TANGENTIAL = "tangential"  # Interneuron migration
    SOMAL_TRANSLOCATION = "somal_translocation"  # Direct movement


@dataclass
class NeuralStemCell:
    """Neural stem cell for neurogenesis.
    
    Neural stem cells can:
    - Self-renew (symmetric division)
    - Produce progenitors (asymmetric division)
    - Produce neurons directly
    - Produce glia
    """
    
    id: int
    position: Tuple[int, int, int, int]
    
    # Proliferation
    division_probability: float = 0.1
    symmetric_division_probability: float = 0.3
    
    # Fate determination
    neurogenic_potential: float = 0.8
    gliogenic_potential: float = 0.2
    
    # Age
    divisions_remaining: int = 20
    
    def can_divide(self) -> bool:
        """Check if stem cell can divide.
        
        Returns:
            True if division is possible
        """
        return self.divisions_remaining > 0
    
    def divide(self) -> Tuple[CellFate, Optional[CellFate]]:
        """Perform cell division.
        
        Returns:
            Tuple of (cell1_fate, cell2_fate)
            If symmetric: both are stem cells or both differentiate
            If asymmetric: one stem, one progenitor
        """
        if not self.can_divide():
            return CellFate.APOPTOTIC, None
        
        self.divisions_remaining -= 1
        
        # Symmetric vs asymmetric
        if np.random.random() < self.symmetric_division_probability:
            # Symmetric division
            if np.random.random() < 0.5:
                # Both remain stem cells
                return CellFate.NEURAL_STEM, CellFate.NEURAL_STEM
            else:
                # Both differentiate
                fate = self.determine_fate()
                return fate, fate
        else:
            # Asymmetric division (typical)
            # One remains stem, one becomes progenitor
            return CellFate.NEURAL_STEM, CellFate.PROGENITOR
    
    def determine_fate(self) -> CellFate:
        """Determine differentiation fate.
        
        Returns:
            Cell fate
        """
        if np.random.random() < self.neurogenic_potential:
            return CellFate.PROGENITOR  # Will become neuron
        else:
            # Gliogenesis occurs later in development
            if np.random.random() < 0.5:
                return CellFate.ASTROCYTE
            else:
                return CellFate.OLIGODENDROCYTE


@dataclass
class NeuralProgenitor:
    """Neural progenitor cell (intermediate progenitor).
    
    More committed than stem cells:
    - Limited divisions (1-3)
    - Neuronal fate determined
    - Undergoes terminal differentiation
    """
    
    id: int
    position: Tuple[int, int, int, int]
    
    # Differentiation
    divisions_remaining: int = 2
    neuron_type: str = "excitatory"  # Determined by location/signals
    
    # Markers
    neuronal_markers: List[str] = field(default_factory=lambda: ["Pax6", "Tbr2"])
    
    def can_divide(self) -> bool:
        """Check if progenitor can divide.
        
        Returns:
            True if division is possible
        """
        return self.divisions_remaining > 0
    
    def divide(self) -> Tuple[CellFate, CellFate]:
        """Divide into two neurons.
        
        Returns:
            Tuple of (neuron, neuron) or (progenitor, progenitor)
        """
        self.divisions_remaining -= 1
        
        if self.divisions_remaining == 0:
            # Terminal division -> neurons
            return CellFate.NEURON, CellFate.NEURON
        else:
            # Amplifying division -> more progenitors
            return CellFate.PROGENITOR, CellFate.PROGENITOR


@dataclass
class MigratingNeuron:
    """Neuron undergoing migration to final position.
    
    Migration modes:
    - Radial: Along radial glia (excitatory neurons)
    - Tangential: Perpendicular to radial (interneurons)
    """
    
    id: int
    current_position: np.ndarray
    target_layer: int
    
    # Migration parameters
    migration_mode: MigrationMode = MigrationMode.RADIAL
    migration_speed: float = 10.0  # μm/hour in reality, here per step
    
    # Guidance
    following_glia_id: Optional[int] = None
    
    # Completion
    has_reached_target: bool = False
    
    def migrate_step(
        self,
        target_position: Optional[np.ndarray] = None,
        guidance_signals: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """Perform one migration step.
        
        Args:
            target_position: Optional target position
            guidance_signals: Optional guidance cues
            
        Returns:
            New position
        """
        if self.has_reached_target:
            return self.current_position
        
        # Determine migration direction
        if self.migration_mode == MigrationMode.RADIAL:
            # Migrate toward target layer (typically upward)
            if target_position is not None:
                direction = target_position - self.current_position
                distance = np.linalg.norm(direction)
                
                if distance < self.migration_speed:
                    self.has_reached_target = True
                    return target_position
                
                # Move along radial direction
                direction = direction / distance
                self.current_position = self.current_position + direction * self.migration_speed
        
        elif self.migration_mode == MigrationMode.TANGENTIAL:
            # Tangential migration (interneurons from ganglionic eminence)
            # Follow chemical gradients
            if guidance_signals and "tangential_attractor" in guidance_signals:
                direction = guidance_signals["tangential_attractor"]
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                    self.current_position = self.current_position + direction * self.migration_speed * 0.5
        
        return self.current_position


@dataclass
class CircuitAssembly:
    """Circuit assembly during development.
    
    Includes:
    - Axon pathfinding
    - Target recognition
    - Synapse formation
    - Activity-dependent refinement
    """
    
    # Guidance molecules (gradients)
    netrin_gradient: Dict[Tuple, float] = field(default_factory=dict)
    slit_gradient: Dict[Tuple, float] = field(default_factory=dict)
    semaphorin_gradient: Dict[Tuple, float] = field(default_factory=dict)
    
    # Target-derived factors
    ngf_sources: List[Tuple] = field(default_factory=list)  # Nerve growth factor
    bdnf_sources: List[Tuple] = field(default_factory=list)  # Brain-derived neurotrophic factor
    
    def calculate_guidance(
        self,
        position: np.ndarray,
        target_type: str
    ) -> np.ndarray:
        """Calculate guidance vector for axon.
        
        Args:
            position: Current axon position
            target_type: Type of target being sought
            
        Returns:
            Guidance direction vector
        """
        guidance = np.zeros_like(position, dtype=float)
        
        # Netrin: chemoattractant for commissural axons
        netrin_vec = self._gradient_vector(position, self.netrin_gradient)
        
        # Slit: chemorepellent, prevents midline recrossing
        slit_vec = -self._gradient_vector(position, self.slit_gradient)
        
        # Semaphorin: area-specific repellent
        sema_vec = -self._gradient_vector(position, self.semaphorin_gradient)
        
        guidance = netrin_vec + slit_vec * 0.5 + sema_vec * 0.3
        
        # Target-derived neurotrophic factors
        if target_type == "cortical":
            for bdnf_source in self.bdnf_sources:
                direction = np.array(bdnf_source) - position
                dist = np.linalg.norm(direction)
                if dist > 0:
                    guidance += direction / (dist ** 2 + 1.0)
        
        # Normalize
        norm = np.linalg.norm(guidance)
        if norm > 0:
            guidance = guidance / norm
        
        return guidance
    
    def _gradient_vector(
        self,
        position: np.ndarray,
        gradient: Dict[Tuple, float]
    ) -> np.ndarray:
        """Calculate gradient vector at position.
        
        Args:
            position: Position to evaluate
            gradient: Gradient dictionary
            
        Returns:
            Gradient direction
        """
        # Simple finite difference
        pos_tuple = tuple(position.astype(int))
        
        if not gradient:
            return np.zeros_like(position, dtype=float)
        
        grad_vec = np.zeros_like(position, dtype=float)
        
        # Sample nearby points
        for dim in range(len(position)):
            pos_plus = list(pos_tuple)
            pos_minus = list(pos_tuple)
            pos_plus[dim] += 1
            pos_minus[dim] -= 1
            
            val_plus = gradient.get(tuple(pos_plus), 0.0)
            val_minus = gradient.get(tuple(pos_minus), 0.0)
            
            grad_vec[dim] = val_plus - val_minus
        
        return grad_vec


@dataclass
class DevelopmentalApoptosis:
    """Programmed cell death during development.
    
    About 50% of neurons die during development:
    - Competition for neurotrophic factors
    - Elimination of targeting errors
    - Matching population sizes
    """
    
    # Survival factors
    neurotrophic_factor_availability: float = 0.5
    
    # Competition
    synaptic_competition_threshold: float = 0.3
    
    # Timing
    apoptosis_window_start: int = 5000
    apoptosis_window_end: int = 15000
    
    def should_survive(
        self,
        neuron_age: int,
        synaptic_connections: int,
        target_connections: int,
        current_time: int
    ) -> bool:
        """Determine if neuron should survive.
        
        Args:
            neuron_age: Age of neuron
            synaptic_connections: Number of connections made
            target_connections: Expected number of connections
            current_time: Current developmental time
            
        Returns:
            True if neuron survives
        """
        # Outside apoptosis window
        if current_time < self.apoptosis_window_start or current_time > self.apoptosis_window_end:
            return True
        
        # Immature neurons always survive
        if neuron_age < 100:
            return True
        
        # Survival depends on synaptic competition
        connection_ratio = synaptic_connections / max(target_connections, 1)
        
        # Neurons with few connections are eliminated
        if connection_ratio < self.synaptic_competition_threshold:
            # Probabilistic death
            death_probability = 1.0 - connection_ratio / self.synaptic_competition_threshold
            return np.random.random() > death_probability
        
        # Well-connected neurons survive
        return True


@dataclass
class DevelopmentalManager:
    """Manager for developmental processes."""
    
    # Current developmental stage
    developmental_time: int = 0
    
    # Cell populations
    stem_cells: Dict[int, NeuralStemCell] = field(default_factory=dict)
    progenitors: Dict[int, NeuralProgenitor] = field(default_factory=dict)
    migrating_neurons: Dict[int, MigratingNeuron] = field(default_factory=dict)
    
    # Processes
    circuit_assembly: CircuitAssembly = field(default_factory=CircuitAssembly)
    apoptosis: DevelopmentalApoptosis = field(default_factory=DevelopmentalApoptosis)
    
    # Developmental phases
    neurogenesis_peak: int = 5000
    neurogenesis_end: int = 15000
    gliogenesis_start: int = 10000
    gliogenesis_peak: int = 20000
    
    # Statistics
    neurons_generated: int = 0
    neurons_died: int = 0
    glia_generated: int = 0
    
    def is_neurogenic_phase(self) -> bool:
        """Check if in neurogenic phase.
        
        Returns:
            True if neurogenesis is active
        """
        return self.developmental_time < self.neurogenesis_end
    
    def is_gliogenic_phase(self) -> bool:
        """Check if in gliogenic phase.
        
        Returns:
            True if gliogenesis is active
        """
        return self.developmental_time >= self.gliogenesis_start
    
    def neurogenesis_rate(self) -> float:
        """Get current neurogenesis rate.
        
        Returns:
            Rate multiplier (0-1)
        """
        if self.developmental_time < self.neurogenesis_peak:
            # Ramp up
            return self.developmental_time / self.neurogenesis_peak
        elif self.developmental_time < self.neurogenesis_end:
            # Decline
            progress = (self.developmental_time - self.neurogenesis_peak) / (self.neurogenesis_end - self.neurogenesis_peak)
            return 1.0 - progress
        else:
            # Adult neurogenesis (limited)
            return 0.01
    
    def step(self, dt: int = 1) -> Dict[str, int]:
        """Update developmental processes.
        
        Args:
            dt: Time step
            
        Returns:
            Statistics dictionary
        """
        self.developmental_time += dt
        
        # Neurogenesis
        if self.is_neurogenic_phase():
            rate = self.neurogenesis_rate()
            # Simulate some stem cell divisions
            for stem_id, stem_cell in list(self.stem_cells.items()):
                if np.random.random() < stem_cell.division_probability * rate:
                    fate1, fate2 = stem_cell.divide()
                    
                    if fate1 == CellFate.NEURON or fate1 == CellFate.PROGENITOR:
                        self.neurons_generated += 1
        
        # Migration
        for neuron_id, migrating in list(self.migrating_neurons.items()):
            if not migrating.has_reached_target:
                migrating.migrate_step()
        
        # Apoptosis
        # (Would need neuron reference to actually remove)
        
        return {
            "neurons_generated": self.neurons_generated,
            "neurons_died": self.neurons_died,
            "glia_generated": self.glia_generated,
            "developmental_time": self.developmental_time,
        }


def create_cortical_development_timeline() -> Dict[str, Tuple[int, int]]:
    """Create timeline of cortical development.
    
    Returns:
        Dictionary mapping phase to (start_time, end_time)
    """
    return {
        "proliferation": (0, 5000),
        "neurogenesis_peak": (2000, 8000),
        "migration": (3000, 12000),
        "layer_formation": (5000, 15000),
        "axon_pathfinding": (6000, 20000),
        "synaptogenesis": (10000, 30000),
        "refinement": (15000, 50000),
        "gliogenesis": (10000, 40000),
        "myelination": (20000, 100000),
    }
