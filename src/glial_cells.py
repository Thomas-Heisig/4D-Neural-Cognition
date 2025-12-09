"""Glial cell implementations for 4D Neural Cognition.

This module implements non-neuronal cells that are critical for neural function:
- Astrocytes: Tripartite synapse, neurotransmitter uptake, metabolic support
- Microglia: Immune function, synaptic pruning
- Oligodendrocytes: Myelination and saltatory conduction
- NG2 Glia (OPCs): Remyelination and modulation

References:
- Verkhratsky, A., & Nedergaard, M. (2018). Physiology of Astroglia
- Salter, M.W., & Stevens, B. (2017). Microglia emerge as central players
- Nave, K.A. (2010). Myelination and support of axonal integrity
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional
import numpy as np

if TYPE_CHECKING:
    try:
        from .brain_model import Neuron, Synapse
    except ImportError:
        from brain_model import Neuron, Synapse


@dataclass
class Astrocyte:
    """Astrocyte glial cell for tripartite synapse and metabolic support.
    
    Astrocytes play critical roles in:
    - Neurotransmitter uptake (glutamate, GABA)
    - Ion homeostasis (K+ buffering)
    - Metabolic support (glucose-lactate shuttle)
    - Synaptic modulation via gliotransmitters
    - Synaptic pruning during development
    """
    
    id: int
    x: int
    y: int
    z: int
    w: int
    
    # Neurotransmitter levels (absorbed from synapses)
    glutamate_level: float = 0.0
    gaba_level: float = 0.0
    
    # Calcium signaling (for gliotransmitter release)
    calcium_level: float = 0.0
    calcium_threshold: float = 0.5
    
    # Metabolic support
    lactate_production: float = 1.0
    glucose_uptake: float = 1.0
    
    # Potassium buffering
    k_buffering_capacity: float = 1.0
    
    # Gliotransmitter release parameters
    gliotransmitter_release_rate: float = 0.1
    
    # Associated synapses (tripartite synapse)
    associated_synapse_ids: List[int] = field(default_factory=list)
    
    # Age and health
    age: int = 0
    health: float = 1.0
    
    def position(self) -> tuple:
        """Return the 4D position tuple."""
        return (self.x, self.y, self.z, self.w)
    
    def uptake_neurotransmitter(
        self,
        neurotransmitter_type: str,
        amount: float,
        uptake_rate: float = 0.8
    ) -> float:
        """Uptake neurotransmitter from synaptic cleft.
        
        Args:
            neurotransmitter_type: Type of neurotransmitter ('glutamate' or 'gaba')
            amount: Amount available
            uptake_rate: Fraction to uptake (0-1)
            
        Returns:
            Amount remaining in cleft
        """
        uptake = amount * uptake_rate
        
        if neurotransmitter_type == "glutamate":
            self.glutamate_level += uptake
            # Glutamate uptake triggers calcium signaling
            self.calcium_level += uptake * 0.1
        elif neurotransmitter_type == "gaba":
            self.gaba_level += uptake
        
        return amount - uptake
    
    def buffer_potassium(self, k_concentration: float) -> float:
        """Buffer excess potassium to maintain homeostasis.
        
        Args:
            k_concentration: Current K+ concentration
            
        Returns:
            Buffered K+ concentration
        """
        # Normal K+ is ~3-5 mM, excess is > 5 mM
        excess_k = max(0.0, k_concentration - 5.0)
        buffered = excess_k * self.k_buffering_capacity
        
        return k_concentration - buffered
    
    def release_gliotransmitter(self) -> Dict[str, float]:
        """Release gliotransmitters based on calcium level.
        
        Returns:
            Dictionary of gliotransmitter types and amounts
        """
        if self.calcium_level < self.calcium_threshold:
            return {}
        
        # Calcium-triggered gliotransmitter release
        release = {
            "glutamate": self.glutamate_level * self.gliotransmitter_release_rate,
            "d-serine": 0.1 * self.gliotransmitter_release_rate,  # NMDA co-agonist
            "ATP": 0.1 * self.gliotransmitter_release_rate,  # Purinergic signaling
        }
        
        # Deplete internal stores
        self.glutamate_level *= (1.0 - self.gliotransmitter_release_rate)
        
        return release
    
    def metabolic_support(self) -> float:
        """Provide metabolic support via lactate.
        
        Returns:
            Lactate production rate
        """
        return self.lactate_production * self.glucose_uptake
    
    def decay(self, decay_rate: float = 0.1) -> None:
        """Decay internal neurotransmitter and calcium levels.
        
        Args:
            decay_rate: Rate of decay per step
        """
        self.glutamate_level *= (1.0 - decay_rate)
        self.gaba_level *= (1.0 - decay_rate)
        self.calcium_level *= (1.0 - decay_rate * 0.5)  # Slower calcium decay
    
    def modulate_synapse(
        self,
        synapse: "Synapse",
        modulation_strength: float = 0.1
    ) -> float:
        """Modulate synaptic strength via gliotransmitter release.
        
        Args:
            synapse: Synapse to modulate
            modulation_strength: Strength of modulation
            
        Returns:
            Modulation factor (multiplier for synaptic weight)
        """
        if self.calcium_level > self.calcium_threshold:
            # High calcium -> enhance synaptic transmission
            return 1.0 + modulation_strength
        return 1.0


@dataclass
class Microglia:
    """Microglial cell for immune function and synaptic pruning.
    
    Microglia are the brain's immune cells that:
    - Surveil the brain for damage or pathogens
    - Phagocytose dead cells and debris
    - Prune synapses during development and learning
    - Release inflammatory cytokines when activated
    """
    
    id: int
    x: int
    y: int
    z: int
    w: int
    
    # Activation state
    activation_level: float = 0.0  # 0=resting, 1=fully activated
    activation_threshold: float = 0.5
    
    # Surveillance radius
    surveillance_radius: float = 10.0
    
    # Phagocytosis parameters
    phagocytosis_rate: float = 0.1
    
    # Synaptic pruning
    pruning_threshold: float = 0.3  # Weak synapses below this are candidates
    pruning_rate: float = 0.05
    
    # Cytokine release
    cytokine_level: float = 0.0
    
    # Age and health
    age: int = 0
    health: float = 1.0
    
    def position(self) -> tuple:
        """Return the 4D position tuple."""
        return (self.x, self.y, self.z, self.w)
    
    def detect_damage(
        self,
        neurons: Dict[int, "Neuron"],
        damage_threshold: float = 0.5
    ) -> List[int]:
        """Detect damaged neurons in surveillance area.
        
        Args:
            neurons: Dictionary of neurons
            damage_threshold: Health threshold for damage detection
            
        Returns:
            List of damaged neuron IDs
        """
        damaged = []
        pos = np.array(self.position())
        
        for neuron_id, neuron in neurons.items():
            if neuron.health < damage_threshold:
                neuron_pos = np.array(neuron.position())
                distance = np.linalg.norm(pos - neuron_pos)
                
                if distance <= self.surveillance_radius:
                    damaged.append(neuron_id)
        
        return damaged
    
    def activate(self, activation_signal: float) -> None:
        """Activate microglia in response to damage signals.
        
        Args:
            activation_signal: Strength of activation signal
        """
        self.activation_level = min(1.0, self.activation_level + activation_signal)
        
        if self.activation_level > self.activation_threshold:
            # Release pro-inflammatory cytokines
            self.cytokine_level = self.activation_level
    
    def prune_synapse(self, synapse: "Synapse") -> bool:
        """Determine if synapse should be pruned.
        
        Args:
            synapse: Synapse to evaluate
            
        Returns:
            True if synapse should be removed
        """
        if self.activation_level < 0.3:
            return False  # Not active enough to prune
        
        # Prune weak synapses
        if abs(synapse.weight) < self.pruning_threshold:
            # Probabilistic pruning based on activation
            return np.random.random() < self.pruning_rate * self.activation_level
        
        return False
    
    def phagocytose(self, neuron: "Neuron") -> bool:
        """Phagocytose (remove) dead or dying neuron.
        
        Args:
            neuron: Neuron to evaluate
            
        Returns:
            True if neuron should be removed
        """
        if neuron.health <= 0.1 and self.activation_level > 0.5:
            return True
        return False
    
    def decay(self, decay_rate: float = 0.1) -> None:
        """Decay activation and cytokine levels toward resting state.
        
        Args:
            decay_rate: Rate of decay per step
        """
        self.activation_level *= (1.0 - decay_rate)
        self.cytokine_level *= (1.0 - decay_rate)


@dataclass
class Oligodendrocyte:
    """Oligodendrocyte for axon myelination and saltatory conduction.
    
    Oligodendrocytes:
    - Wrap axons with myelin sheaths (insulation)
    - Enable saltatory conduction (fast signal propagation)
    - Support axonal integrity and metabolism
    - One oligodendrocyte myelinates multiple axons
    """
    
    id: int
    x: int
    y: int
    z: int
    w: int
    
    # Myelination parameters
    myelinated_axon_ids: List[int] = field(default_factory=list)
    max_axons: int = 50  # Can myelinate up to 50 axon segments
    myelin_thickness: float = 1.0
    
    # Conduction velocity enhancement
    velocity_multiplier: float = 10.0  # 10x faster with myelin
    
    # Metabolic support to axons
    metabolic_support_rate: float = 0.1
    
    # Age and health
    age: int = 0
    health: float = 1.0
    
    def position(self) -> tuple:
        """Return the 4D position tuple."""
        return (self.x, self.y, self.z, self.w)
    
    def myelinate_axon(self, neuron_id: int) -> bool:
        """Myelinate an axon.
        
        Args:
            neuron_id: ID of neuron whose axon to myelinate
            
        Returns:
            True if successfully myelinated
        """
        if len(self.myelinated_axon_ids) >= self.max_axons:
            return False
        
        if neuron_id not in self.myelinated_axon_ids:
            self.myelinated_axon_ids.append(neuron_id)
            return True
        
        return False
    
    def is_myelinated(self, neuron_id: int) -> bool:
        """Check if an axon is myelinated.
        
        Args:
            neuron_id: ID of neuron to check
            
        Returns:
            True if axon is myelinated
        """
        return neuron_id in self.myelinated_axon_ids
    
    def get_delay_reduction(self) -> float:
        """Get synaptic delay reduction due to myelination.
        
        Returns:
            Delay reduction factor (divide delay by this)
        """
        return self.velocity_multiplier * self.myelin_thickness
    
    def support_axon(self, neuron: "Neuron") -> float:
        """Provide metabolic support to myelinated axon.
        
        Args:
            neuron: Neuron to support
            
        Returns:
            Metabolic support amount
        """
        if neuron.id in self.myelinated_axon_ids:
            return self.metabolic_support_rate * self.health
        return 0.0


@dataclass
class NG2Glia:
    """NG2 Glia (Oligodendrocyte Precursor Cells) for remyelination.
    
    NG2 Glia (also called OPCs):
    - Are precursors to oligodendrocytes
    - Can differentiate into myelinating oligodendrocytes
    - Receive synaptic input from neurons
    - Modulate synaptic transmission
    - Involved in remyelination after damage
    """
    
    id: int
    x: int
    y: int
    z: int
    w: int
    
    # Differentiation state
    differentiation_level: float = 0.0  # 0=OPC, 1=mature oligodendrocyte
    differentiation_signal: float = 0.0
    
    # Synaptic input (NG2 cells receive synaptic connections)
    synaptic_input: float = 0.0
    
    # Modulation parameters
    modulation_strength: float = 0.05
    
    # Age and health
    age: int = 0
    health: float = 1.0
    
    def position(self) -> tuple:
        """Return the 4D position tuple."""
        return (self.x, self.y, self.z, self.w)
    
    def receive_synaptic_input(self, input_current: float) -> None:
        """Receive synaptic input from neurons.
        
        Args:
            input_current: Synaptic input current
        """
        self.synaptic_input += input_current
        
        # High activity promotes differentiation
        if self.synaptic_input > 1.0:
            self.differentiation_signal += 0.01
    
    def differentiate(self) -> Optional["Oligodendrocyte"]:
        """Differentiate into mature oligodendrocyte.
        
        Returns:
            New Oligodendrocyte if differentiation occurs, None otherwise
        """
        self.differentiation_level += self.differentiation_signal
        
        if self.differentiation_level >= 1.0:
            # Create new oligodendrocyte at same position
            return Oligodendrocyte(
                id=self.id,  # Reuse ID
                x=self.x,
                y=self.y,
                z=self.z,
                w=self.w,
                health=self.health
            )
        
        return None
    
    def modulate_nearby_synapses(self) -> float:
        """Modulate nearby synaptic transmission.
        
        Returns:
            Modulation factor
        """
        return 1.0 + self.modulation_strength * self.synaptic_input
    
    def decay(self, decay_rate: float = 0.1) -> None:
        """Decay synaptic input.
        
        Args:
            decay_rate: Rate of decay per step
        """
        self.synaptic_input *= (1.0 - decay_rate)


@dataclass
class GlialNetwork:
    """Network of glial cells interacting with neurons."""
    
    astrocytes: Dict[int, Astrocyte] = field(default_factory=dict)
    microglia: Dict[int, Microglia] = field(default_factory=dict)
    oligodendrocytes: Dict[int, Oligodendrocyte] = field(default_factory=dict)
    ng2_glia: Dict[int, NG2Glia] = field(default_factory=dict)
    
    _next_astrocyte_id: int = 0
    _next_microglia_id: int = 0
    _next_oligodendrocyte_id: int = 0
    _next_ng2_id: int = 0
    
    def add_astrocyte(
        self,
        x: int,
        y: int,
        z: int,
        w: int,
        **kwargs
    ) -> Astrocyte:
        """Add an astrocyte to the network.
        
        Args:
            x, y, z, w: 4D position
            **kwargs: Additional astrocyte parameters
            
        Returns:
            Created Astrocyte
        """
        astrocyte = Astrocyte(
            id=self._next_astrocyte_id,
            x=x, y=y, z=z, w=w,
            **kwargs
        )
        self.astrocytes[astrocyte.id] = astrocyte
        self._next_astrocyte_id += 1
        return astrocyte
    
    def add_microglia(
        self,
        x: int,
        y: int,
        z: int,
        w: int,
        **kwargs
    ) -> Microglia:
        """Add a microglia to the network.
        
        Args:
            x, y, z, w: 4D position
            **kwargs: Additional microglia parameters
            
        Returns:
            Created Microglia
        """
        microglia = Microglia(
            id=self._next_microglia_id,
            x=x, y=y, z=z, w=w,
            **kwargs
        )
        self.microglia[microglia.id] = microglia
        self._next_microglia_id += 1
        return microglia
    
    def add_oligodendrocyte(
        self,
        x: int,
        y: int,
        z: int,
        w: int,
        **kwargs
    ) -> Oligodendrocyte:
        """Add an oligodendrocyte to the network.
        
        Args:
            x, y, z, w: 4D position
            **kwargs: Additional oligodendrocyte parameters
            
        Returns:
            Created Oligodendrocyte
        """
        oligo = Oligodendrocyte(
            id=self._next_oligodendrocyte_id,
            x=x, y=y, z=z, w=w,
            **kwargs
        )
        self.oligodendrocytes[oligo.id] = oligo
        self._next_oligodendrocyte_id += 1
        return oligo
    
    def add_ng2_glia(
        self,
        x: int,
        y: int,
        z: int,
        w: int,
        **kwargs
    ) -> NG2Glia:
        """Add an NG2 glia to the network.
        
        Args:
            x, y, z, w: 4D position
            **kwargs: Additional NG2 glia parameters
            
        Returns:
            Created NG2Glia
        """
        ng2 = NG2Glia(
            id=self._next_ng2_id,
            x=x, y=y, z=z, w=w,
            **kwargs
        )
        self.ng2_glia[ng2.id] = ng2
        self._next_ng2_id += 1
        return ng2
    
    def step(
        self,
        neurons: Dict[int, "Neuron"],
        synapses: List["Synapse"]
    ) -> None:
        """Update all glial cells for one time step.
        
        Args:
            neurons: Dictionary of neurons
            synapses: List of synapses
        """
        # Update astrocytes
        for astrocyte in self.astrocytes.values():
            astrocyte.decay()
        
        # Update microglia
        for microglia in self.microglia.values():
            microglia.decay()
            # Detect and respond to damage
            damaged = microglia.detect_damage(neurons)
            if damaged:
                microglia.activate(0.1 * len(damaged))
        
        # Update NG2 glia
        for ng2 in list(self.ng2_glia.values()):
            ng2.decay()
            # Check for differentiation
            new_oligo = ng2.differentiate()
            if new_oligo is not None:
                # Replace NG2 with oligodendrocyte
                del self.ng2_glia[ng2.id]
                self.oligodendrocytes[new_oligo.id] = new_oligo
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about glial network.
        
        Returns:
            Dictionary with counts of each glial cell type
        """
        return {
            "astrocytes": len(self.astrocytes),
            "microglia": len(self.microglia),
            "oligodendrocytes": len(self.oligodendrocytes),
            "ng2_glia": len(self.ng2_glia),
        }
