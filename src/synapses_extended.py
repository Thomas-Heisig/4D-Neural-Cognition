"""Extended synapse types for 4D Neural Cognition.

This module implements various synapse types beyond chemical synapses:
- Electrical synapses (gap junctions) for fast synchronization
- Ribbon synapses for tonic release in sensory neurons
- Silent synapses (NMDA-only) for potential connections
- Dendrodendritic synapses for local circuit processing
- Triadic synapses with astrocyte involvement

References:
- Connors, B.W., & Long, M.A. (2004). Electrical synapses in the mammalian brain
- Sterling, P., & Matthews, G. (2005). Structure and function of ribbon synapses
- Isaac, J.T., et al. (1995). Silent synapses during development
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Dict
import numpy as np

if TYPE_CHECKING:
    try:
        from .brain_model import Neuron, Synapse
        from .glial_cells import Astrocyte
    except ImportError:
        from brain_model import Neuron, Synapse
        from glial_cells import Astrocyte


@dataclass
class ElectricalSynapse:
    """Electrical synapse (gap junction) for bidirectional current flow.
    
    Gap junctions allow direct electrical coupling between neurons via
    connexin (vertebrates) or innexin (invertebrates) proteins.
    
    Properties:
    - Bidirectional current flow
    - No synaptic delay
    - Linear voltage-dependent current
    - Important for synchronization and metabolic coupling
    """
    
    neuron1_id: int
    neuron2_id: int
    
    # Coupling conductance (nS)
    conductance: float = 1.0
    
    # Junctional resistance can be modulated
    modulation_factor: float = 1.0
    
    # Gap junction type
    connexin_type: str = "Cx36"  # Cx36 most common neuronal gap junction
    
    def current(
        self,
        v1: float,
        v2: float
    ) -> tuple[float, float]:
        """Calculate bidirectional current through gap junction.
        
        Args:
            v1: Membrane potential of neuron 1 (mV)
            v2: Membrane potential of neuron 2 (mV)
            
        Returns:
            Tuple of (current to neuron 1, current to neuron 2)
        """
        # Ohm's law: I = g * (V1 - V2)
        effective_conductance = self.conductance * self.modulation_factor
        
        # Current flows from higher to lower potential
        i_12 = effective_conductance * (v1 - v2)  # Current from 1 to 2
        
        # By Kirchhoff's law: current into 2 = -current out of 1
        return -i_12, i_12
    
    def modulate(self, factor: float) -> None:
        """Modulate gap junction conductance.
        
        Args:
            factor: Modulation factor (1.0 = no change)
        """
        self.modulation_factor *= factor
        self.modulation_factor = np.clip(self.modulation_factor, 0.0, 2.0)


@dataclass
class RibbonSynapse:
    """Ribbon synapse for sustained, high-rate neurotransmitter release.
    
    Found in sensory neurons (photoreceptors, hair cells, bipolar cells).
    
    Properties:
    - Large readily-releasable pool
    - Sustained tonic release
    - Minimal depression
    - Graded potential-dependent (not spike-dependent)
    """
    
    pre_id: int
    post_id: int
    
    # Release parameters
    max_release_rate: float = 1.0
    vesicle_pool_size: int = 1000
    available_vesicles: int = 1000
    
    # Graded release based on membrane potential
    v_half: float = -40.0  # Half-activation voltage
    slope: float = 5.0  # Voltage sensitivity
    
    # Ribbon structure parameters
    ribbon_length: float = 1.0  # Relative size
    refill_rate: float = 100.0  # Vesicles/ms
    
    # Postsynaptic strength
    weight: float = 0.1
    
    def __post_init__(self):
        """Initialize vesicle pool."""
        self.available_vesicles = self.vesicle_pool_size
    
    def release_probability(self, v_pre: float) -> float:
        """Calculate release probability from presynaptic voltage.
        
        Args:
            v_pre: Presynaptic membrane potential (mV)
            
        Returns:
            Release probability (0-1)
        """
        # Sigmoid function of voltage
        return 1.0 / (1.0 + np.exp(-(v_pre - self.v_half) / self.slope))
    
    def release(self, v_pre: float, dt: float = 1.0) -> float:
        """Calculate graded neurotransmitter release.
        
        Args:
            v_pre: Presynaptic membrane potential (mV)
            dt: Time step (ms)
            
        Returns:
            Postsynaptic current
        """
        if self.available_vesicles <= 0:
            return 0.0
        
        # Graded release based on voltage
        p_release = self.release_probability(v_pre)
        
        # Number of vesicles released
        n_release = p_release * self.max_release_rate * dt * self.ribbon_length
        n_release = min(n_release, self.available_vesicles)
        
        # Deplete pool
        self.available_vesicles -= int(n_release)
        
        # Refill pool
        refill = min(self.refill_rate * dt, self.vesicle_pool_size - self.available_vesicles)
        self.available_vesicles += int(refill)
        
        # Postsynaptic current
        return n_release * self.weight
    
    def get_pool_fraction(self) -> float:
        """Get fraction of available vesicles.
        
        Returns:
            Fraction of pool available (0-1)
        """
        return self.available_vesicles / self.vesicle_pool_size


@dataclass
class SilentSynapse:
    """Silent synapse containing only NMDA receptors (no AMPA).
    
    Silent synapses are:
    - Non-functional at resting potential (Mg2+ block)
    - Can be "unsilenced" by inserting AMPA receptors
    - Important for development and plasticity
    - Reserve connections for learning
    """
    
    pre_id: int
    post_id: int
    
    # NMDA-only (no AMPA)
    nmda_weight: float = 0.1
    ampa_weight: float = 0.0  # Silent = no AMPA
    
    # Unsilencing parameters
    unsilencing_threshold: float = 0.8  # Cumulative activity needed
    activity_accumulator: float = 0.0
    
    # Once unsilenced, becomes regular synapse
    is_silent: bool = True
    
    def accumulate_activity(self, pre_active: bool, post_active: bool) -> None:
        """Accumulate coincident activity for potential unsilencing.
        
        Args:
            pre_active: Presynaptic spike
            post_active: Postsynaptic spike (strong depolarization)
        """
        if pre_active and post_active:
            self.activity_accumulator += 0.1
            
            # Check for unsilencing
            if self.activity_accumulator >= self.unsilencing_threshold:
                self.unsilence()
    
    def unsilence(self) -> None:
        """Unsilence synapse by inserting AMPA receptors."""
        if self.is_silent:
            self.is_silent = False
            self.ampa_weight = self.nmda_weight  # Insert AMPA
    
    def current(self, v_post: float, pre_spike: bool) -> float:
        """Calculate synaptic current.
        
        Args:
            v_post: Postsynaptic membrane potential (mV)
            pre_spike: Presynaptic spike
            
        Returns:
            Synaptic current
        """
        if not pre_spike:
            return 0.0
        
        if self.is_silent:
            # Only NMDA current (blocked at rest)
            mg_block = 1.0 / (1.0 + 1.0 * np.exp(-0.062 * v_post))
            return self.nmda_weight * mg_block
        else:
            # AMPA + NMDA
            mg_block = 1.0 / (1.0 + 1.0 * np.exp(-0.062 * v_post))
            return self.ampa_weight + self.nmda_weight * mg_block


@dataclass
class DendrodendriticSynapse:
    """Dendrodendritic synapse for reciprocal local circuit processing.
    
    Found in:
    - Olfactory bulb (mitral-granule cells)
    - Thalamus (reciprocal connections)
    - Local inhibitory circuits
    
    Properties:
    - Reciprocal connections
    - Often inhibitory
    - No axonal involvement
    - Local processing
    """
    
    neuron1_id: int
    neuron2_id: int
    
    # Reciprocal connections (often one excitatory, one inhibitory)
    weight_1to2: float = 0.1
    weight_2to1: float = -0.1  # Inhibitory
    
    synapse_type_1to2: str = "excitatory"
    synapse_type_2to1: str = "inhibitory"
    
    # Dendritic location (proximal vs distal affects integration)
    dendritic_distance: float = 0.5  # 0=soma, 1=distal
    
    def transmission_1to2(self, spike1: bool) -> float:
        """Transmission from neuron 1 to neuron 2.
        
        Args:
            spike1: Spike from neuron 1
            
        Returns:
            Synaptic current to neuron 2
        """
        if spike1:
            # Attenuate based on dendritic distance
            attenuation = np.exp(-self.dendritic_distance)
            return self.weight_1to2 * attenuation
        return 0.0
    
    def transmission_2to1(self, spike2: bool) -> float:
        """Transmission from neuron 2 to neuron 1.
        
        Args:
            spike2: Spike from neuron 2
            
        Returns:
            Synaptic current to neuron 1
        """
        if spike2:
            attenuation = np.exp(-self.dendritic_distance)
            return self.weight_2to1 * attenuation
        return 0.0


@dataclass
class TriadicSynapse:
    """Triadic synapse with astrocyte participation (tripartite synapse).
    
    Structure: Presynaptic neuron → Postsynaptic neuron + Astrocyte
    
    The astrocyte:
    - Uptakes neurotransmitter from cleft
    - Releases gliotransmitters
    - Modulates synaptic strength
    - Integrates activity from multiple synapses
    """
    
    pre_id: int
    post_id: int
    astrocyte_id: Optional[int] = None
    
    # Basic synaptic parameters
    weight: float = 0.1
    synapse_type: str = "excitatory"
    
    # Astrocyte modulation
    astrocyte_modulation: float = 1.0
    gliotransmitter_level: float = 0.0
    
    # Heterosynaptic modulation via astrocyte
    heterosynaptic_strength: float = 0.1
    
    def transmit(
        self,
        pre_spike: bool,
        astrocyte: Optional["Astrocyte"] = None
    ) -> tuple[float, Dict[str, float]]:
        """Transmit signal with astrocyte modulation.
        
        Args:
            pre_spike: Presynaptic spike
            astrocyte: Associated astrocyte (if available)
            
        Returns:
            Tuple of (postsynaptic current, astrocyte effects)
        """
        if not pre_spike:
            return 0.0, {}
        
        # Basic synaptic transmission
        base_current = self.weight * self.astrocyte_modulation
        
        astrocyte_effects = {}
        
        if astrocyte is not None:
            # Astrocyte uptakes neurotransmitter
            neurotransmitter_type = "glutamate" if self.synapse_type == "excitatory" else "gaba"
            remaining = astrocyte.uptake_neurotransmitter(
                neurotransmitter_type,
                amount=1.0,
                uptake_rate=0.8
            )
            
            # Reduced neurotransmitter in cleft
            base_current *= remaining
            
            # Astrocyte may release gliotransmitters
            gliotransmitters = astrocyte.release_gliotransmitter()
            
            if gliotransmitters:
                # D-serine enhances NMDA receptors
                if "d-serine" in gliotransmitters:
                    astrocyte_effects["nmda_enhancement"] = gliotransmitters["d-serine"]
                
                # Glutamate can modulate nearby synapses (heterosynaptic)
                if "glutamate" in gliotransmitters:
                    astrocyte_effects["heterosynaptic_modulation"] = (
                        gliotransmitters["glutamate"] * self.heterosynaptic_strength
                    )
                
                # ATP can modulate synaptic strength
                if "ATP" in gliotransmitters:
                    self.astrocyte_modulation *= (1.0 + gliotransmitters["ATP"] * 0.1)
        
        return base_current, astrocyte_effects
    
    def update_modulation(self, astrocyte_calcium: float) -> None:
        """Update astrocyte modulation based on astrocyte calcium.
        
        Args:
            astrocyte_calcium: Astrocyte calcium level
        """
        # High astrocyte calcium increases modulation
        self.astrocyte_modulation = 1.0 + astrocyte_calcium * 0.5
        self.astrocyte_modulation = np.clip(self.astrocyte_modulation, 0.5, 2.0)


@dataclass
class ExtendedSynapseNetwork:
    """Network managing multiple synapse types."""
    
    # Different synapse types
    electrical_synapses: Dict[tuple[int, int], ElectricalSynapse] = field(default_factory=dict)
    ribbon_synapses: Dict[tuple[int, int], RibbonSynapse] = field(default_factory=dict)
    silent_synapses: Dict[tuple[int, int], SilentSynapse] = field(default_factory=dict)
    dendrodendritic_synapses: Dict[tuple[int, int], DendrodendriticSynapse] = field(default_factory=dict)
    triadic_synapses: Dict[tuple[int, int], TriadicSynapse] = field(default_factory=dict)
    
    def add_electrical_synapse(
        self,
        neuron1_id: int,
        neuron2_id: int,
        conductance: float = 1.0,
        **kwargs
    ) -> ElectricalSynapse:
        """Add a gap junction between two neurons.
        
        Args:
            neuron1_id: First neuron ID
            neuron2_id: Second neuron ID
            conductance: Coupling conductance
            **kwargs: Additional parameters
            
        Returns:
            Created ElectricalSynapse
        """
        # Use sorted tuple as key (bidirectional)
        key = tuple(sorted([neuron1_id, neuron2_id]))
        
        synapse = ElectricalSynapse(
            neuron1_id=neuron1_id,
            neuron2_id=neuron2_id,
            conductance=conductance,
            **kwargs
        )
        
        self.electrical_synapses[key] = synapse
        return synapse
    
    def add_ribbon_synapse(
        self,
        pre_id: int,
        post_id: int,
        **kwargs
    ) -> RibbonSynapse:
        """Add a ribbon synapse.
        
        Args:
            pre_id: Presynaptic neuron ID
            post_id: Postsynaptic neuron ID
            **kwargs: Additional parameters
            
        Returns:
            Created RibbonSynapse
        """
        synapse = RibbonSynapse(
            pre_id=pre_id,
            post_id=post_id,
            **kwargs
        )
        
        self.ribbon_synapses[(pre_id, post_id)] = synapse
        return synapse
    
    def add_silent_synapse(
        self,
        pre_id: int,
        post_id: int,
        **kwargs
    ) -> SilentSynapse:
        """Add a silent synapse.
        
        Args:
            pre_id: Presynaptic neuron ID
            post_id: Postsynaptic neuron ID
            **kwargs: Additional parameters
            
        Returns:
            Created SilentSynapse
        """
        synapse = SilentSynapse(
            pre_id=pre_id,
            post_id=post_id,
            **kwargs
        )
        
        self.silent_synapses[(pre_id, post_id)] = synapse
        return synapse
    
    def add_triadic_synapse(
        self,
        pre_id: int,
        post_id: int,
        astrocyte_id: Optional[int] = None,
        **kwargs
    ) -> TriadicSynapse:
        """Add a triadic synapse with astrocyte.
        
        Args:
            pre_id: Presynaptic neuron ID
            post_id: Postsynaptic neuron ID
            astrocyte_id: Associated astrocyte ID
            **kwargs: Additional parameters
            
        Returns:
            Created TriadicSynapse
        """
        synapse = TriadicSynapse(
            pre_id=pre_id,
            post_id=post_id,
            astrocyte_id=astrocyte_id,
            **kwargs
        )
        
        self.triadic_synapses[(pre_id, post_id)] = synapse
        return synapse
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about synapse types.
        
        Returns:
            Dictionary with counts of each synapse type
        """
        return {
            "electrical": len(self.electrical_synapses),
            "ribbon": len(self.ribbon_synapses),
            "silent": len(self.silent_synapses),
            "silent_unsilenced": sum(
                1 for s in self.silent_synapses.values() if not s.is_silent
            ),
            "dendrodendritic": len(self.dendrodendritic_synapses),
            "triadic": len(self.triadic_synapses),
        }
