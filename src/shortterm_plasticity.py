"""Short-term plasticity for working memory function.

This module implements short-term synaptic plasticity mechanisms including
facilitation and depression, which are crucial for working memory and
temporal processing.
"""

from typing import Dict, Tuple
import numpy as np


class ShortTermPlasticity:
    """Short-term synaptic plasticity with facilitation and depression.
    
    This class implements the Tsodyks-Markram model of short-term plasticity,
    which includes both synaptic facilitation (increasing neurotransmitter
    release probability) and depression (depletion of available resources).
    """
    
    def __init__(
        self,
        U: float = 0.5,
        tau_facilitation: float = 50.0,
        tau_depression: float = 200.0,
        baseline_U: float = 0.5
    ):
        """Initialize short-term plasticity.
        
        Args:
            U: Initial utilization of synaptic efficacy
            tau_facilitation: Time constant for facilitation decay (ms)
            tau_depression: Time constant for depression recovery (ms)
            baseline_U: Baseline utilization parameter
        """
        self.baseline_U = baseline_U
        self.tau_f = tau_facilitation
        self.tau_d = tau_depression
        
        # Synapse state variables
        self.u_values: Dict[int, float] = {}  # Utilization per synapse
        self.x_values: Dict[int, float] = {}  # Available resources per synapse
        self.last_spike_time: Dict[int, int] = {}  # Last presynaptic spike time
    
    def initialize_synapse(self, synapse_id: int) -> None:
        """Initialize state for a new synapse.
        
        Args:
            synapse_id: Synapse identifier
        """
        self.u_values[synapse_id] = self.baseline_U
        self.x_values[synapse_id] = 1.0  # Fully recovered
        self.last_spike_time[synapse_id] = -1000  # Far in the past
    
    def process_spike(
        self,
        synapse_id: int,
        current_time: int,
        dt: float = 1.0
    ) -> float:
        """Process a presynaptic spike and return effective strength.
        
        Args:
            synapse_id: Synapse identifier
            current_time: Current simulation time
            dt: Time step
            
        Returns:
            Effective synaptic strength multiplier
        """
        # Initialize if needed
        if synapse_id not in self.u_values:
            self.initialize_synapse(synapse_id)
        
        # Get time since last spike
        delta_t = current_time - self.last_spike_time[synapse_id]
        
        # Decay u and x since last spike
        u_decayed = self.u_values[synapse_id] * np.exp(-delta_t / self.tau_f)
        x_recovered = 1.0 - (1.0 - self.x_values[synapse_id]) * np.exp(-delta_t / self.tau_d)
        
        # Update utilization with facilitation
        u_new = u_decayed + self.baseline_U * (1.0 - u_decayed)
        
        # Calculate effective strength
        effective_strength = u_new * x_recovered
        
        # Update available resources with depression
        x_new = x_recovered - u_new * x_recovered
        
        # Store new values
        self.u_values[synapse_id] = u_new
        self.x_values[synapse_id] = x_new
        self.last_spike_time[synapse_id] = current_time
        
        return effective_strength
    
    def get_current_state(self, synapse_id: int, current_time: int) -> Tuple[float, float]:
        """Get current state without processing a spike.
        
        Args:
            synapse_id: Synapse identifier
            current_time: Current simulation time
            
        Returns:
            Tuple of (u, x) values
        """
        if synapse_id not in self.u_values:
            return self.baseline_U, 1.0
        
        delta_t = current_time - self.last_spike_time[synapse_id]
        
        u_decayed = self.u_values[synapse_id] * np.exp(-delta_t / self.tau_f)
        x_recovered = 1.0 - (1.0 - self.x_values[synapse_id]) * np.exp(-delta_t / self.tau_d)
        
        return u_decayed, x_recovered


class FacilitatingPlasticity(ShortTermPlasticity):
    """Facilitating short-term plasticity.
    
    This variant emphasizes facilitation with weak initial release and
    strong facilitation, typical of synapses important for temporal
    summation and working memory.
    """
    
    def __init__(self):
        """Initialize facilitating plasticity with appropriate parameters."""
        super().__init__(
            U=0.15,  # Low initial release probability
            tau_facilitation=750.0,  # Long facilitation
            tau_depression=50.0,  # Fast depression recovery
            baseline_U=0.15
        )


class DepressingPlasticity(ShortTermPlasticity):
    """Depressing short-term plasticity.
    
    This variant emphasizes depression with strong initial release and
    slow recovery, typical of synapses important for transient signaling
    and adaptation.
    """
    
    def __init__(self):
        """Initialize depressing plasticity with appropriate parameters."""
        super().__init__(
            U=0.6,  # High initial release probability
            tau_facilitation=20.0,  # Fast facilitation decay
            tau_depression=800.0,  # Slow depression recovery
            baseline_U=0.6
        )


class ShortTermPlasticityManager:
    """Manager for short-term plasticity across all synapses."""
    
    def __init__(
        self,
        default_type: str = "balanced",
        enable_stp: bool = True
    ):
        """Initialize short-term plasticity manager.
        
        Args:
            default_type: Default STP type ('balanced', 'facilitating', 'depressing')
            enable_stp: Whether to enable STP
        """
        self.enable_stp = enable_stp
        self.default_type = default_type
        
        # Create default mechanism
        if default_type == "facilitating":
            self.default_mechanism = FacilitatingPlasticity()
        elif default_type == "depressing":
            self.default_mechanism = DepressingPlasticity()
        else:  # balanced
            self.default_mechanism = ShortTermPlasticity()
        
        # Per-synapse mechanisms (for custom types)
        self.synapse_mechanisms: Dict[int, ShortTermPlasticity] = {}
    
    def set_synapse_type(
        self,
        synapse_id: int,
        stp_type: str
    ) -> None:
        """Set STP type for a specific synapse.
        
        Args:
            synapse_id: Synapse identifier
            stp_type: STP type ('balanced', 'facilitating', 'depressing')
        """
        if stp_type == "facilitating":
            self.synapse_mechanisms[synapse_id] = FacilitatingPlasticity()
        elif stp_type == "depressing":
            self.synapse_mechanisms[synapse_id] = DepressingPlasticity()
        else:  # balanced or default
            self.synapse_mechanisms[synapse_id] = ShortTermPlasticity()
    
    def get_mechanism(self, synapse_id: int) -> ShortTermPlasticity:
        """Get STP mechanism for a synapse.
        
        Args:
            synapse_id: Synapse identifier
            
        Returns:
            ShortTermPlasticity mechanism
        """
        return self.synapse_mechanisms.get(synapse_id, self.default_mechanism)
    
    def process_spikes(
        self,
        synapse_ids: np.ndarray,
        current_time: int,
        dt: float = 1.0
    ) -> np.ndarray:
        """Process spikes for multiple synapses.
        
        Args:
            synapse_ids: Array of synapse IDs that received presynaptic spikes
            current_time: Current simulation time
            dt: Time step
            
        Returns:
            Array of effective strength multipliers
        """
        if not self.enable_stp:
            return np.ones_like(synapse_ids, dtype=float)
        
        strengths = np.zeros(len(synapse_ids))
        
        for i, syn_id in enumerate(synapse_ids):
            mechanism = self.get_mechanism(int(syn_id))
            strengths[i] = mechanism.process_spike(int(syn_id), current_time, dt)
        
        return strengths
    
    def apply_to_weights(
        self,
        weights: np.ndarray,
        synapse_ids: np.ndarray,
        spiking_mask: np.ndarray,
        current_time: int,
        dt: float = 1.0
    ) -> np.ndarray:
        """Apply STP to synaptic weights.
        
        Args:
            weights: Array of base synaptic weights
            synapse_ids: Array of synapse IDs
            spiking_mask: Boolean array of which synapses received spikes
            current_time: Current simulation time
            dt: Time step
            
        Returns:
            Modified weights with STP applied
        """
        if not self.enable_stp:
            return weights
        
        modified_weights = weights.copy()
        
        # Only process synapses that received spikes
        spiking_indices = np.where(spiking_mask)[0]
        
        if len(spiking_indices) > 0:
            spiking_syn_ids = synapse_ids[spiking_indices]
            strengths = self.process_spikes(spiking_syn_ids, current_time, dt)
            modified_weights[spiking_indices] *= strengths
        
        return modified_weights
    
    def get_statistics(self) -> Dict[str, any]:
        """Get statistics about short-term plasticity.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'enabled': self.enable_stp,
            'default_type': self.default_type,
            'num_custom_mechanisms': len(self.synapse_mechanisms)
        }
        
        # Collect state statistics
        all_u = []
        all_x = []
        
        # From default mechanism
        all_u.extend(self.default_mechanism.u_values.values())
        all_x.extend(self.default_mechanism.x_values.values())
        
        # From custom mechanisms
        for mechanism in self.synapse_mechanisms.values():
            all_u.extend(mechanism.u_values.values())
            all_x.extend(mechanism.x_values.values())
        
        if all_u:
            stats['mean_utilization'] = np.mean(all_u)
            stats['std_utilization'] = np.std(all_u)
        
        if all_x:
            stats['mean_available_resources'] = np.mean(all_x)
            stats['std_available_resources'] = np.std(all_x)
        
        return stats


class WorkingMemoryCircuit:
    """Working memory circuit using short-term plasticity.
    
    This class implements a simple working memory circuit that uses
    facilitating synapses to maintain persistent activity.
    """
    
    def __init__(
        self,
        num_items: int = 7,
        maintenance_threshold: float = 0.5
    ):
        """Initialize working memory circuit.
        
        Args:
            num_items: Number of items that can be held (capacity)
            maintenance_threshold: Threshold for maintaining an item
        """
        self.num_items = num_items
        self.maintenance_threshold = maintenance_threshold
        
        # Each item has a facilitating synapse
        self.stp_mechanisms = [FacilitatingPlasticity() for _ in range(num_items)]
        
        # Item activation levels
        self.activations = np.zeros(num_items)
        
        # Item presence indicators
        self.items_present = np.zeros(num_items, dtype=bool)
    
    def encode_item(
        self,
        item_id: int,
        current_time: int
    ) -> None:
        """Encode an item into working memory.
        
        Args:
            item_id: Item identifier (0 to num_items-1)
            current_time: Current simulation time
        """
        if 0 <= item_id < self.num_items:
            # Strong initial activation
            strength = self.stp_mechanisms[item_id].process_spike(
                item_id, current_time
            )
            self.activations[item_id] = strength
            self.items_present[item_id] = True
    
    def maintain(
        self,
        current_time: int,
        dt: float = 1.0
    ) -> None:
        """Maintain items in working memory through rehearsal.
        
        Args:
            current_time: Current simulation time
            dt: Time step
        """
        # Rehearse active items
        for item_id in range(self.num_items):
            if self.items_present[item_id]:
                # Periodic reactivation maintains facilitation
                if current_time % 10 == 0:  # Rehearse every 10 steps
                    strength = self.stp_mechanisms[item_id].process_spike(
                        item_id, current_time, dt
                    )
                    self.activations[item_id] = strength
                else:
                    # Decay between rehearsals
                    u, x = self.stp_mechanisms[item_id].get_current_state(
                        item_id, current_time
                    )
                    self.activations[item_id] = u * x
                
                # Check if still above threshold
                if self.activations[item_id] < self.maintenance_threshold:
                    self.items_present[item_id] = False
    
    def get_capacity(self) -> int:
        """Get current number of items in working memory.
        
        Returns:
            Number of active items
        """
        return int(np.sum(self.items_present))
    
    def get_activations(self) -> np.ndarray:
        """Get current activation levels.
        
        Returns:
            Array of activation levels
        """
        return self.activations.copy()
