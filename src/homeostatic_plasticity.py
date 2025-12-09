"""Homeostatic plasticity mechanisms for stability in large networks.

This module implements homeostatic plasticity mechanisms that help maintain
stable neural activity levels despite changes in network structure or input.
"""

from typing import Dict, List, Optional
import numpy as np


class HomeostaticMechanism:
    """Base class for homeostatic plasticity mechanisms."""
    
    def __init__(
        self,
        target_rate: float = 5.0,
        time_constant: float = 1000.0
    ):
        """Initialize homeostatic mechanism.
        
        Args:
            target_rate: Target firing rate in Hz
            time_constant: Time constant for adaptation (in time steps)
        """
        self.target_rate = target_rate
        self.time_constant = time_constant
    
    def update(
        self,
        neuron_id: int,
        current_rate: float,
        current_time: int
    ) -> float:
        """Update homeostatic variables.
        
        Args:
            neuron_id: Neuron identifier
            current_rate: Current firing rate
            current_time: Current simulation time
            
        Returns:
            Scaling factor to apply
        """
        raise NotImplementedError


class SynapticScaling(HomeostaticMechanism):
    """Synaptic scaling - multiplicative adjustment of all synaptic weights.
    
    This mechanism scales all synaptic weights up or down to maintain
    a target firing rate, implementing a form of homeostatic plasticity.
    """
    
    def __init__(
        self,
        target_rate: float = 5.0,
        time_constant: float = 10000.0,
        learning_rate: float = 0.0001
    ):
        """Initialize synaptic scaling.
        
        Args:
            target_rate: Target firing rate in Hz
            time_constant: Time constant for rate estimation
            learning_rate: Rate of scaling adjustment
        """
        super().__init__(target_rate, time_constant)
        self.learning_rate = learning_rate
        self.avg_rates: Dict[int, float] = {}
        self.scaling_factors: Dict[int, float] = {}
    
    def update_rates(
        self,
        neuron_id: int,
        did_spike: bool,
        current_time: int,
        dt: float = 1.0
    ) -> None:
        """Update running average of firing rate.
        
        Args:
            neuron_id: Neuron identifier
            did_spike: Whether neuron spiked this time step
            current_time: Current simulation time
            dt: Time step duration
        """
        if neuron_id not in self.avg_rates:
            self.avg_rates[neuron_id] = 0.0
            self.scaling_factors[neuron_id] = 1.0
        
        # Update exponential moving average of firing rate
        alpha = dt / self.time_constant
        spike_rate = 1.0 / dt if did_spike else 0.0
        self.avg_rates[neuron_id] = (
            (1 - alpha) * self.avg_rates[neuron_id] + alpha * spike_rate
        )
    
    def get_scaling_factor(self, neuron_id: int) -> float:
        """Get current scaling factor for a neuron.
        
        Args:
            neuron_id: Neuron identifier
            
        Returns:
            Scaling factor to multiply synaptic weights by
        """
        if neuron_id not in self.scaling_factors:
            return 1.0
        
        # Update scaling factor based on rate error
        avg_rate = self.avg_rates.get(neuron_id, 0.0)
        rate_error = self.target_rate - avg_rate
        
        # Adjust scaling factor
        delta_scaling = self.learning_rate * rate_error
        self.scaling_factors[neuron_id] = max(
            0.1,  # Minimum scaling
            min(
                10.0,  # Maximum scaling
                self.scaling_factors[neuron_id] + delta_scaling
            )
        )
        
        return self.scaling_factors[neuron_id]
    
    def scale_weights(
        self,
        weights: np.ndarray,
        post_neuron_ids: np.ndarray
    ) -> np.ndarray:
        """Apply scaling factors to synaptic weights.
        
        Args:
            weights: Array of synaptic weights
            post_neuron_ids: Array of postsynaptic neuron IDs
            
        Returns:
            Scaled weights
        """
        scaled_weights = weights.copy()
        
        for i, post_id in enumerate(post_neuron_ids):
            scaling = self.get_scaling_factor(int(post_id))
            scaled_weights[i] *= scaling
        
        return scaled_weights


class IntrinsicExcitability(HomeostaticMechanism):
    """Intrinsic excitability regulation - adjustment of neuron threshold.
    
    This mechanism adjusts the neuron's firing threshold to maintain
    a target firing rate.
    """
    
    def __init__(
        self,
        target_rate: float = 5.0,
        time_constant: float = 10000.0,
        learning_rate: float = 0.001,
        threshold_min: float = -60.0,
        threshold_max: float = -40.0
    ):
        """Initialize intrinsic excitability regulation.
        
        Args:
            target_rate: Target firing rate in Hz
            time_constant: Time constant for rate estimation
            learning_rate: Rate of threshold adjustment
            threshold_min: Minimum threshold value
            threshold_max: Maximum threshold value
        """
        super().__init__(target_rate, time_constant)
        self.learning_rate = learning_rate
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.avg_rates: Dict[int, float] = {}
        self.thresholds: Dict[int, float] = {}
    
    def update_rates(
        self,
        neuron_id: int,
        did_spike: bool,
        current_time: int,
        dt: float = 1.0
    ) -> None:
        """Update running average of firing rate.
        
        Args:
            neuron_id: Neuron identifier
            did_spike: Whether neuron spiked this time step
            current_time: Current simulation time
            dt: Time step duration
        """
        if neuron_id not in self.avg_rates:
            self.avg_rates[neuron_id] = 0.0
            self.thresholds[neuron_id] = -50.0  # Default threshold
        
        # Update exponential moving average
        alpha = dt / self.time_constant
        spike_rate = 1.0 / dt if did_spike else 0.0
        self.avg_rates[neuron_id] = (
            (1 - alpha) * self.avg_rates[neuron_id] + alpha * spike_rate
        )
    
    def get_threshold(self, neuron_id: int) -> float:
        """Get adjusted threshold for a neuron.
        
        Args:
            neuron_id: Neuron identifier
            
        Returns:
            Adjusted firing threshold
        """
        if neuron_id not in self.thresholds:
            return -50.0
        
        # Update threshold based on rate error
        avg_rate = self.avg_rates.get(neuron_id, 0.0)
        rate_error = self.target_rate - avg_rate
        
        # Adjust threshold (higher threshold = lower excitability)
        # If rate too low, decrease threshold; if too high, increase threshold
        delta_threshold = -self.learning_rate * rate_error
        self.thresholds[neuron_id] = max(
            self.threshold_min,
            min(
                self.threshold_max,
                self.thresholds[neuron_id] + delta_threshold
            )
        )
        
        return self.thresholds[neuron_id]


class HomeostaticPlasticityManager:
    """Manager for multiple homeostatic plasticity mechanisms."""
    
    def __init__(
        self,
        enable_synaptic_scaling: bool = True,
        enable_intrinsic_excitability: bool = True,
        target_rate: float = 5.0
    ):
        """Initialize homeostatic plasticity manager.
        
        Args:
            enable_synaptic_scaling: Whether to enable synaptic scaling
            enable_intrinsic_excitability: Whether to enable intrinsic excitability
            target_rate: Target firing rate for all mechanisms
        """
        self.enable_synaptic_scaling = enable_synaptic_scaling
        self.enable_intrinsic_excitability = enable_intrinsic_excitability
        
        self.synaptic_scaling: Optional[SynapticScaling] = None
        self.intrinsic_excitability: Optional[IntrinsicExcitability] = None
        
        if enable_synaptic_scaling:
            self.synaptic_scaling = SynapticScaling(target_rate=target_rate)
        
        if enable_intrinsic_excitability:
            self.intrinsic_excitability = IntrinsicExcitability(target_rate=target_rate)
    
    def update_neuron(
        self,
        neuron_id: int,
        did_spike: bool,
        current_time: int,
        dt: float = 1.0
    ) -> None:
        """Update homeostatic mechanisms for a neuron.
        
        Args:
            neuron_id: Neuron identifier
            did_spike: Whether neuron spiked this time step
            current_time: Current simulation time
            dt: Time step duration
        """
        if self.synaptic_scaling:
            self.synaptic_scaling.update_rates(neuron_id, did_spike, current_time, dt)
        
        if self.intrinsic_excitability:
            self.intrinsic_excitability.update_rates(neuron_id, did_spike, current_time, dt)
    
    def apply_to_weights(
        self,
        weights: np.ndarray,
        post_neuron_ids: np.ndarray
    ) -> np.ndarray:
        """Apply synaptic scaling to weights.
        
        Args:
            weights: Array of synaptic weights
            post_neuron_ids: Array of postsynaptic neuron IDs
            
        Returns:
            Scaled weights
        """
        if self.synaptic_scaling:
            return self.synaptic_scaling.scale_weights(weights, post_neuron_ids)
        return weights
    
    def get_threshold_adjustment(self, neuron_id: int) -> float:
        """Get threshold adjustment for a neuron.
        
        Args:
            neuron_id: Neuron identifier
            
        Returns:
            Adjusted threshold value
        """
        if self.intrinsic_excitability:
            return self.intrinsic_excitability.get_threshold(neuron_id)
        return -50.0  # Default threshold
    
    def get_statistics(self) -> Dict[str, any]:
        """Get statistics about homeostatic mechanisms.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'synaptic_scaling_enabled': self.enable_synaptic_scaling,
            'intrinsic_excitability_enabled': self.enable_intrinsic_excitability
        }
        
        if self.synaptic_scaling:
            stats['num_neurons_tracked_scaling'] = len(self.synaptic_scaling.avg_rates)
            if self.synaptic_scaling.avg_rates:
                rates = list(self.synaptic_scaling.avg_rates.values())
                stats['mean_firing_rate'] = np.mean(rates)
                stats['std_firing_rate'] = np.std(rates)
        
        if self.intrinsic_excitability:
            stats['num_neurons_tracked_excitability'] = len(self.intrinsic_excitability.avg_rates)
            if self.intrinsic_excitability.thresholds:
                thresholds = list(self.intrinsic_excitability.thresholds.values())
                stats['mean_threshold'] = np.mean(thresholds)
                stats['std_threshold'] = np.std(thresholds)
        
        return stats
