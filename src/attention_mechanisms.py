"""Attention mechanisms for selective processing and focus.

This module implements various attention mechanisms including top-down
(goal-driven) and bottom-up (stimulus-driven) attention, as well as
winner-take-all circuits.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class AttentionMechanism:
    """Base class for attention mechanisms."""
    
    def __init__(self):
        """Initialize attention mechanism."""
        self.attention_weights: Dict[int, float] = {}
    
    def compute_attention(
        self,
        neuron_ids: np.ndarray,
        activations: np.ndarray,
        context: Dict[str, any] = None
    ) -> np.ndarray:
        """Compute attention weights for neurons.
        
        Args:
            neuron_ids: Array of neuron IDs
            activations: Array of neuron activations
            context: Optional context information
            
        Returns:
            Array of attention weights (0-1)
        """
        raise NotImplementedError


class BottomUpAttention(AttentionMechanism):
    """Bottom-up (stimulus-driven) attention based on salience.
    
    This mechanism automatically directs attention to the most salient
    (distinctive or surprising) stimuli in the input.
    """
    
    def __init__(
        self,
        salience_threshold: float = 0.5,
        inhibition_radius: float = 5.0
    ):
        """Initialize bottom-up attention.
        
        Args:
            salience_threshold: Minimum activation for salience
            inhibition_radius: Radius of lateral inhibition
        """
        super().__init__()
        self.salience_threshold = salience_threshold
        self.inhibition_radius = inhibition_radius
    
    def compute_attention(
        self,
        neuron_ids: np.ndarray,
        activations: np.ndarray,
        positions: Optional[np.ndarray] = None,
        context: Dict[str, any] = None
    ) -> np.ndarray:
        """Compute bottom-up attention based on activation salience.
        
        Args:
            neuron_ids: Array of neuron IDs
            activations: Array of neuron activations
            positions: Optional array of neuron positions (N x 4)
            context: Optional context information
            
        Returns:
            Array of attention weights
        """
        # Normalize activations
        if np.max(activations) > 0:
            normalized = activations / np.max(activations)
        else:
            normalized = activations
        
        # Compute salience (deviation from mean)
        mean_activation = np.mean(activations)
        salience = np.abs(activations - mean_activation)
        
        # Normalize salience
        if np.max(salience) > 0:
            salience = salience / np.max(salience)
        
        # Apply threshold
        attention = np.where(salience > self.salience_threshold, salience, 0.0)
        
        # Apply lateral inhibition if positions available
        if positions is not None and len(positions) == len(neuron_ids):
            attention = self._apply_lateral_inhibition(
                attention, positions
            )
        
        return attention
    
    def _apply_lateral_inhibition(
        self,
        attention: np.ndarray,
        positions: np.ndarray
    ) -> np.ndarray:
        """Apply lateral inhibition to create winner-take-all dynamics.
        
        Args:
            attention: Current attention values
            positions: Neuron positions
            
        Returns:
            Attention after lateral inhibition
        """
        modified_attention = attention.copy()
        
        # For each neuron, inhibit nearby neurons with lower activation
        for i in range(len(attention)):
            if attention[i] == 0:
                continue
            
            # Compute distances to all other neurons
            distances = np.sqrt(np.sum((positions - positions[i])**2, axis=1))
            
            # Find nearby neurons within inhibition radius
            nearby = distances < self.inhibition_radius
            nearby[i] = False  # Don't inhibit self
            
            # Inhibit nearby neurons with lower activation
            for j in np.where(nearby)[0]:
                if attention[j] < attention[i]:
                    modified_attention[j] *= 0.5  # Reduce by half
        
        return modified_attention


class TopDownAttention(AttentionMechanism):
    """Top-down (goal-driven) attention based on task relevance.
    
    This mechanism directs attention based on task goals and
    expectations, modulating processing of task-relevant features.
    """
    
    def __init__(
        self,
        feature_templates: Optional[Dict[str, np.ndarray]] = None
    ):
        """Initialize top-down attention.
        
        Args:
            feature_templates: Dictionary of feature templates to attend to
        """
        super().__init__()
        self.feature_templates = feature_templates or {}
        self.current_target: Optional[str] = None
    
    def set_target(self, target_name: str) -> None:
        """Set the current attention target.
        
        Args:
            target_name: Name of target feature template
        """
        if target_name in self.feature_templates:
            self.current_target = target_name
        else:
            raise ValueError(f"Unknown target: {target_name}")
    
    def add_template(self, name: str, template: np.ndarray) -> None:
        """Add a feature template.
        
        Args:
            name: Template name
            template: Feature template array
        """
        self.feature_templates[name] = template
    
    def compute_attention(
        self,
        neuron_ids: np.ndarray,
        activations: np.ndarray,
        features: Optional[np.ndarray] = None,
        context: Dict[str, any] = None
    ) -> np.ndarray:
        """Compute top-down attention based on feature matching.
        
        Args:
            neuron_ids: Array of neuron IDs
            activations: Array of neuron activations
            features: Optional feature vectors for each neuron
            context: Optional context information
            
        Returns:
            Array of attention weights
        """
        if self.current_target is None or features is None:
            # No target or features, return uniform attention
            return np.ones_like(activations)
        
        target_template = self.feature_templates[self.current_target]
        
        # Compute similarity to target template
        if features.shape[1] == target_template.shape[0]:
            # Dot product similarity
            similarities = np.dot(features, target_template)
            
            # Normalize to 0-1
            if np.max(np.abs(similarities)) > 0:
                attention = (similarities - np.min(similarities)) / (
                    np.max(similarities) - np.min(similarities)
                )
            else:
                attention = np.zeros_like(similarities)
        else:
            attention = np.ones_like(activations)
        
        return attention


class WinnerTakeAll:
    """Winner-take-all circuit for competitive selection.
    
    This mechanism implements competitive dynamics where only the
    strongest neurons remain active while others are inhibited.
    """
    
    def __init__(
        self,
        k_winners: int = 1,
        inhibition_strength: float = 2.0
    ):
        """Initialize winner-take-all circuit.
        
        Args:
            k_winners: Number of winners to select
            inhibition_strength: Strength of mutual inhibition
        """
        self.k_winners = k_winners
        self.inhibition_strength = inhibition_strength
    
    def select_winners(
        self,
        neuron_ids: np.ndarray,
        activations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select winner neurons through competition.
        
        Args:
            neuron_ids: Array of neuron IDs
            activations: Array of neuron activations
            
        Returns:
            Tuple of (winner_ids, winner_activations)
        """
        if len(activations) == 0:
            return np.array([]), np.array([])
        
        # Find top-k activations
        k = min(self.k_winners, len(activations))
        top_k_indices = np.argpartition(activations, -k)[-k:]
        
        # Sort by activation (descending)
        sorted_indices = top_k_indices[np.argsort(activations[top_k_indices])[::-1]]
        
        winner_ids = neuron_ids[sorted_indices]
        winner_activations = activations[sorted_indices]
        
        return winner_ids, winner_activations
    
    def apply_competition(
        self,
        activations: np.ndarray,
        dt: float = 1.0
    ) -> np.ndarray:
        """Apply competitive dynamics to activations.
        
        Args:
            activations: Current activation levels
            dt: Time step
            
        Returns:
            Updated activations after competition
        """
        # Compute global inhibition
        mean_activation = np.mean(activations)
        inhibition = self.inhibition_strength * mean_activation
        
        # Apply inhibition and ensure non-negative
        new_activations = activations - inhibition * dt
        new_activations = np.maximum(new_activations, 0.0)
        
        return new_activations


class AttentionManager:
    """Manager for coordinating multiple attention mechanisms."""
    
    def __init__(
        self,
        enable_bottom_up: bool = True,
        enable_top_down: bool = True,
        enable_wta: bool = True
    ):
        """Initialize attention manager.
        
        Args:
            enable_bottom_up: Enable bottom-up attention
            enable_top_down: Enable top-down attention
            enable_wta: Enable winner-take-all
        """
        self.bottom_up: Optional[BottomUpAttention] = None
        self.top_down: Optional[TopDownAttention] = None
        self.wta: Optional[WinnerTakeAll] = None
        
        if enable_bottom_up:
            self.bottom_up = BottomUpAttention()
        
        if enable_top_down:
            self.top_down = TopDownAttention()
        
        if enable_wta:
            self.wta = WinnerTakeAll()
    
    def compute_combined_attention(
        self,
        neuron_ids: np.ndarray,
        activations: np.ndarray,
        positions: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        bottom_up_weight: float = 0.5,
        top_down_weight: float = 0.5
    ) -> np.ndarray:
        """Compute combined attention from multiple mechanisms.
        
        Args:
            neuron_ids: Array of neuron IDs
            activations: Array of neuron activations
            positions: Optional neuron positions
            features: Optional feature vectors
            bottom_up_weight: Weight for bottom-up attention
            top_down_weight: Weight for top-down attention
            
        Returns:
            Combined attention weights
        """
        attention = np.ones_like(activations, dtype=float)
        
        # Apply bottom-up attention
        if self.bottom_up is not None:
            bu_attention = self.bottom_up.compute_attention(
                neuron_ids, activations, positions
            )
            attention *= (1 - bottom_up_weight) + bottom_up_weight * bu_attention
        
        # Apply top-down attention
        if self.top_down is not None:
            td_attention = self.top_down.compute_attention(
                neuron_ids, activations, features
            )
            attention *= (1 - top_down_weight) + top_down_weight * td_attention
        
        # Normalize
        if np.max(attention) > 0:
            attention = attention / np.max(attention)
        
        return attention
    
    def apply_attention_to_activations(
        self,
        activations: np.ndarray,
        attention_weights: np.ndarray
    ) -> np.ndarray:
        """Apply attention weights to neuron activations.
        
        Args:
            activations: Original activations
            attention_weights: Attention weights to apply
            
        Returns:
            Modulated activations
        """
        return activations * attention_weights
    
    def select_attended_neurons(
        self,
        neuron_ids: np.ndarray,
        attention_weights: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """Select neurons with attention above threshold.
        
        Args:
            neuron_ids: Array of neuron IDs
            attention_weights: Attention weights
            threshold: Minimum attention for selection
            
        Returns:
            Array of selected neuron IDs
        """
        selected_mask = attention_weights >= threshold
        return neuron_ids[selected_mask]
    
    def get_statistics(self) -> Dict[str, any]:
        """Get attention mechanism statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'bottom_up_enabled': self.bottom_up is not None,
            'top_down_enabled': self.top_down is not None,
            'wta_enabled': self.wta is not None
        }
        
        if self.top_down is not None:
            stats['num_templates'] = len(self.top_down.feature_templates)
            stats['current_target'] = self.top_down.current_target
        
        if self.wta is not None:
            stats['k_winners'] = self.wta.k_winners
        
        return stats


class SpatialAttention:
    """Spatial attention for location-based selection.
    
    This mechanism allows attention to be directed to specific
    spatial locations in the 4D lattice.
    """
    
    def __init__(
        self,
        focus_radius: float = 3.0,
        falloff_rate: float = 0.5
    ):
        """Initialize spatial attention.
        
        Args:
            focus_radius: Radius of attention spotlight
            falloff_rate: Rate of attention falloff with distance
        """
        self.focus_radius = focus_radius
        self.falloff_rate = falloff_rate
        self.focus_location: Optional[np.ndarray] = None
    
    def set_focus(self, location: np.ndarray) -> None:
        """Set the spatial focus location.
        
        Args:
            location: 4D coordinates [x, y, z, w]
        """
        self.focus_location = np.array(location)
    
    def compute_spatial_attention(
        self,
        positions: np.ndarray
    ) -> np.ndarray:
        """Compute attention based on distance from focus.
        
        Args:
            positions: Array of neuron positions (N x 4)
            
        Returns:
            Array of spatial attention weights
        """
        if self.focus_location is None:
            return np.ones(len(positions))
        
        # Compute distances from focus
        distances = np.sqrt(np.sum(
            (positions - self.focus_location)**2, axis=1
        ))
        
        # Compute attention with Gaussian falloff
        attention = np.exp(-(distances / self.focus_radius) ** 2 * self.falloff_rate)
        
        return attention
    
    def shift_focus(
        self,
        direction: np.ndarray,
        step_size: float = 1.0
    ) -> None:
        """Shift attention focus in a direction.
        
        Args:
            direction: Direction vector [dx, dy, dz, dw]
            step_size: Size of shift step
        """
        if self.focus_location is not None:
            direction_normalized = direction / np.linalg.norm(direction)
            self.focus_location += direction_normalized * step_size
