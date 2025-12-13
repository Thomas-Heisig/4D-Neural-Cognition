"""Abstraction mechanisms for 4D cognitive architecture.

This module implements the abstraction hierarchy using the w-axis as a
meta-programmable dimension. Different w-coordinates represent different
levels of abstraction from sensory to metacognitive processing.
"""

from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    try:
        from ..brain_model import BrainModel
    except ImportError:
        from brain_model import BrainModel


class AbstractionManager:
    """Manages abstraction hierarchy across the w-axis.
    
    The w-coordinate is used as an abstraction axis:
    - w=0-2: Sensory layers (raw input, features, objects)
    - w=3-6: Associative layers (patterns, relationships)
    - w=7-10: Executive layers (working memory, decisions)
    - w=11+: Metacognitive layers (learning control, self-monitoring)
    """
    
    def __init__(
        self,
        model: "BrainModel",
        abstraction_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize abstraction manager.
        
        Args:
            model: Brain model with 4D lattice
            abstraction_config: Configuration for abstraction layers
        """
        self.model = model
        self.config = abstraction_config or self._default_config()
        
        # Track neurons by abstraction level
        self.layer_neurons: Dict[str, List[int]] = {}
        self._organize_neurons_by_layer()
    
    def _default_config(self) -> Dict[str, Any]:
        """Get default abstraction configuration."""
        return {
            "sensory_layers": list(range(0, 3)),
            "associative_layers": list(range(3, 7)),
            "executive_layers": list(range(7, 11)),
            "metacognitive_layers": [11],
            "abstraction_rules": {
                "delta_w_threshold": 3,  # Minimum w difference for abstraction connections
                "compression_ratio": 0.5,  # Information compression in bottom-up
                "expansion_ratio": 2.0,    # Information expansion in top-down
            }
        }
    
    def _organize_neurons_by_layer(self) -> None:
        """Organize neurons into abstraction layers based on w-coordinate."""
        self.layer_neurons = {
            "sensory": [],
            "associative": [],
            "executive": [],
            "metacognitive": []
        }
        
        for neuron_id, neuron in self.model.neurons.items():
            w = neuron.position[3]  # w-coordinate
            
            if w in self.config["sensory_layers"]:
                self.layer_neurons["sensory"].append(neuron_id)
            elif w in self.config["associative_layers"]:
                self.layer_neurons["associative"].append(neuron_id)
            elif w in self.config["executive_layers"]:
                self.layer_neurons["executive"].append(neuron_id)
            elif w in self.config["metacognitive_layers"]:
                self.layer_neurons["metacognitive"].append(neuron_id)
    
    def get_abstraction_level(self, neuron_id: int) -> str:
        """Get the abstraction level of a neuron.
        
        Args:
            neuron_id: ID of the neuron
            
        Returns:
            Abstraction level name ('sensory', 'associative', 'executive', 'metacognitive')
        """
        if neuron_id not in self.model.neurons:
            return "unknown"
        
        w = self.model.neurons[neuron_id].position[3]
        
        if w in self.config["sensory_layers"]:
            return "sensory"
        elif w in self.config["associative_layers"]:
            return "associative"
        elif w in self.config["executive_layers"]:
            return "executive"
        elif w in self.config["metacognitive_layers"]:
            return "metacognitive"
        else:
            return "unknown"
    
    def is_abstraction_connection(
        self,
        pre_neuron_id: int,
        post_neuron_id: int
    ) -> bool:
        """Check if a connection is an abstraction connection.
        
        Abstraction connections have large w-difference (Δw > threshold).
        
        Args:
            pre_neuron_id: Pre-synaptic neuron ID
            post_neuron_id: Post-synaptic neuron ID
            
        Returns:
            True if this is an abstraction connection
        """
        if (pre_neuron_id not in self.model.neurons or
            post_neuron_id not in self.model.neurons):
            return False
        
        w_pre = self.model.neurons[pre_neuron_id].position[3]
        w_post = self.model.neurons[post_neuron_id].position[3]
        delta_w = abs(w_post - w_pre)
        
        threshold = self.config["abstraction_rules"]["delta_w_threshold"]
        return delta_w >= threshold
    
    def compute_abstraction_weight_modifier(
        self,
        pre_neuron_id: int,
        post_neuron_id: int,
        base_weight: float
    ) -> float:
        """Compute weight modifier for abstraction connections.
        
        Bottom-up connections (increasing w) compress information.
        Top-down connections (decreasing w) expand/unfold information.
        
        Args:
            pre_neuron_id: Pre-synaptic neuron ID
            post_neuron_id: Post-synaptic neuron ID
            base_weight: Base synaptic weight
            
        Returns:
            Modified weight
        """
        if not self.is_abstraction_connection(pre_neuron_id, post_neuron_id):
            return base_weight
        
        w_pre = self.model.neurons[pre_neuron_id].position[3]
        w_post = self.model.neurons[post_neuron_id].position[3]
        
        rules = self.config["abstraction_rules"]
        
        if w_post > w_pre:  # Bottom-up (compression)
            return base_weight * rules["compression_ratio"]
        else:  # Top-down (expansion)
            return base_weight * rules["expansion_ratio"]
    
    def get_layer_activity(self, layer_name: str) -> np.ndarray:
        """Get activity levels for all neurons in a layer.
        
        Args:
            layer_name: Name of abstraction layer
            
        Returns:
            Array of neuron activities
        """
        if layer_name not in self.layer_neurons:
            return np.array([])
        
        neuron_ids = self.layer_neurons[layer_name]
        activities = []
        
        for neuron_id in neuron_ids:
            if neuron_id in self.model.neurons:
                neuron = self.model.neurons[neuron_id]
                # Use membrane potential as activity proxy
                activities.append(neuron.v - neuron.v_rest)
        
        return np.array(activities)
    
    def compute_abstraction_gradient(self) -> Dict[str, float]:
        """Compute information flow gradients across abstraction layers.
        
        Returns:
            Dictionary mapping layer transitions to gradient values
        """
        gradients = {}
        
        layer_pairs = [
            ("sensory", "associative"),
            ("associative", "executive"),
            ("executive", "metacognitive")
        ]
        
        for lower, higher in layer_pairs:
            lower_activity = self.get_layer_activity(lower)
            higher_activity = self.get_layer_activity(higher)
            
            if len(lower_activity) > 0 and len(higher_activity) > 0:
                # Compute activity difference as gradient proxy
                gradient = np.mean(higher_activity) - np.mean(lower_activity)
                gradients[f"{lower}_to_{higher}"] = float(gradient)
            else:
                gradients[f"{lower}_to_{higher}"] = 0.0
        
        return gradients
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about abstraction hierarchy.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "layer_counts": {},
            "layer_activities": {},
            "abstraction_gradients": self.compute_abstraction_gradient()
        }
        
        for layer_name, neuron_ids in self.layer_neurons.items():
            stats["layer_counts"][layer_name] = len(neuron_ids)
            activity = self.get_layer_activity(layer_name)
            if len(activity) > 0:
                stats["layer_activities"][layer_name] = {
                    "mean": float(np.mean(activity)),
                    "std": float(np.std(activity)),
                    "max": float(np.max(activity))
                }
        
        return stats


__all__ = ["AbstractionManager"]
