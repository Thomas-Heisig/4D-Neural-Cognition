"""Learning systems framework for 4D Neural Cognition.

This module integrates both biological/psychological learning systems
and machine learning approaches, bridging natural and artificial intelligence.

The fundamental difference: Biological learning is consciousness-capable, flexible,
and context-dependent, while machine learning represents statistical pattern
recognition based on algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class LearningCategory(Enum):
    """Categories of learning systems."""
    BIOLOGICAL = "biological"  # Biological/psychological learning
    MACHINE = "machine"  # Machine learning systems
    HYBRID = "hybrid"  # Hybrid approaches


@dataclass
class LearningContext:
    """Context information for learning processes."""
    
    timestep: int = 0
    environment_state: Dict[str, Any] = field(default_factory=dict)
    internal_state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningResult:
    """Result of a learning process."""
    
    success: bool = False
    learning_delta: float = 0.0  # Amount of learning that occurred
    updated_parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    feedback: str = ""


class LearningSystem(ABC):
    """Abstract base class for all learning systems.
    
    Defines the common interface for both biological and machine learning approaches.
    """
    
    def __init__(self, name: str, category: LearningCategory, config: Optional[Dict] = None):
        """Initialize learning system.
        
        Args:
            name: Name of the learning system
            category: Category (biological, machine, or hybrid)
            config: Configuration dictionary
        """
        self.name = name
        self.category = category
        self.config = config or {}
        self.is_active = False
        self.learning_history: List[LearningResult] = []
        
    @abstractmethod
    def learn(self, context: LearningContext, data: Any) -> LearningResult:
        """Execute learning process.
        
        Args:
            context: Current learning context
            data: Input data for learning
            
        Returns:
            LearningResult with outcome
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get description of this learning system."""
        pass
    
    def activate(self) -> None:
        """Activate this learning system."""
        self.is_active = True
        
    def deactivate(self) -> None:
        """Deactivate this learning system."""
        self.is_active = False
        
    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics for this learning system."""
        if not self.learning_history:
            return {}
        
        successes = sum(1 for r in self.learning_history if r.success)
        total_delta = sum(r.learning_delta for r in self.learning_history)
        
        return {
            "success_rate": successes / len(self.learning_history),
            "average_learning_delta": total_delta / len(self.learning_history),
            "total_learning_episodes": len(self.learning_history)
        }


# ============================================================================
# BIOLOGICAL/PSYCHOLOGICAL LEARNING SYSTEMS
# ============================================================================

class AssociativeLearning(LearningSystem):
    """Associative learning - linking stimuli/actions/consequences.
    
    Forms connections between different events or experiences.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Associative Learning", LearningCategory.BIOLOGICAL, config)
        self.associations: Dict[Tuple[str, str], float] = {}
        self.learning_rate = self.config.get("learning_rate", 0.1)
        self.strength_min = self.config.get("strength_min", 0.0)
        self.strength_max = self.config.get("strength_max", 1.0)
        
    def learn(self, context: LearningContext, data: Any) -> LearningResult:
        """Learn associations between stimuli."""
        if not isinstance(data, dict) or "stimulus_a" not in data or "stimulus_b" not in data:
            return LearningResult(success=False, feedback="Invalid data format")
            
        stim_a = data["stimulus_a"]
        stim_b = data["stimulus_b"]
        strength = data.get("strength", 1.0)
        
        key = (stim_a, stim_b)
        old_strength = self.associations.get(key, 0.0)
        new_strength = old_strength + self.learning_rate * strength
        # Clip to configured range (default [0.0, 1.0] for normalized strength)
        self.associations[key] = np.clip(new_strength, self.strength_min, self.strength_max)
        
        result = LearningResult(
            success=True,
            learning_delta=abs(new_strength - old_strength),
            updated_parameters={"association": {str(key): new_strength}},
            metrics={"association_strength": new_strength},
            feedback=f"Associated {stim_a} with {stim_b}"
        )
        self.learning_history.append(result)
        return result
    
    def get_description(self) -> str:
        return "Linking stimuli, actions, and consequences through repeated co-occurrence"


class NonAssociativeLearning(LearningSystem):
    """Non-associative learning - habituation and sensitization.
    
    Changes in response to repeated stimuli without forming associations.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Non-Associative Learning", LearningCategory.BIOLOGICAL, config)
        self.stimulus_responses: Dict[str, float] = {}
        self.habituation_rate = self.config.get("habituation_rate", 0.05)
        self.sensitization_rate = self.config.get("sensitization_rate", 0.05)
        
    def learn(self, context: LearningContext, data: Any) -> LearningResult:
        """Adjust response to repeated stimuli."""
        if not isinstance(data, dict) or "stimulus" not in data:
            return LearningResult(success=False, feedback="Invalid data format")
            
        stimulus = data["stimulus"]
        response_type = data.get("type", "habituation")  # or "sensitization"
        
        current_response = self.stimulus_responses.get(stimulus, 1.0)
        
        if response_type == "habituation":
            # Decrease response with repeated exposure
            new_response = current_response * (1 - self.habituation_rate)
        else:  # sensitization
            # Increase response with repeated exposure
            new_response = min(1.0, current_response + self.sensitization_rate)
            
        self.stimulus_responses[stimulus] = new_response
        delta = abs(new_response - current_response)
        
        result = LearningResult(
            success=True,
            learning_delta=delta,
            updated_parameters={"response": {stimulus: new_response}},
            metrics={"response_strength": new_response},
            feedback=f"{response_type.capitalize()} to {stimulus}"
        )
        self.learning_history.append(result)
        return result
    
    def get_description(self) -> str:
        return "Habituation and sensitization through repeated stimulus exposure"


class OperantConditioning(LearningSystem):
    """Operant conditioning - learning through rewards and punishments."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Operant Conditioning", LearningCategory.BIOLOGICAL, config)
        self.behavior_values: Dict[str, float] = {}
        self.learning_rate = self.config.get("learning_rate", 0.1)
        
    def learn(self, context: LearningContext, data: Any) -> LearningResult:
        """Learn from consequences of actions."""
        if not isinstance(data, dict) or "behavior" not in data or "reward" not in data:
            return LearningResult(success=False, feedback="Invalid data format")
            
        behavior = data["behavior"]
        reward = data["reward"]
        
        current_value = self.behavior_values.get(behavior, 0.0)
        new_value = current_value + self.learning_rate * reward
        self.behavior_values[behavior] = np.clip(new_value, -1.0, 1.0)
        
        delta = abs(new_value - current_value)
        
        result = LearningResult(
            success=True,
            learning_delta=delta,
            updated_parameters={"behavior_value": {behavior: new_value}},
            metrics={"value": new_value},
            feedback=f"Behavior {behavior} reinforced"
        )
        self.learning_history.append(result)
        return result
    
    def get_description(self) -> str:
        return "Learning through rewards and punishments"


# ============================================================================
# MACHINE LEARNING SYSTEMS
# ============================================================================

class SupervisedLearning(LearningSystem):
    """Supervised learning - learning from labeled training data."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Supervised Learning", LearningCategory.MACHINE, config)
        self.model_parameters: Dict[str, np.ndarray] = {}
        self.learning_rate = self.config.get("learning_rate", 0.01)
        self.training_samples = 0
        
    def learn(self, context: LearningContext, data: Any) -> LearningResult:
        """Learn from labeled examples."""
        if not isinstance(data, dict) or "input" not in data or "label" not in data:
            return LearningResult(success=False, feedback="Invalid data format")
            
        self.training_samples += 1
        prediction_error = data.get("error", 0.5)
        delta = self.learning_rate * prediction_error
        
        result = LearningResult(
            success=True,
            learning_delta=abs(delta),
            updated_parameters={"training_samples": self.training_samples},
            metrics={"samples": self.training_samples, "error": prediction_error},
            feedback=f"Supervised learning update (sample {self.training_samples})"
        )
        self.learning_history.append(result)
        return result
    
    def get_description(self) -> str:
        return "Learning from labeled training data"


class UnsupervisedLearning(LearningSystem):
    """Unsupervised learning - pattern recognition without labels.
    
    Implements similarity-based clustering using distance metrics to group
    similar patterns together. Supports both numeric and non-numeric data.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Unsupervised Learning", LearningCategory.MACHINE, config)
        self.clusters: Dict[int, List[Any]] = {}
        self.cluster_centroids: Dict[int, Any] = {}
        self.num_clusters = self.config.get("num_clusters", 5)
        self.distance_threshold = self.config.get("distance_threshold", 1.0)
        
    def _compute_distance(self, data1: Any, data2: Any) -> float:
        """Compute distance between two data points.
        
        Args:
            data1: First data point (can be numeric array, string, or other)
            data2: Second data point
            
        Returns:
            Distance metric (lower = more similar)
        """
        # Handle numpy arrays
        if isinstance(data1, np.ndarray) and isinstance(data2, np.ndarray):
            if data1.shape == data2.shape:
                return float(np.linalg.norm(data1 - data2))
            else:
                return float('inf')
        
        # Handle lists of numbers
        if isinstance(data1, (list, tuple)) and isinstance(data2, (list, tuple)):
            if len(data1) == len(data2):
                try:
                    arr1 = np.array(data1, dtype=float)
                    arr2 = np.array(data2, dtype=float)
                    return float(np.linalg.norm(arr1 - arr2))
                except (ValueError, TypeError):
                    pass
        
        # Handle strings (character-based similarity with length difference penalty)
        if isinstance(data1, str) and isinstance(data2, str):
            # Simple character-level similarity
            if data1 == data2:
                return 0.0
            max_len = max(len(data1), len(data2))
            if max_len == 0:
                return 0.0
            # Count matching positions and penalize length differences
            min_len = min(len(data1), len(data2))
            matches = sum(c1 == c2 for c1, c2 in zip(data1, data2))
            # Normalized distance: positions + length difference
            position_diff = 1.0 - (matches / max_len)
            length_diff = abs(len(data1) - len(data2)) / max_len
            return (position_diff + length_diff) / 2.0
        
        # Handle same type generic comparison
        if type(data1) == type(data2):
            return 0.0 if data1 == data2 else 1.0
        
        # Different types = maximum distance
        return float('inf')
    
    def _find_nearest_cluster(self, input_data: Any) -> Optional[int]:
        """Find the nearest cluster for the given data point.
        
        Args:
            input_data: Data point to cluster
            
        Returns:
            Cluster ID of nearest cluster, or None if no suitable cluster
        """
        if not self.cluster_centroids:
            return None
        
        min_distance = float('inf')
        nearest_cluster = None
        
        for cluster_id, centroid in self.cluster_centroids.items():
            distance = self._compute_distance(input_data, centroid)
            if distance < min_distance:
                min_distance = distance
                nearest_cluster = cluster_id
        
        # Only assign if within threshold
        if min_distance <= self.distance_threshold:
            return nearest_cluster
        return None
    
    def _update_centroid(self, cluster_id: int) -> None:
        """Update the centroid for a given cluster.
        
        Args:
            cluster_id: ID of cluster to update
        """
        if cluster_id not in self.clusters or not self.clusters[cluster_id]:
            return
        
        cluster_data = self.clusters[cluster_id]
        
        # For numeric arrays, compute mean
        if all(isinstance(d, np.ndarray) for d in cluster_data):
            try:
                self.cluster_centroids[cluster_id] = np.mean(cluster_data, axis=0)
                return
            except (ValueError, TypeError):
                pass
        
        # For lists of numbers, compute mean
        if all(isinstance(d, (list, tuple)) for d in cluster_data):
            try:
                arr = np.array(cluster_data, dtype=float)
                self.cluster_centroids[cluster_id] = arr.mean(axis=0).tolist()
                return
            except (ValueError, TypeError):
                pass
        
        # For other types, use the most recent item as centroid
        self.cluster_centroids[cluster_id] = cluster_data[-1]
        
    def learn(self, context: LearningContext, data: Any) -> LearningResult:
        """Discover patterns in unlabeled data using similarity-based clustering.
        
        Assigns data to nearest cluster based on distance metric. Creates new
        clusters when no suitable cluster exists.
        """
        if not isinstance(data, dict) or "input" not in data:
            return LearningResult(success=False, feedback="Invalid data format")
            
        input_data = data["input"]
        
        # Find nearest cluster
        cluster_id = self._find_nearest_cluster(input_data)
        
        # Create new cluster if needed and we haven't reached max clusters
        if cluster_id is None:
            if len(self.clusters) < self.num_clusters:
                cluster_id = len(self.clusters)
                self.clusters[cluster_id] = []
                self.cluster_centroids[cluster_id] = input_data
            else:
                # Force assignment to nearest cluster even if beyond threshold
                cluster_id = min(
                    self.cluster_centroids.keys(),
                    key=lambda cid: self._compute_distance(input_data, self.cluster_centroids[cid])
                )
        
        # Add to cluster and update centroid
        self.clusters[cluster_id].append(input_data)
        self._update_centroid(cluster_id)
        
        total_patterns = sum(len(c) for c in self.clusters.values())
        
        result = LearningResult(
            success=True,
            learning_delta=1.0 / max(1, total_patterns),
            updated_parameters={"clusters": {k: len(v) for k, v in self.clusters.items()}},
            metrics={"total_patterns": total_patterns, "num_clusters": len(self.clusters)},
            feedback=f"Pattern assigned to cluster {cluster_id} using similarity-based clustering"
        )
        self.learning_history.append(result)
        return result
    
    def get_description(self) -> str:
        return "Discovering patterns in unlabeled data"


class ReinforcementLearning(LearningSystem):
    """Reinforcement learning - learning through reward signals."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Reinforcement Learning", LearningCategory.MACHINE, config)
        self.q_values: Dict[Tuple[str, str], float] = {}
        self.learning_rate = self.config.get("learning_rate", 0.1)
        self.discount_factor = self.config.get("discount_factor", 0.9)
        
    def learn(self, context: LearningContext, data: Any) -> LearningResult:
        """Update policy based on rewards."""
        if not isinstance(data, dict) or "state" not in data or "action" not in data or "reward" not in data:
            return LearningResult(success=False, feedback="Invalid data format")
            
        state = data["state"]
        action = data["action"]
        reward = data["reward"]
        next_state = data.get("next_state", state)
        
        key = (state, action)
        old_q = self.q_values.get(key, 0.0)
        
        next_q_values = [v for k, v in self.q_values.items() if k[0] == next_state]
        max_next_q = max(next_q_values) if next_q_values else 0.0
        
        new_q = old_q + self.learning_rate * (reward + self.discount_factor * max_next_q - old_q)
        self.q_values[key] = new_q
        
        delta = abs(new_q - old_q)
        
        result = LearningResult(
            success=True,
            learning_delta=delta,
            updated_parameters={"q_value": {str(key): new_q}},
            metrics={"q_value": new_q, "reward": reward},
            feedback=f"RL update: state={state}, action={action}"
        )
        self.learning_history.append(result)
        return result
    
    def get_description(self) -> str:
        return "Learning optimal actions through trial-and-error with reward signals"


class TransferLearning(LearningSystem):
    """Transfer learning - applying knowledge from one domain to another."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Transfer Learning", LearningCategory.MACHINE, config)
        self.source_knowledge: Dict[str, Any] = {}
        self.target_adaptations: Dict[str, Any] = {}
        # Minimum similarity threshold for successful transfer (configurable)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.3)
        
    def learn(self, context: LearningContext, data: Any) -> LearningResult:
        """Transfer knowledge to new domain."""
        if not isinstance(data, dict) or "source_domain" not in data or "target_domain" not in data:
            return LearningResult(success=False, feedback="Invalid data format")
            
        source = data["source_domain"]
        target = data["target_domain"]
        similarity = data.get("domain_similarity", 0.7)
        
        transfer_key = f"{source}_to_{target}"
        self.target_adaptations[transfer_key] = {"similarity": similarity}
        
        # Transfer is successful if similarity exceeds configured threshold
        result = LearningResult(
            success=similarity > self.similarity_threshold,
            learning_delta=similarity,
            updated_parameters={"transfers": list(self.target_adaptations.keys())},
            metrics={"similarity": similarity, "transfers": len(self.target_adaptations)},
            feedback=f"Transferred knowledge from {source} to {target}"
        )
        self.learning_history.append(result)
        return result
    
    def get_description(self) -> str:
        return "Transferring knowledge from source to target domains"


class MetaLearning(LearningSystem):
    """Meta-learning - learning to learn."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Meta-Learning", LearningCategory.MACHINE, config)
        self.learning_strategies: Dict[str, Dict] = {}
        self.strategy_performance: Dict[str, List[float]] = {}
        
    def learn(self, context: LearningContext, data: Any) -> LearningResult:
        """Learn effective learning strategies."""
        if not isinstance(data, dict) or "strategy" not in data:
            return LearningResult(success=False, feedback="Invalid data format")
            
        strategy = data["strategy"]
        performance = data.get("performance", 0.5)
        
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = []
        self.strategy_performance[strategy].append(performance)
        
        avg_performance = np.mean(self.strategy_performance[strategy])
        self.learning_strategies[strategy] = {"avg_performance": avg_performance}
        
        result = LearningResult(
            success=True,
            learning_delta=performance,
            updated_parameters={"strategies": dict(self.learning_strategies)},
            metrics={"avg_performance": avg_performance},
            feedback=f"Meta-learning: tested {strategy}"
        )
        self.learning_history.append(result)
        return result
    
    def get_description(self) -> str:
        return "Learning to learn: optimizing the learning process itself"


# ============================================================================
# LEARNING SYSTEM MANAGER
# ============================================================================

class LearningSystemManager:
    """Manages multiple learning systems and their interactions."""
    
    def __init__(self):
        """Initialize the learning system manager."""
        self.systems: Dict[str, LearningSystem] = {}
        self.active_systems: List[str] = []
        
    def register_system(self, system: LearningSystem) -> None:
        """Register a new learning system."""
        self.systems[system.name] = system
        
    def activate_system(self, system_name: str) -> None:
        """Activate a learning system."""
        if system_name in self.systems:
            self.systems[system_name].activate()
            if system_name not in self.active_systems:
                self.active_systems.append(system_name)
                
    def deactivate_system(self, system_name: str) -> None:
        """Deactivate a learning system."""
        if system_name in self.systems:
            self.systems[system_name].deactivate()
            if system_name in self.active_systems:
                self.active_systems.remove(system_name)
                
    def learn(self, context: LearningContext, data: Dict[str, Any]) -> Dict[str, LearningResult]:
        """Execute learning across all active systems."""
        results = {}
        for system_name in self.active_systems:
            if system_name in data:
                system = self.systems[system_name]
                results[system_name] = system.learn(context, data[system_name])
        return results
    
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get metrics from all registered systems."""
        return {name: system.get_metrics() for name, system in self.systems.items()}
    
    def get_biological_systems(self) -> List[LearningSystem]:
        """Get all biological learning systems."""
        return [s for s in self.systems.values() if s.category == LearningCategory.BIOLOGICAL]
    
    def get_machine_systems(self) -> List[LearningSystem]:
        """Get all machine learning systems."""
        return [s for s in self.systems.values() if s.category == LearningCategory.MACHINE]


def create_default_learning_systems() -> LearningSystemManager:
    """Create and register all default learning systems."""
    manager = LearningSystemManager()
    
    # Register biological/psychological learning systems
    manager.register_system(AssociativeLearning())
    manager.register_system(NonAssociativeLearning())
    manager.register_system(OperantConditioning())
    
    # Register machine learning systems
    manager.register_system(SupervisedLearning())
    manager.register_system(UnsupervisedLearning())
    manager.register_system(ReinforcementLearning())
    manager.register_system(TransferLearning())
    manager.register_system(MetaLearning())
    
    return manager
