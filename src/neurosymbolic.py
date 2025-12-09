"""Neuro-symbolic integration for bridging neural and symbolic reasoning.

This module implements interfaces between neural activity patterns and
symbolic representations, enabling hybrid reasoning capabilities.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from collections import defaultdict
import json


class Concept:
    """Represents a symbolic concept mapped to neural activity."""
    
    def __init__(
        self,
        name: str,
        neuron_cluster: Set[int],
        activation_pattern: Optional[np.ndarray] = None
    ):
        """Initialize a concept.
        
        Args:
            name: Concept name
            neuron_cluster: Set of neuron IDs representing this concept
            activation_pattern: Optional characteristic activation pattern
        """
        self.name = name
        self.neuron_cluster = neuron_cluster
        self.activation_pattern = activation_pattern
        self.activation_history: List[float] = []
    
    def get_activation(
        self,
        neuron_activations: Dict[int, float]
    ) -> float:
        """Get current activation level of concept.
        
        Args:
            neuron_activations: Dictionary mapping neuron IDs to activations
            
        Returns:
            Average activation of concept neurons
        """
        if not self.neuron_cluster:
            return 0.0
        
        activations = [
            neuron_activations.get(nid, 0.0)
            for nid in self.neuron_cluster
        ]
        return np.mean(activations)
    
    def update_history(self, activation: float) -> None:
        """Update activation history.
        
        Args:
            activation: Current activation level
        """
        self.activation_history.append(activation)
        # Keep only recent history
        if len(self.activation_history) > 1000:
            self.activation_history = self.activation_history[-1000:]


class SymbolicRule:
    """Represents a symbolic rule over concepts."""
    
    def __init__(
        self,
        name: str,
        antecedents: List[str],
        consequent: str,
        confidence: float = 1.0
    ):
        """Initialize a symbolic rule.
        
        Args:
            name: Rule name
            antecedents: List of concept names in rule condition
            consequent: Concept name in rule conclusion
            confidence: Rule confidence (0-1)
        """
        self.name = name
        self.antecedents = antecedents
        self.consequent = consequent
        self.confidence = confidence
        self.activation_count = 0
    
    def evaluate(
        self,
        concept_activations: Dict[str, float],
        threshold: float = 0.5
    ) -> Tuple[bool, float]:
        """Evaluate if rule should fire.
        
        Args:
            concept_activations: Dictionary of concept activations
            threshold: Activation threshold for rule firing
            
        Returns:
            Tuple of (should_fire, activation_strength)
        """
        # Check if all antecedents are active
        antecedent_activations = [
            concept_activations.get(name, 0.0)
            for name in self.antecedents
        ]
        
        if not antecedent_activations:
            return False, 0.0
        
        # Use minimum activation (AND logic)
        min_activation = min(antecedent_activations)
        should_fire = min_activation >= threshold
        
        if should_fire:
            self.activation_count += 1
        
        return should_fire, min_activation * self.confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'name': self.name,
            'antecedents': self.antecedents,
            'consequent': self.consequent,
            'confidence': self.confidence,
            'activation_count': self.activation_count
        }


class SymbolicLayer:
    """Bridges neural activity with symbolic reasoning."""
    
    def __init__(
        self,
        clustering_threshold: float = 0.7,
        min_cluster_size: int = 5
    ):
        """Initialize symbolic layer.
        
        Args:
            clustering_threshold: Similarity threshold for clustering
            min_cluster_size: Minimum neurons per concept cluster
        """
        self.clustering_threshold = clustering_threshold
        self.min_cluster_size = min_cluster_size
        
        self.concepts: Dict[str, Concept] = {}
        self.rules: List[SymbolicRule] = []
        
        # Mapping from neurons to concepts
        self.neuron_to_concepts: Dict[int, Set[str]] = defaultdict(set)
    
    def extract_concepts(
        self,
        neuron_activations: Dict[int, np.ndarray],
        activation_threshold: float = 0.5
    ) -> Dict[str, Concept]:
        """Extract concepts from neuron activation patterns.
        
        Args:
            neuron_activations: Dictionary mapping neuron IDs to activation histories
            activation_threshold: Minimum activation for consideration
            
        Returns:
            Dictionary of extracted concepts
        """
        # Cluster neurons with similar activation patterns
        neuron_ids = list(neuron_activations.keys())
        
        if not neuron_ids:
            return {}
        
        # Compute pairwise correlations
        n = len(neuron_ids)
        correlations = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                pattern_i = neuron_activations[neuron_ids[i]]
                pattern_j = neuron_activations[neuron_ids[j]]
                
                # Ensure same length
                min_len = min(len(pattern_i), len(pattern_j))
                if min_len > 0:
                    corr = np.corrcoef(
                        pattern_i[-min_len:],
                        pattern_j[-min_len:]
                    )[0, 1]
                    correlations[i, j] = corr
                    correlations[j, i] = corr
        
        # Simple clustering: group highly correlated neurons
        clusters = []
        assigned = set()
        
        for i in range(n):
            if i in assigned:
                continue
            
            # Find all neurons correlated with neuron i
            cluster = {neuron_ids[i]}
            for j in range(n):
                if j not in assigned and correlations[i, j] > self.clustering_threshold:
                    cluster.add(neuron_ids[j])
            
            if len(cluster) >= self.min_cluster_size:
                clusters.append(cluster)
                assigned.update(cluster)
        
        # Create concepts from clusters
        new_concepts = {}
        for idx, cluster in enumerate(clusters):
            concept_name = f"concept_{idx}"
            
            # Compute average activation pattern
            patterns = [neuron_activations[nid] for nid in cluster]
            min_len = min(len(p) for p in patterns)
            avg_pattern = np.mean([p[-min_len:] for p in patterns], axis=0)
            
            concept = Concept(concept_name, cluster, avg_pattern)
            new_concepts[concept_name] = concept
            
            # Update mappings
            for nid in cluster:
                self.neuron_to_concepts[nid].add(concept_name)
        
        # Merge with existing concepts
        self.concepts.update(new_concepts)
        
        return new_concepts
    
    def add_concept(
        self,
        name: str,
        neuron_cluster: Set[int],
        activation_pattern: Optional[np.ndarray] = None
    ) -> None:
        """Manually add a concept.
        
        Args:
            name: Concept name
            neuron_cluster: Set of neuron IDs
            activation_pattern: Optional activation pattern
        """
        concept = Concept(name, neuron_cluster, activation_pattern)
        self.concepts[name] = concept
        
        for nid in neuron_cluster:
            self.neuron_to_concepts[nid].add(name)
    
    def add_rule(self, rule: SymbolicRule) -> None:
        """Add a symbolic rule.
        
        Args:
            rule: SymbolicRule to add
        """
        self.rules.append(rule)
    
    def add_rule_from_dict(self, rule_dict: Dict[str, Any]) -> None:
        """Add a rule from dictionary specification.
        
        Args:
            rule_dict: Dictionary with rule specification
        """
        rule = SymbolicRule(
            name=rule_dict['name'],
            antecedents=rule_dict['antecedents'],
            consequent=rule_dict['consequent'],
            confidence=rule_dict.get('confidence', 1.0)
        )
        self.add_rule(rule)
    
    def get_concept_activations(
        self,
        neuron_activations: Dict[int, float]
    ) -> Dict[str, float]:
        """Get activation levels for all concepts.
        
        Args:
            neuron_activations: Current neuron activations
            
        Returns:
            Dictionary of concept activations
        """
        concept_activations = {}
        
        for name, concept in self.concepts.items():
            activation = concept.get_activation(neuron_activations)
            concept_activations[name] = activation
            concept.update_history(activation)
        
        return concept_activations
    
    def apply_rules(
        self,
        concept_activations: Dict[str, float],
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Apply symbolic rules to boost consequent concepts.
        
        Args:
            concept_activations: Current concept activations
            threshold: Threshold for rule firing
            
        Returns:
            Dictionary of rule-based activation boosts
        """
        boosts = defaultdict(float)
        
        for rule in self.rules:
            should_fire, strength = rule.evaluate(concept_activations, threshold)
            
            if should_fire:
                # Boost consequent concept
                boosts[rule.consequent] += strength
        
        return dict(boosts)
    
    def reason(
        self,
        neuron_activations: Dict[int, float],
        rule_threshold: float = 0.5,
        num_iterations: int = 3
    ) -> Dict[str, float]:
        """Perform symbolic reasoning over neural activations.
        
        Args:
            neuron_activations: Current neuron activations
            rule_threshold: Threshold for rule firing
            num_iterations: Number of reasoning iterations
            
        Returns:
            Final concept activations after reasoning
        """
        # Get initial concept activations from neural activity
        concept_activations = self.get_concept_activations(neuron_activations)
        
        # Iteratively apply rules
        for _ in range(num_iterations):
            boosts = self.apply_rules(concept_activations, rule_threshold)
            
            # Apply boosts
            for concept_name, boost in boosts.items():
                if concept_name in concept_activations:
                    concept_activations[concept_name] = min(
                        1.0,
                        concept_activations[concept_name] + boost
                    )
        
        return concept_activations
    
    def map_to_neurons(
        self,
        concept_activations: Dict[str, float]
    ) -> Dict[int, float]:
        """Map concept activations back to neuron activations.
        
        Args:
            concept_activations: Concept activation levels
            
        Returns:
            Dictionary of neuron activation boosts
        """
        neuron_boosts = defaultdict(float)
        
        for concept_name, activation in concept_activations.items():
            if concept_name in self.concepts:
                concept = self.concepts[concept_name]
                
                # Distribute activation to concept neurons
                for nid in concept.neuron_cluster:
                    neuron_boosts[nid] += activation / len(concept.neuron_cluster)
        
        return dict(neuron_boosts)
    
    def get_active_concepts(
        self,
        concept_activations: Dict[str, float],
        threshold: float = 0.5
    ) -> List[str]:
        """Get list of currently active concepts.
        
        Args:
            concept_activations: Current concept activations
            threshold: Minimum activation for considering active
            
        Returns:
            List of active concept names
        """
        return [
            name for name, activation in concept_activations.items()
            if activation >= threshold
        ]
    
    def export_knowledge(self, path: str) -> None:
        """Export concepts and rules to JSON file.
        
        Args:
            path: Output file path
        """
        data = {
            'concepts': {
                name: {
                    'neuron_cluster': list(concept.neuron_cluster),
                    'activation_pattern': (
                        concept.activation_pattern.tolist()
                        if concept.activation_pattern is not None
                        else None
                    )
                }
                for name, concept in self.concepts.items()
            },
            'rules': [rule.to_dict() for rule in self.rules]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def import_knowledge(self, path: str) -> None:
        """Import concepts and rules from JSON file.
        
        Args:
            path: Input file path
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Import concepts
        for name, concept_data in data.get('concepts', {}).items():
            cluster = set(concept_data['neuron_cluster'])
            pattern = (
                np.array(concept_data['activation_pattern'])
                if concept_data['activation_pattern'] is not None
                else None
            )
            self.add_concept(name, cluster, pattern)
        
        # Import rules
        for rule_data in data.get('rules', []):
            self.add_rule_from_dict(rule_data)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about symbolic layer.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'num_concepts': len(self.concepts),
            'num_rules': len(self.rules),
            'num_neurons_mapped': len(self.neuron_to_concepts)
        }
        
        if self.concepts:
            cluster_sizes = [len(c.neuron_cluster) for c in self.concepts.values()]
            stats['avg_cluster_size'] = np.mean(cluster_sizes)
            stats['min_cluster_size'] = min(cluster_sizes)
            stats['max_cluster_size'] = max(cluster_sizes)
        
        if self.rules:
            rule_activations = [r.activation_count for r in self.rules]
            stats['total_rule_activations'] = sum(rule_activations)
            stats['avg_rule_activations'] = np.mean(rule_activations)
        
        return stats
