"""
DNA Bank - Central Parameter Repository

This module implements a centralized DNA/parameter bank that stores genetic
information for neurons and glial cells. Cells can retrieve and inherit
parameters with mutations for evolutionary development.

Zentrale DNA/Parameterbank, von der Zellen ihre Konfiguration abrufen können.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import json
import numpy as np
from enum import Enum


class ParameterCategory(Enum):
    """
    Categories of genetic parameters.
    
    Kategorien von genetischen Parametern.
    """
    NEURON_BASIC = "neuron_basic"
    NEURON_ELECTRICAL = "neuron_electrical"
    SYNAPSE = "synapse"
    GLIA = "glia"
    METABOLISM = "metabolism"
    PLASTICITY = "plasticity"
    LIFECYCLE = "lifecycle"


@dataclass
class GeneticParameters:
    """
    Container for a set of genetic parameters.
    
    Enthält genetische Parameter die vererbt und mutiert werden können.
    
    Attributes:
        parameter_id: Unique identifier for this parameter set
        category: Category of parameters
        generation: Generation number
        parent_id: ID of parent parameter set
        parameters: Dictionary of parameter values
        mutation_rate: Rate of parameter mutation
        fitness_score: Fitness evaluation (for selection)
    """
    parameter_id: int
    category: ParameterCategory
    generation: int = 0
    parent_id: int = -1
    parameters: Dict[str, Any] = field(default_factory=dict)
    mutation_rate: float = 0.05
    fitness_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def mutate(self, rng: np.random.Generator) -> 'GeneticParameters':
        """
        Create a mutated copy of these parameters.
        
        Erzeugt eine mutierte Kopie der Parameter für Vererbung.
        
        Args:
            rng: Random number generator
            
        Returns:
            New GeneticParameters with mutations
        """
        mutated_params = {}
        for key, value in self.parameters.items():
            if isinstance(value, (int, float)):
                # Apply Gaussian mutation
                if rng.random() < self.mutation_rate:
                    mutation_factor = 1.0 + rng.normal(0, 0.1)
                    mutated_value = value * mutation_factor
                    # Ensure positive values for parameters that must be positive
                    if value > 0:
                        mutated_value = abs(mutated_value)
                    mutated_params[key] = mutated_value
                else:
                    mutated_params[key] = value
            else:
                mutated_params[key] = value
        
        return GeneticParameters(
            parameter_id=-1,  # Will be assigned by DNABank
            category=self.category,
            generation=self.generation + 1,
            parent_id=self.parameter_id,
            parameters=mutated_params,
            mutation_rate=self.mutation_rate,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'parameter_id': self.parameter_id,
            'category': self.category.value,
            'generation': self.generation,
            'parent_id': self.parent_id,
            'parameters': self.parameters,
            'mutation_rate': self.mutation_rate,
            'fitness_score': self.fitness_score,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GeneticParameters':
        """Create from dictionary representation."""
        return cls(
            parameter_id=data['parameter_id'],
            category=ParameterCategory(data['category']),
            generation=data.get('generation', 0),
            parent_id=data.get('parent_id', -1),
            parameters=data['parameters'],
            mutation_rate=data.get('mutation_rate', 0.05),
            fitness_score=data.get('fitness_score', 0.0),
            metadata=data.get('metadata', {}),
        )


class DNABank:
    """
    Central repository for genetic parameters.
    
    Zentrale Bank für genetische Parameter die von Neuronen und Gliazellen
    abgerufen werden können. Unterstützt Vererbung mit Mutation.
    
    Attributes:
        parameters: Dictionary mapping parameter IDs to GeneticParameters
        next_id: Next available parameter ID
        default_templates: Template parameters for each category
        rng: Random number generator for mutations
    """
    
    # Class constants for configuration
    DEFAULT_FITNESS_THRESHOLD = 0.5  # Threshold for pruning low-fitness parameters
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize DNA bank.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.parameters: Dict[int, GeneticParameters] = {}
        self.next_id: int = 0
        self.default_templates: Dict[ParameterCategory, Dict[str, Any]] = {}
        self.rng = np.random.default_rng(seed)
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """
        Initialize default parameter templates.
        
        Initialisiert Standard-Parametervorlagen für verschiedene Zelltypen.
        """
        # Neuron basic parameters
        self.default_templates[ParameterCategory.NEURON_BASIC] = {
            'soma_diameter': 20.0,  # μm
            'dendrite_count': 5,
            'dendrite_length': 100.0,  # μm
            'dendrite_diameter': 2.0,  # μm
            'axon_length': 500.0,  # μm
            'axon_diameter': 1.0,  # μm
        }
        
        # Neuron electrical parameters
        self.default_templates[ParameterCategory.NEURON_ELECTRICAL] = {
            'resting_potential': -65.0,  # mV
            'threshold': -50.0,  # mV
            'reset_potential': -70.0,  # mV
            'membrane_capacitance': 100.0,  # pF
            'leak_conductance': 0.3,  # mS/cm²
            'na_conductance': 120.0,  # mS/cm²
            'k_conductance': 36.0,  # mS/cm²
        }
        
        # Synapse parameters
        self.default_templates[ParameterCategory.SYNAPSE] = {
            'initial_weight': 0.5,
            'max_weight': 2.0,
            'min_weight': 0.0,
            'learning_rate': 0.01,
            'decay_rate': 0.001,
        }
        
        # Glia cell parameters
        self.default_templates[ParameterCategory.GLIA] = {
            'coverage_radius': 50.0,  # μm
            'metabolic_rate': 1.0,
            'activation_threshold': 0.5,
            'max_associations': 100,
        }
        
        # Metabolism parameters
        self.default_templates[ParameterCategory.METABOLISM] = {
            'energy_production': 1.0,
            'energy_consumption': 0.8,
            'glucose_uptake': 1.0,
            'oxygen_consumption': 1.0,
        }
        
        # Plasticity parameters
        self.default_templates[ParameterCategory.PLASTICITY] = {
            'stdp_tau_plus': 20.0,  # ms
            'stdp_tau_minus': 20.0,  # ms
            'stdp_a_plus': 0.01,
            'stdp_a_minus': 0.01,
            'homeostatic_rate': 0.001,
        }
        
        # Lifecycle parameters
        self.default_templates[ParameterCategory.LIFECYCLE] = {
            'max_age': 100000,  # simulation steps
            'reproduction_threshold': 0.8,
            'health_decay_rate': 0.0001,
            'mutation_rate': 0.05,
        }
    
    def create_parameter_set(self, category: ParameterCategory,
                            custom_params: Optional[Dict[str, Any]] = None,
                            parent_id: int = -1) -> GeneticParameters:
        """
        Create a new parameter set.
        
        Erzeugt einen neuen Parametersatz basierend auf Templates.
        
        Args:
            category: Category of parameters
            custom_params: Optional custom parameter overrides
            parent_id: ID of parent parameter set (for inheritance)
            
        Returns:
            New GeneticParameters object
        """
        # Start with template
        params = self.default_templates.get(category, {}).copy()
        
        # Apply custom parameters
        if custom_params:
            params.update(custom_params)
        
        # Create parameter object
        genetic_params = GeneticParameters(
            parameter_id=self.next_id,
            category=category,
            generation=0,
            parent_id=parent_id,
            parameters=params,
        )
        
        # Store in bank
        self.parameters[self.next_id] = genetic_params
        self.next_id += 1
        
        return genetic_params
    
    def get_parameters(self, parameter_id: int) -> Optional[GeneticParameters]:
        """
        Retrieve parameters by ID.
        
        Args:
            parameter_id: ID of parameter set
            
        Returns:
            GeneticParameters if found, None otherwise
        """
        return self.parameters.get(parameter_id)
    
    def inherit_parameters(self, parent_id: int,
                          apply_mutation: bool = True) -> Optional[GeneticParameters]:
        """
        Create child parameters inheriting from parent with optional mutation.
        
        Erzeugt Kindparameter durch Vererbung von Elternparametern mit Mutation.
        
        Args:
            parent_id: ID of parent parameter set
            apply_mutation: Whether to apply mutations
            
        Returns:
            New GeneticParameters with inherited values
        """
        parent = self.get_parameters(parent_id)
        if parent is None:
            return None
        
        # Create mutated copy
        if apply_mutation:
            child = parent.mutate(self.rng)
        else:
            # Copy without mutation
            child = GeneticParameters(
                parameter_id=-1,
                category=parent.category,
                generation=parent.generation + 1,
                parent_id=parent.parameter_id,
                parameters=parent.parameters.copy(),
                mutation_rate=parent.mutation_rate,
            )
        
        # Assign ID and store
        child.parameter_id = self.next_id
        self.parameters[self.next_id] = child
        self.next_id += 1
        
        return child
    
    def update_fitness(self, parameter_id: int, fitness_score: float):
        """
        Update fitness score for a parameter set.
        
        Aktualisiert den Fitness-Score für evolutionäre Selektion.
        
        Args:
            parameter_id: ID of parameter set
            fitness_score: New fitness score
        """
        if parameter_id in self.parameters:
            self.parameters[parameter_id].fitness_score = fitness_score
    
    def get_best_parameters(self, category: ParameterCategory,
                           top_n: int = 10) -> List[GeneticParameters]:
        """
        Get the best performing parameter sets in a category.
        
        Args:
            category: Parameter category to filter
            top_n: Number of top performers to return
            
        Returns:
            List of top GeneticParameters sorted by fitness
        """
        category_params = [
            p for p in self.parameters.values()
            if p.category == category
        ]
        category_params.sort(key=lambda p: p.fitness_score, reverse=True)
        return category_params[:top_n]
    
    def prune_old_parameters(self, keep_generations: int = 5, 
                            fitness_threshold: Optional[float] = None):
        """
        Remove old parameter sets to save memory.
        
        Entfernt alte Parametersätze um Speicher zu sparen.
        
        Args:
            keep_generations: Number of recent generations to keep
            fitness_threshold: Minimum fitness to keep (default: class constant)
        """
        if not self.parameters:
            return
        
        if fitness_threshold is None:
            fitness_threshold = self.DEFAULT_FITNESS_THRESHOLD
        
        max_generation = max(p.generation for p in self.parameters.values())
        min_generation_to_keep = max_generation - keep_generations
        
        # Keep parameters from recent generations or with high fitness
        to_remove = [
            pid for pid, p in self.parameters.items()
            if p.generation < min_generation_to_keep and p.fitness_score < fitness_threshold
        ]
        
        for pid in to_remove:
            del self.parameters[pid]
    
    def save_to_file(self, filepath: str):
        """
        Save DNA bank to JSON file.
        
        Args:
            filepath: Path to save file
        """
        data = {
            'next_id': self.next_id,
            'parameters': {
                str(k): v.to_dict()
                for k, v in self.parameters.items()
            }
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filepath: str):
        """
        Load DNA bank from JSON file.
        
        Args:
            filepath: Path to load file
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.next_id = data['next_id']
        self.parameters = {
            int(k): GeneticParameters.from_dict(v)
            for k, v in data['parameters'].items()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the DNA bank.
        
        Returns:
            Dictionary with statistics
        """
        if not self.parameters:
            return {'total_parameters': 0}
        
        by_category = {}
        for p in self.parameters.values():
            cat = p.category.value
            if cat not in by_category:
                by_category[cat] = {'count': 0, 'avg_fitness': 0.0, 'max_generation': 0}
            by_category[cat]['count'] += 1
            by_category[cat]['avg_fitness'] += p.fitness_score
            by_category[cat]['max_generation'] = max(
                by_category[cat]['max_generation'], p.generation
            )
        
        # Calculate averages
        for cat_stats in by_category.values():
            if cat_stats['count'] > 0:
                cat_stats['avg_fitness'] /= cat_stats['count']
        
        return {
            'total_parameters': len(self.parameters),
            'next_id': self.next_id,
            'by_category': by_category,
        }
