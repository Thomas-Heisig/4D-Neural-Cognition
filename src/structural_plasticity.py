"""Structural plasticity mechanisms for 4D Neural Cognition.

This module implements structural changes in neural networks:
- Synaptogenesis (new synapse formation)
- Synaptic pruning (synapse elimination)
- Dendritic remodeling (branch growth/retraction)
- Axon guidance (chemoattraction/repulsion)
- Activity-dependent structural changes

References:
- Holtmaat, A., & Svoboda, K. (2009). Experience-dependent structural synaptic plasticity
- Caroni, P., et al. (2012). Structural plasticity upon learning
- Chklovskii, D.B., et al. (2004). Wiring optimization in cortical circuits
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class SynaptogenesisRule:
    """Rules for forming new synapses.
    
    New synapses form based on:
    - Spatial proximity
    - Correlated activity
    - Molecular guidance cues
    - Resource availability
    """
    
    # Formation parameters
    formation_rate: float = 0.01  # Base rate of new synapse formation
    activity_threshold: float = 0.5  # Activity needed for formation
    distance_max: float = 100.0  # Maximum distance for synapse formation
    
    # Initial properties of new synapses
    initial_weight: float = 0.05  # Weak initially
    initial_stability: float = 0.1  # Unstable initially
    
    # Resource constraints
    max_synapses_per_neuron: int = 1000
    
    def can_form_synapse(
        self,
        pre_neuron_synapses: int,
        post_neuron_synapses: int,
        distance: float,
        pre_activity: float,
        post_activity: float
    ) -> bool:
        """Determine if synapse can form.
        
        Args:
            pre_neuron_synapses: Number of existing presynaptic connections
            post_neuron_synapses: Number of existing postsynaptic connections
            distance: Distance between neurons
            pre_activity: Presynaptic activity level
            post_activity: Postsynaptic activity level
            
        Returns:
            True if synapse can form
        """
        # Check resource constraints
        if (pre_neuron_synapses >= self.max_synapses_per_neuron or
            post_neuron_synapses >= self.max_synapses_per_neuron):
            return False
        
        # Check distance
        if distance > self.distance_max:
            return False
        
        # Check activity correlation
        activity_correlation = pre_activity * post_activity
        if activity_correlation < self.activity_threshold:
            return False
        
        # Probabilistic formation
        formation_prob = self.formation_rate * activity_correlation / (distance + 1.0)
        return np.random.random() < formation_prob
    
    def create_new_synapse(
        self,
        pre_id: int,
        post_id: int
    ) -> Dict:
        """Create parameters for new synapse.
        
        Args:
            pre_id: Presynaptic neuron ID
            post_id: Postsynaptic neuron ID
            
        Returns:
            Dictionary of synapse parameters
        """
        return {
            "pre_id": pre_id,
            "post_id": post_id,
            "weight": self.initial_weight,
            "stability": self.initial_stability,
            "age": 0,
        }


@dataclass
class SynapticPruningRule:
    """Rules for eliminating synapses.
    
    Synapses are pruned based on:
    - Low weight/efficacy
    - Lack of activity
    - Competition with other synapses
    - Energy constraints
    """
    
    # Pruning parameters
    pruning_rate: float = 0.005  # Base rate of pruning
    weak_synapse_threshold: float = 0.1  # Synapses below this are candidates
    inactivity_threshold: int = 1000  # Steps without activity
    
    # Competitive pruning
    competition_strength: float = 0.5
    
    # Age-dependent pruning (critical periods)
    critical_period_end: int = 10000
    enhanced_pruning_factor: float = 2.0
    
    def should_prune_synapse(
        self,
        weight: float,
        last_active: int,
        current_step: int,
        stability: float = 0.5,
        nearby_synapses: int = 0
    ) -> bool:
        """Determine if synapse should be pruned.
        
        Args:
            weight: Synaptic weight
            last_active: Last time synapse was active
            current_step: Current simulation step
            stability: Synapse stability (0-1)
            nearby_synapses: Number of nearby competing synapses
            
        Returns:
            True if synapse should be pruned
        """
        # Protected if stable
        if stability > 0.8:
            return False
        
        # Weak synapse
        if abs(weight) < self.weak_synapse_threshold:
            # More likely to prune if inactive
            inactivity = current_step - last_active
            if inactivity > self.inactivity_threshold:
                prune_prob = self.pruning_rate * 10.0
                
                # Enhanced during critical period
                if current_step < self.critical_period_end:
                    prune_prob *= self.enhanced_pruning_factor
                
                # Competition from nearby synapses
                competition = 1.0 + nearby_synapses * self.competition_strength * 0.1
                prune_prob *= competition
                
                return np.random.random() < prune_prob
        
        # Random pruning at low rate
        return np.random.random() < self.pruning_rate * (1.0 - stability)


@dataclass
class DendriticRemodelingRule:
    """Rules for dendritic branch growth and retraction.
    
    Dendrites remodel based on:
    - Local activity patterns
    - Available resources
    - Guidance cues
    - Competition
    """
    
    # Growth parameters
    growth_rate: float = 0.1  # Dendrite extension rate (μm/step)
    retraction_rate: float = 0.05  # Dendrite retraction rate
    
    # Activity dependence
    activity_promotes_growth: float = 1.5
    inactivity_promotes_retraction: float = 2.0
    
    # Resource constraints
    max_dendritic_length: float = 1000.0  # μm
    
    # Branching
    branching_probability: float = 0.01
    max_branch_order: int = 5
    
    def update_branch(
        self,
        current_length: float,
        activity_level: float,
        total_dendritic_length: float,
        branch_order: int
    ) -> Tuple[float, bool]:
        """Update dendritic branch.
        
        Args:
            current_length: Current branch length
            activity_level: Activity in this branch (0-1)
            total_dendritic_length: Total dendrite length of neuron
            branch_order: Branch order (1=primary, 2=secondary, etc.)
            
        Returns:
            Tuple of (length_change, should_branch)
        """
        # Check resource constraints
        if total_dendritic_length >= self.max_dendritic_length:
            return -self.retraction_rate, False
        
        # Activity-dependent growth/retraction
        if activity_level > 0.5:
            # Active branch: grow
            length_change = self.growth_rate * activity_level * self.activity_promotes_growth
        elif activity_level < 0.2:
            # Inactive branch: retract
            length_change = -self.retraction_rate * (1.0 - activity_level) * self.inactivity_promotes_retraction
        else:
            # Neutral
            length_change = 0.0
        
        # Branch growth is slower for higher-order branches
        length_change /= (1.0 + branch_order * 0.5)
        
        # Branching
        should_branch = False
        if (branch_order < self.max_branch_order and
            activity_level > 0.6 and
            np.random.random() < self.branching_probability):
            should_branch = True
        
        return length_change, should_branch


@dataclass
class AxonGuidanceRule:
    """Rules for axon guidance during growth.
    
    Axons navigate based on:
    - Chemoattractants (e.g., Netrin, Semaphorins)
    - Chemorepellents
    - Contact guidance (fasciculation)
    - Target-derived factors
    """
    
    # Guidance molecules
    attractants: Dict[str, float] = field(default_factory=dict)
    repellents: Dict[str, float] = field(default_factory=dict)
    
    # Growth parameters
    extension_rate: float = 10.0  # μm/step
    turning_sensitivity: float = 0.5
    
    # Target finding
    target_affinity: float = 1.0
    
    def calculate_guidance_vector(
        self,
        current_position: np.ndarray,
        target_position: Optional[np.ndarray] = None,
        attractant_sources: List[Tuple[np.ndarray, str, float]] = None,
        repellent_sources: List[Tuple[np.ndarray, str, float]] = None
    ) -> np.ndarray:
        """Calculate guidance vector for axon growth.
        
        Args:
            current_position: Current axon tip position
            target_position: Optional target position
            attractant_sources: List of (position, molecule_type, concentration)
            repellent_sources: List of (position, molecule_type, concentration)
            
        Returns:
            Guidance vector (direction of growth)
        """
        guidance = np.zeros_like(current_position, dtype=float)
        
        # Target-derived factor
        if target_position is not None:
            direction = target_position - current_position
            distance = np.linalg.norm(direction)
            if distance > 0:
                guidance += (direction / distance) * self.target_affinity
        
        # Attractants
        if attractant_sources:
            for source_pos, molecule_type, concentration in attractant_sources:
                if molecule_type in self.attractants:
                    direction = source_pos - current_position
                    distance = np.linalg.norm(direction)
                    if distance > 0:
                        # Inverse square law for diffusion
                        strength = concentration * self.attractants[molecule_type] / (distance**2 + 1.0)
                        guidance += (direction / distance) * strength
        
        # Repellents
        if repellent_sources:
            for source_pos, molecule_type, concentration in repellent_sources:
                if molecule_type in self.repellents:
                    direction = current_position - source_pos  # Away from source
                    distance = np.linalg.norm(direction)
                    if distance > 0:
                        strength = concentration * self.repellents[molecule_type] / (distance**2 + 1.0)
                        guidance += (direction / distance) * strength
        
        # Normalize
        norm = np.linalg.norm(guidance)
        if norm > 0:
            guidance = guidance / norm
        
        return guidance * self.turning_sensitivity


@dataclass
class StructuralPlasticityManager:
    """Manager for all structural plasticity processes."""
    
    synaptogenesis: SynaptogenesisRule = field(default_factory=SynaptogenesisRule)
    pruning: SynapticPruningRule = field(default_factory=SynapticPruningRule)
    dendritic_remodeling: DendriticRemodelingRule = field(default_factory=DendriticRemodelingRule)
    axon_guidance: AxonGuidanceRule = field(default_factory=AxonGuidanceRule)
    
    # Statistics
    synapses_formed: int = 0
    synapses_pruned: int = 0
    branches_added: int = 0
    branches_removed: int = 0
    
    def step(
        self,
        neurons: Dict,
        synapses: List,
        current_step: int,
        neuron_activities: Dict[int, float]
    ) -> Tuple[List[Dict], List[int]]:
        """Execute one step of structural plasticity.
        
        Args:
            neurons: Dictionary of neurons
            synapses: List of existing synapses
            current_step: Current simulation step
            neuron_activities: Recent activity levels for each neuron
            
        Returns:
            Tuple of (new_synapses, pruned_synapse_indices)
        """
        new_synapses = []
        pruned_indices = []
        
        # Count synapses per neuron
        pre_counts = {}
        post_counts = {}
        for synapse in synapses:
            pre_counts[synapse.pre_id] = pre_counts.get(synapse.pre_id, 0) + 1
            post_counts[synapse.post_id] = post_counts.get(synapse.post_id, 0) + 1
        
        # Synaptogenesis: consider pairs of neurons
        neuron_ids = list(neurons.keys())
        n_pairs_to_check = min(100, len(neuron_ids) * 10)  # Sample to avoid O(n^2)
        
        for _ in range(n_pairs_to_check):
            pre_id = np.random.choice(neuron_ids)
            post_id = np.random.choice(neuron_ids)
            
            if pre_id == post_id:
                continue
            
            # Check if connection already exists
            existing = any(s.pre_id == pre_id and s.post_id == post_id for s in synapses)
            if existing:
                continue
            
            # Get neurons
            pre_neuron = neurons[pre_id]
            post_neuron = neurons[post_id]
            
            # Calculate distance
            pre_pos = np.array(pre_neuron.position())
            post_pos = np.array(post_neuron.position())
            distance = np.linalg.norm(pre_pos - post_pos)
            
            # Get activities
            pre_activity = neuron_activities.get(pre_id, 0.0)
            post_activity = neuron_activities.get(post_id, 0.0)
            
            # Check if synapse can form
            if self.synaptogenesis.can_form_synapse(
                pre_counts.get(pre_id, 0),
                post_counts.get(post_id, 0),
                distance,
                pre_activity,
                post_activity
            ):
                new_syn = self.synaptogenesis.create_new_synapse(pre_id, post_id)
                new_synapses.append(new_syn)
                self.synapses_formed += 1
        
        # Synaptic pruning: check existing synapses
        for idx, synapse in enumerate(synapses):
            # Get synapse properties
            weight = synapse.weight
            last_active = getattr(synapse, 'last_active', 0)
            stability = getattr(synapse, 'stability', 0.5)
            
            # Count nearby synapses (same postsynaptic neuron)
            nearby = post_counts.get(synapse.post_id, 1) - 1
            
            # Check if should prune
            if self.pruning.should_prune_synapse(
                weight,
                last_active,
                current_step,
                stability,
                nearby
            ):
                pruned_indices.append(idx)
                self.synapses_pruned += 1
        
        return new_synapses, pruned_indices
    
    def get_statistics(self) -> Dict[str, int]:
        """Get structural plasticity statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "synapses_formed": self.synapses_formed,
            "synapses_pruned": self.synapses_pruned,
            "net_synapse_change": self.synapses_formed - self.synapses_pruned,
            "branches_added": self.branches_added,
            "branches_removed": self.branches_removed,
        }


@dataclass
class CriticalPeriod:
    """Critical period for enhanced plasticity.
    
    Critical periods are developmental windows with heightened
    structural plasticity, allowing experience to shape circuits.
    """
    
    name: str
    start_time: int
    end_time: int
    
    # Enhanced plasticity factors
    synaptogenesis_multiplier: float = 3.0
    pruning_multiplier: float = 2.0
    
    # Affected brain regions
    regions: List[str] = field(default_factory=list)
    
    def is_active(self, current_time: int) -> bool:
        """Check if critical period is currently active.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            True if in critical period
        """
        return self.start_time <= current_time <= self.end_time
    
    def get_plasticity_multiplier(self, current_time: int) -> Tuple[float, float]:
        """Get plasticity multipliers during critical period.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Tuple of (synaptogenesis_mult, pruning_mult)
        """
        if not self.is_active(current_time):
            return 1.0, 1.0
        
        # Gradual onset and offset
        duration = self.end_time - self.start_time
        time_in_period = current_time - self.start_time
        
        # Gaussian-like profile
        center = duration / 2
        width = duration / 4
        profile = np.exp(-((time_in_period - center) ** 2) / (2 * width ** 2))
        
        syn_mult = 1.0 + (self.synaptogenesis_multiplier - 1.0) * profile
        prune_mult = 1.0 + (self.pruning_multiplier - 1.0) * profile
        
        return syn_mult, prune_mult


def create_visual_critical_period() -> CriticalPeriod:
    """Create critical period for visual development.
    
    Returns:
        Visual CriticalPeriod
    """
    return CriticalPeriod(
        name="Visual critical period",
        start_time=1000,
        end_time=5000,
        synaptogenesis_multiplier=3.0,
        pruning_multiplier=2.5,
        regions=["V1", "V2", "V4"]
    )


def create_language_critical_period() -> CriticalPeriod:
    """Create critical period for language development.
    
    Returns:
        Language CriticalPeriod
    """
    return CriticalPeriod(
        name="Language critical period",
        start_time=500,
        end_time=10000,  # Extended period
        synaptogenesis_multiplier=2.5,
        pruning_multiplier=2.0,
        regions=["Broca", "Wernicke", "auditory_cortex"]
    )
