"""Neuromodulation systems for 4D Neural Cognition.

This module implements biologically-inspired neuromodulation systems:
- Dopamine: Reward learning and prediction error signaling
- Serotonin: Mood/state regulation and behavioral inhibition
- Norepinephrine: Arousal and attention regulation

Based on computational neuroscience research:
- Dayan, P. (2012). Twenty-Five Lessons from Computational Neuromodulation
- Shine, J.M. (2021). Neuromodulatory influences on integration and segregation
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    try:
        from .brain_model import BrainModel, Neuron, Synapse
    except ImportError:
        from brain_model import BrainModel, Neuron, Synapse


@dataclass
class NeuromodulatorState:
    """State of a neuromodulator system."""
    
    baseline: float = 0.5  # Baseline level
    decay_rate: float = 0.1  # How quickly it returns to baseline
    level: float = None  # Current level (0.0 to 1.0), defaults to baseline
    
    def __post_init__(self):
        """Initialize level to baseline if not provided."""
        if self.level is None:
            self.level = self.baseline
    
    def update_baseline(self, new_baseline: float) -> None:
        """Update the baseline level.
        
        Args:
            new_baseline: New baseline level (0.0 to 1.0)
        """
        self.baseline = max(0.0, min(1.0, new_baseline))
    
    def decay_toward_baseline(self) -> None:
        """Decay the level toward baseline."""
        if self.level > self.baseline:
            self.level = max(self.baseline, self.level - self.decay_rate)
        elif self.level < self.baseline:
            self.level = min(self.baseline, self.level + self.decay_rate)


@dataclass
class DopamineSystem:
    """Dopamine system for reward learning.
    
    Implements temporal difference (TD) learning-inspired reward prediction
    error signaling. High dopamine strengthens recently active synapses,
    low dopamine weakens them.
    """
    
    state: NeuromodulatorState = field(default_factory=lambda: NeuromodulatorState(baseline=0.5))
    reward_history: List[float] = field(default_factory=list)
    max_history: int = 100
    learning_rate_modulation: float = 2.0  # How much dopamine affects learning
    
    def update(self, reward: float, expected_reward: float = 0.0) -> float:
        """Update dopamine based on reward prediction error.
        
        Args:
            reward: Actual reward received
            expected_reward: Expected reward
            
        Returns:
            Reward prediction error (dopamine signal)
        """
        # Temporal difference error
        prediction_error = reward - expected_reward
        
        # Update dopamine level based on prediction error
        # Positive error -> increase dopamine
        # Negative error -> decrease dopamine
        self.state.level = max(0.0, min(1.0, self.state.baseline + prediction_error))
        
        # Track history
        self.reward_history.append(reward)
        if len(self.reward_history) > self.max_history:
            self.reward_history.pop(0)
        
        return prediction_error
    
    def get_learning_rate_multiplier(self) -> float:
        """Get learning rate multiplier based on dopamine level.
        
        Returns:
            Multiplier for learning rate (0.0 to learning_rate_modulation)
        """
        return self.state.level * self.learning_rate_modulation
    
    def modulate_plasticity(self, base_delta_w: float) -> float:
        """Modulate synaptic weight change based on dopamine level.
        
        Args:
            base_delta_w: Base weight change from plasticity rule
            
        Returns:
            Modulated weight change
        """
        return base_delta_w * self.get_learning_rate_multiplier()


@dataclass
class SerotoninSystem:
    """Serotonin system for mood/state regulation.
    
    Serotonin influences:
    - Behavioral inhibition (reduces impulsive actions)
    - Patience and delayed gratification
    - Aversive processing and punishment signals
    """
    
    state: NeuromodulatorState = field(default_factory=lambda: NeuromodulatorState(baseline=0.5))
    punishment_history: List[float] = field(default_factory=list)
    max_history: int = 100
    inhibition_strength: float = 0.5  # How much serotonin inhibits activity
    
    def update(self, punishment: float = 0.0, stress: float = 0.0) -> None:
        """Update serotonin level based on punishment and stress.
        
        Args:
            punishment: Punishment signal (positive = aversive event)
            stress: Stress level
        """
        # High punishment/stress -> lower serotonin
        # Low punishment/stress -> higher serotonin
        aversive_signal = punishment + stress
        self.state.level = max(0.0, min(1.0, self.state.baseline - aversive_signal * 0.5))
        
        # Track punishment history
        if punishment > 0:
            self.punishment_history.append(punishment)
            if len(self.punishment_history) > self.max_history:
                self.punishment_history.pop(0)
    
    def get_inhibition_factor(self) -> float:
        """Get inhibition factor for neural activity.
        
        Higher serotonin = more inhibition (behavioral control)
        
        Returns:
            Inhibition factor (0.0 to 1.0)
        """
        return self.state.level * self.inhibition_strength
    
    def modulate_threshold(self, base_threshold: float) -> float:
        """Modulate firing threshold based on serotonin level.
        
        Higher serotonin -> higher threshold (harder to fire)
        
        Args:
            base_threshold: Base firing threshold
            
        Returns:
            Modulated threshold
        """
        return base_threshold * (1.0 + self.get_inhibition_factor())


@dataclass
class NorepinephrineSystem:
    """Norepinephrine system for arousal and attention.
    
    Norepinephrine influences:
    - Arousal state (alertness)
    - Signal-to-noise ratio (neural gain)
    - Attention and vigilance
    - Response to uncertainty
    """
    
    state: NeuromodulatorState = field(default_factory=lambda: NeuromodulatorState(baseline=0.5))
    uncertainty_history: List[float] = field(default_factory=list)
    max_history: int = 100
    gain_modulation: float = 2.0  # How much NE affects neural gain
    
    def update(self, uncertainty: float = 0.0, novelty: float = 0.0) -> None:
        """Update norepinephrine level based on uncertainty and novelty.
        
        Args:
            uncertainty: Uncertainty in environment/task
            novelty: Novelty of stimuli
        """
        # High uncertainty/novelty -> higher norepinephrine
        arousal_signal = uncertainty + novelty
        self.state.level = max(0.0, min(1.0, self.state.baseline + arousal_signal * 0.5))
        
        # Track uncertainty
        if uncertainty > 0:
            self.uncertainty_history.append(uncertainty)
            if len(self.uncertainty_history) > self.max_history:
                self.uncertainty_history.pop(0)
    
    def get_gain_multiplier(self) -> float:
        """Get neural gain multiplier based on norepinephrine level.
        
        Higher NE -> higher gain (amplified responses)
        
        Returns:
            Gain multiplier (1.0 to 1.0 + gain_modulation)
        """
        return 1.0 + (self.state.level * self.gain_modulation)
    
    def modulate_input(self, input_current: float) -> float:
        """Modulate input current based on norepinephrine level.
        
        Args:
            input_current: Base input current
            
        Returns:
            Modulated input current
        """
        return input_current * self.get_gain_multiplier()


@dataclass
class NeuromodulationSystem:
    """Complete neuromodulation system managing all neuromodulators."""
    
    dopamine: DopamineSystem = field(default_factory=DopamineSystem)
    serotonin: SerotoninSystem = field(default_factory=SerotoninSystem)
    norepinephrine: NorepinephrineSystem = field(default_factory=NorepinephrineSystem)
    
    def step(self) -> None:
        """Update all neuromodulator systems (decay toward baseline)."""
        self.dopamine.state.decay_toward_baseline()
        self.serotonin.state.decay_toward_baseline()
        self.norepinephrine.state.decay_toward_baseline()
    
    def get_state(self) -> Dict[str, float]:
        """Get current state of all neuromodulators.
        
        Returns:
            Dictionary with neuromodulator levels
        """
        return {
            "dopamine": self.dopamine.state.level,
            "serotonin": self.serotonin.state.level,
            "norepinephrine": self.norepinephrine.state.level,
        }
    
    def modulate_learning(self, base_delta_w: float) -> float:
        """Apply all neuromodulatory effects to learning.
        
        Args:
            base_delta_w: Base weight change from plasticity rule
            
        Returns:
            Fully modulated weight change
        """
        # Dopamine modulates magnitude of learning
        delta_w = self.dopamine.modulate_plasticity(base_delta_w)
        return delta_w
    
    def modulate_neuron_update(
        self,
        neuron: "Neuron",
        synaptic_input: float,
        threshold: float
    ) -> tuple[float, float]:
        """Apply neuromodulatory effects to neuron update.
        
        Args:
            neuron: Neuron being updated
            synaptic_input: Synaptic input current
            threshold: Firing threshold
            
        Returns:
            Tuple of (modulated_input, modulated_threshold)
        """
        # Norepinephrine modulates input gain (arousal)
        modulated_input = self.norepinephrine.modulate_input(synaptic_input)
        
        # Serotonin modulates threshold (behavioral inhibition)
        modulated_threshold = self.serotonin.modulate_threshold(threshold)
        
        return modulated_input, modulated_threshold
    
    def process_reward(self, reward: float, expected_reward: float = 0.0) -> float:
        """Process reward signal through dopamine system.
        
        Args:
            reward: Actual reward
            expected_reward: Expected reward
            
        Returns:
            Reward prediction error
        """
        return self.dopamine.update(reward, expected_reward)
    
    def process_punishment(self, punishment: float, stress: float = 0.0) -> None:
        """Process punishment/stress through serotonin system.
        
        Args:
            punishment: Punishment signal
            stress: Stress level
        """
        self.serotonin.update(punishment, stress)
    
    def process_uncertainty(self, uncertainty: float, novelty: float = 0.0) -> None:
        """Process uncertainty/novelty through norepinephrine system.
        
        Args:
            uncertainty: Uncertainty level
            novelty: Novelty of stimuli
        """
        self.norepinephrine.update(uncertainty, novelty)


def create_neuromodulation_system(config: Dict = None) -> NeuromodulationSystem:
    """Create a neuromodulation system from configuration.
    
    Args:
        config: Configuration dictionary with neuromodulator parameters
        
    Returns:
        Configured NeuromodulationSystem
    """
    if config is None:
        config = {}
    
    # Create systems with configured parameters
    dopamine_config = config.get("dopamine", {})
    serotonin_config = config.get("serotonin", {})
    norepinephrine_config = config.get("norepinephrine", {})
    
    dopamine = DopamineSystem(
        state=NeuromodulatorState(
            baseline=dopamine_config.get("baseline", 0.5),
            decay_rate=dopamine_config.get("decay_rate", 0.1)
        ),
        learning_rate_modulation=dopamine_config.get("learning_rate_modulation", 2.0)
    )
    
    serotonin = SerotoninSystem(
        state=NeuromodulatorState(
            baseline=serotonin_config.get("baseline", 0.5),
            decay_rate=serotonin_config.get("decay_rate", 0.1)
        ),
        inhibition_strength=serotonin_config.get("inhibition_strength", 0.5)
    )
    
    norepinephrine = NorepinephrineSystem(
        state=NeuromodulatorState(
            baseline=norepinephrine_config.get("baseline", 0.5),
            decay_rate=norepinephrine_config.get("decay_rate", 0.1)
        ),
        gain_modulation=norepinephrine_config.get("gain_modulation", 2.0)
    )
    
    return NeuromodulationSystem(
        dopamine=dopamine,
        serotonin=serotonin,
        norepinephrine=norepinephrine
    )
