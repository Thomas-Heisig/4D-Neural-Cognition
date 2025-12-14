"""Sensorimotor Reinforcement Learning for embodied AI.

This module implements the critical learning loop that connects action,
perception, and reward through STDP and neuromodulation, enabling the
system to learn from sensorimotor experience.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional
import numpy as np
import logging

if TYPE_CHECKING:
    from .virtual_body import VirtualBody
    from ..brain_model import BrainModel

logger = logging.getLogger(__name__)


class IntrinsicMotivation:
    """Intrinsic reward system for curiosity and competence-driven learning.
    
    Provides intrinsic rewards based on:
    1. Novelty/Surprise (curiosity)
    2. Learning progress (competence)
    3. Prediction error reduction (mastery)
    
    Attributes:
        novelty_weight: Weight for novelty-based rewards
        progress_weight: Weight for progress-based rewards
        history: Recent prediction errors for progress tracking
    """
    
    def __init__(
        self,
        novelty_weight: float = 0.3,
        progress_weight: float = 0.7,
        history_size: int = 100,
    ):
        """Initialize intrinsic motivation system.
        
        Args:
            novelty_weight: Weight for curiosity-driven rewards
            progress_weight: Weight for progress-driven rewards
            history_size: Number of recent errors to track
        """
        self.novelty_weight = novelty_weight
        self.progress_weight = progress_weight
        self.prediction_error_history: List[float] = []
        self.history_size = history_size
        
        # Track state visitation for novelty
        self.state_visits: Dict[str, int] = {}
        
        logger.info(
            f"Initialized IntrinsicMotivation "
            f"(novelty={novelty_weight}, progress={progress_weight})"
        )
    
    def calculate_reward(
        self,
        state: np.ndarray,
        prediction_error: float,
    ) -> float:
        """Calculate intrinsic reward based on novelty and progress.
        
        Args:
            state: Current state representation
            prediction_error: Magnitude of prediction error
            
        Returns:
            Intrinsic reward value
        """
        # Novelty reward (inversely proportional to visit count)
        state_key = self._state_to_key(state)
        visit_count = self.state_visits.get(state_key, 0)
        self.state_visits[state_key] = visit_count + 1
        novelty_reward = 1.0 / (1.0 + visit_count)
        
        # Progress reward (based on error reduction)
        self.prediction_error_history.append(prediction_error)
        if len(self.prediction_error_history) > self.history_size:
            self.prediction_error_history.pop(0)
        
        progress_reward = 0.0
        if len(self.prediction_error_history) >= 10:
            # Compare recent errors to earlier errors
            recent_avg = np.mean(self.prediction_error_history[-10:])
            earlier_avg = np.mean(self.prediction_error_history[:10])
            if earlier_avg > 0:
                progress_reward = max(0.0, (earlier_avg - recent_avg) / earlier_avg)
        
        # Combined intrinsic reward
        intrinsic_reward = (
            self.novelty_weight * novelty_reward +
            self.progress_weight * progress_reward
        )
        
        return intrinsic_reward
    
    def _state_to_key(self, state: np.ndarray) -> str:
        """Convert state to hashable key for visit tracking.
        
        Args:
            state: State array
            
        Returns:
            String key representing discretized state
        """
        # Discretize state to reduce dimensionality
        discretized = np.round(state * 10).astype(int)
        return str(discretized.tolist())
    
    def reset(self) -> None:
        """Reset intrinsic motivation state."""
        self.prediction_error_history = []
        self.state_visits = {}
        logger.info("Intrinsic motivation reset")


class SensorimotorReinforcementLearner:
    """Reinforcement learning system for sensorimotor integration.
    
    This class implements the critical connection between body experience
    and neural plasticity. It:
    1. Encodes proprioceptive feedback into neural activity
    2. Applies dopamine-like neuromodulation based on rewards
    3. Uses STDP to strengthen successful sensorimotor associations
    4. Tracks learning progress over episodes
    
    This enables the system to learn motor skills through trial and error,
    developing a body schema and sense of agency.
    
    Attributes:
        body: Virtual body being controlled
        brain: Brain model with neural substrate
        intrinsic_reward_system: Intrinsic motivation calculator
        learning_rate: Base learning rate for plasticity
        discount_factor: Temporal credit assignment factor
        episode_count: Number of learning episodes completed
    """
    
    def __init__(
        self,
        virtual_body: VirtualBody,
        brain_model: BrainModel,
        learning_rate: float = 0.01,
        discount_factor: float = 0.95,
        dopamine_modulation_strength: float = 0.1,
    ):
        """Initialize sensorimotor reinforcement learner.
        
        Args:
            virtual_body: The body to control
            brain_model: The brain model
            learning_rate: Learning rate for STDP
            discount_factor: Discount factor for temporal credit
            dopamine_modulation_strength: Strength of neuromodulation
        """
        self.body = virtual_body
        self.brain = brain_model
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.dopamine_strength = dopamine_modulation_strength
        
        # Intrinsic motivation system
        self.intrinsic_reward_system = IntrinsicMotivation()
        
        # Learning statistics
        self.episode_count = 0
        self.total_reward_history: List[float] = []
        self.prediction_error_history: List[float] = []
        self.learning_progress: List[Dict] = []
        
        # Track motor neurons (w-slice 10 by convention)
        self.motor_w_slice = 10
        self.sensory_w_slice = 6  # Sensor fusion area
        
        logger.info(
            f"Initialized SensorimotorReinforcementLearner "
            f"(lr={learning_rate}, gamma={discount_factor})"
        )
    
    def learn_from_interaction(
        self,
        action: Dict,
        resulting_feedback: Dict,
        external_reward: float = 0.0,
    ) -> Dict:
        """Learn from sensorimotor interaction (key learning loop).
        
        This is the critical connection between body experience and neural
        plasticity. It implements:
        1. Encoding proprioceptive feedback into neural patterns
        2. Applying dopamine-like neuromodulation for rewards
        3. STDP updates between motor and sensory neurons
        4. Learning progress tracking
        
        Args:
            action: Motor action that was executed
            resulting_feedback: Proprioceptive and sensory feedback
            external_reward: Task-specific external reward
            
        Returns:
            Dictionary with learning metrics
        """
        # 1. Encode proprioceptive feedback into neural activity
        sensory_neural_pattern = self.encode_proprioception_to_neurons(
            resulting_feedback
        )
        
        # 2. Get involved motor neurons (w-slice 10, M1 area)
        motor_neurons = self.get_neurons_in_slice(
            w=self.motor_w_slice,
            area='M1'
        )
        
        # 3. Calculate prediction error for intrinsic reward
        predicted_feedback = self._predict_feedback(action)
        prediction_error = self._calculate_prediction_error(
            predicted_feedback,
            resulting_feedback
        )
        
        # Get intrinsic reward
        body_state_vector = self._feedback_to_vector(resulting_feedback)
        intrinsic_reward = self.intrinsic_reward_system.calculate_reward(
            body_state_vector,
            prediction_error
        )
        
        # Total reward
        total_reward = external_reward + intrinsic_reward
        
        # 4. Apply dopamine-like neuromodulation if reward is positive
        if total_reward > 0:
            self.apply_neuromodulation(
                target_neurons=motor_neurons,
                modulator='dopamine',
                strength=total_reward * self.dopamine_strength
            )
        
        # 5. STDP update between motor and sensory neurons
        self.stdp_update(
            pre_neurons=motor_neurons,
            post_pattern=sensory_neural_pattern,
            time_delta=1  # Action precedes sensory feedback
        )
        
        # 6. Track learning progress
        progress = self.calculate_learning_progress()
        
        # Update histories
        self.total_reward_history.append(total_reward)
        self.prediction_error_history.append(prediction_error)
        self.learning_progress.append(progress)
        
        return {
            'total_reward': total_reward,
            'external_reward': external_reward,
            'intrinsic_reward': intrinsic_reward,
            'prediction_error': prediction_error,
            'learning_progress': progress,
            'motor_neurons_updated': len(motor_neurons),
        }
    
    def encode_proprioception_to_neurons(
        self,
        feedback: Dict
    ) -> Dict[int, float]:
        """Encode proprioceptive feedback into neural activity pattern.
        
        Maps body state (joint angles, muscle tensions) to neural activations
        in sensory cortex (w-slice 6).
        
        Args:
            feedback: Proprioceptive feedback from body
            
        Returns:
            Dictionary mapping neuron IDs to activation levels
        """
        neural_pattern = {}
        
        # Get sensory neurons
        sensory_neurons = self.get_neurons_in_slice(
            w=self.sensory_w_slice,
            area='S1'  # Primary somatosensory cortex
        )
        
        # Extract body state features
        joint_angles = feedback.get('joint_angles', {})
        muscle_tensions = feedback.get('muscle_tensions', {})
        
        # Simple encoding: map each joint/muscle to neurons
        features = []
        for angle in joint_angles.values():
            features.append(angle)
        for tension in muscle_tensions.values():
            features.append(tension)
        
        # Normalize features
        if features:
            features = np.array(features)
            features = (features - features.mean()) / (features.std() + 1e-6)
            
            # Map to neurons (circular mapping)
            for i, feature_val in enumerate(features):
                if i < len(sensory_neurons):
                    neuron_id = sensory_neurons[i]
                    # Convert to activation level (0-1)
                    activation = 1.0 / (1.0 + np.exp(-feature_val))
                    neural_pattern[neuron_id] = activation
        
        return neural_pattern
    
    def get_neurons_in_slice(
        self,
        w: int,
        area: Optional[str] = None
    ) -> List[int]:
        """Get neuron IDs in specified w-slice and area.
        
        Args:
            w: W-coordinate of slice
            area: Optional brain area filter
            
        Returns:
            List of neuron IDs
        """
        neurons = []
        
        for neuron_id, neuron in self.brain.neurons.items():
            if neuron.w == w:
                # Check area if specified
                if area is None or getattr(neuron, 'area', None) == area:
                    neurons.append(neuron_id)
        
        return neurons
    
    def apply_neuromodulation(
        self,
        target_neurons: List[int],
        modulator: str,
        strength: float,
    ) -> None:
        """Apply neuromodulation to target neurons (dopamine-like).
        
        Simulates dopaminergic reward signals that enhance plasticity
        and excitability of recently active neurons.
        
        Args:
            target_neurons: List of neuron IDs to modulate
            modulator: Type of modulator ('dopamine', 'serotonin', etc.)
            strength: Modulation strength (0-1)
        """
        if modulator == 'dopamine':
            # Dopamine increases excitability and learning rate
            for neuron_id in target_neurons:
                if neuron_id in self.brain.neurons:
                    neuron = self.brain.neurons[neuron_id]
                    
                    # Increase excitability (reduce threshold temporarily)
                    if hasattr(neuron, 'params') and 'threshold' in neuron.params:
                        neuron.params['threshold'] *= (1.0 - strength * 0.1)
                    
                    # Mark for enhanced plasticity
                    if not hasattr(neuron, 'plasticity_tag'):
                        neuron.plasticity_tag = 0.0
                    neuron.plasticity_tag += strength
    
    def stdp_update(
        self,
        pre_neurons: List[int],
        post_pattern: Dict[int, float],
        time_delta: int = 1,
    ) -> None:
        """Apply STDP between pre-synaptic and post-synaptic neurons.
        
        Spike-timing-dependent plasticity: strengthens connections when
        pre-synaptic spikes precede post-synaptic activity.
        
        Args:
            pre_neurons: Pre-synaptic neuron IDs (motor neurons)
            post_pattern: Post-synaptic activation pattern (sensory)
            time_delta: Time difference (positive = pre before post)
        """
        # STDP parameters
        tau_plus = 20.0  # ms
        tau_minus = 20.0  # ms
        A_plus = self.learning_rate
        A_minus = -self.learning_rate * 0.5
        
        # Calculate STDP weight change
        if time_delta > 0:
            # Pre before post: potentiation
            stdp_factor = A_plus * np.exp(-time_delta / tau_plus)
        else:
            # Post before pre: depression
            stdp_factor = A_minus * np.exp(time_delta / tau_minus)
        
        # Apply to synapses
        for pre_id in pre_neurons:
            for post_id, activation in post_pattern.items():
                # Find or create synapse
                synapse = self._find_synapse(pre_id, post_id)
                
                if synapse:
                    # Update weight based on STDP and activation
                    weight_change = stdp_factor * activation
                    
                    # Apply plasticity tag if present
                    if pre_id in self.brain.neurons:
                        pre_neuron = self.brain.neurons[pre_id]
                        if hasattr(pre_neuron, 'plasticity_tag'):
                            weight_change *= (1.0 + pre_neuron.plasticity_tag)
                    
                    synapse.weight += weight_change
                    
                    # Clamp weights to valid range
                    synapse.weight = np.clip(synapse.weight, 0.0, 1.0)
    
    def _find_synapse(self, pre_id: int, post_id: int) -> Optional[object]:
        """Find synapse between two neurons.
        
        Args:
            pre_id: Pre-synaptic neuron ID
            post_id: Post-synaptic neuron ID
            
        Returns:
            Synapse object if found, None otherwise
        """
        # Check if using sparse connectivity
        if hasattr(self.brain, '_sparse_synapses') and self.brain._sparse_synapses:
            # Sparse connectivity mode
            weight = self.brain._sparse_synapses.get_weight(pre_id, post_id)
            if weight is not None:
                # Return a simple object with weight attribute
                class SynapseProxy:
                    def __init__(self, sparse_matrix, pre, post, w):
                        self.sparse_matrix = sparse_matrix
                        self.pre_id = pre
                        self.post_id = post
                        self._weight = w
                    
                    @property
                    def weight(self):
                        return self._weight
                    
                    @weight.setter
                    def weight(self, value):
                        self._weight = value
                        self.sparse_matrix.set_weight(
                            self.pre_id, self.post_id, value
                        )
                
                return SynapseProxy(
                    self.brain._sparse_synapses, pre_id, post_id, weight
                )
        else:
            # List-based synapses
            for synapse in self.brain.synapses:
                if synapse.pre_id == pre_id and synapse.post_id == post_id:
                    return synapse
        
        return None
    
    def _predict_feedback(self, action: Dict) -> Dict:
        """Predict sensory feedback from action (forward model).
        
        Args:
            action: Motor action
            
        Returns:
            Predicted feedback
        """
        # Placeholder: would use learned forward model
        # For now, return empty prediction
        return {
            'joint_angles': {},
            'muscle_tensions': {},
        }
    
    def _calculate_prediction_error(
        self,
        predicted: Dict,
        actual: Dict
    ) -> float:
        """Calculate prediction error magnitude.
        
        Args:
            predicted: Predicted feedback
            actual: Actual feedback
            
        Returns:
            Prediction error (0-1)
        """
        errors = []
        
        # Compare joint angles
        pred_angles = predicted.get('joint_angles', {})
        actual_angles = actual.get('joint_angles', {})
        
        for joint_id in actual_angles:
            if joint_id in pred_angles:
                error = abs(pred_angles[joint_id] - actual_angles[joint_id])
                errors.append(error)
        
        # Compare muscle tensions
        pred_tensions = predicted.get('muscle_tensions', {})
        actual_tensions = actual.get('muscle_tensions', {})
        
        for muscle_id in actual_tensions:
            if muscle_id in pred_tensions:
                error = abs(pred_tensions[muscle_id] - actual_tensions[muscle_id])
                errors.append(error)
        
        if errors:
            return float(np.mean(errors))
        return 0.0
    
    def _feedback_to_vector(self, feedback: Dict) -> np.ndarray:
        """Convert feedback to vector for state representation.
        
        Args:
            feedback: Feedback dictionary
            
        Returns:
            State vector
        """
        features = []
        
        for angle in feedback.get('joint_angles', {}).values():
            features.append(angle)
        for tension in feedback.get('muscle_tensions', {}).values():
            features.append(tension)
        
        return np.array(features) if features else np.zeros(1)
    
    def calculate_learning_progress(self) -> Dict:
        """Calculate learning progress metrics.
        
        Returns:
            Dictionary with progress metrics
        """
        if len(self.total_reward_history) < 10:
            return {
                'episodes': self.episode_count,
                'avg_reward': 0.0,
                'reward_trend': 0.0,
                'avg_prediction_error': 0.0,
                'error_reduction': 0.0,
            }
        
        # Recent average reward
        recent_rewards = self.total_reward_history[-10:]
        avg_reward = np.mean(recent_rewards)
        
        # Reward trend (compare recent to earlier)
        if len(self.total_reward_history) >= 20:
            earlier_rewards = self.total_reward_history[-20:-10]
            reward_trend = avg_reward - np.mean(earlier_rewards)
        else:
            reward_trend = 0.0
        
        # Prediction error reduction
        recent_errors = self.prediction_error_history[-10:]
        avg_error = np.mean(recent_errors)
        
        if len(self.prediction_error_history) >= 20:
            earlier_errors = self.prediction_error_history[-20:-10]
            error_reduction = np.mean(earlier_errors) - avg_error
        else:
            error_reduction = 0.0
        
        return {
            'episodes': self.episode_count,
            'avg_reward': float(avg_reward),
            'reward_trend': float(reward_trend),
            'avg_prediction_error': float(avg_error),
            'error_reduction': float(error_reduction),
        }
    
    def start_episode(self) -> None:
        """Start new learning episode."""
        self.episode_count += 1
        logger.debug(f"Starting episode {self.episode_count}")
    
    def end_episode(self) -> Dict:
        """End current episode and return summary.
        
        Returns:
            Episode summary
        """
        progress = self.calculate_learning_progress()
        logger.info(
            f"Episode {self.episode_count} complete: "
            f"avg_reward={progress['avg_reward']:.3f}, "
            f"avg_error={progress['avg_prediction_error']:.3f}"
        )
        return progress
    
    def reset(self) -> None:
        """Reset learner state."""
        self.episode_count = 0
        self.total_reward_history = []
        self.prediction_error_history = []
        self.learning_progress = []
        self.intrinsic_reward_system.reset()
        logger.info("Sensorimotor learner reset")
    
    def get_statistics(self) -> Dict:
        """Get learning statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'episodes': self.episode_count,
            'total_interactions': len(self.total_reward_history),
            'current_progress': self.calculate_learning_progress(),
            'reward_history': self.total_reward_history[-100:],
            'error_history': self.prediction_error_history[-100:],
        }
