"""Task and environment framework for 4D Neural Cognition.

This module provides a standardized interface for defining tasks and environments
that can be used to evaluate and train the neural network model.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional
import numpy as np


@dataclass
class TaskResult:
    """Result of a task evaluation."""
    
    accuracy: float = 0.0
    reward: float = 0.0
    reaction_time: float = 0.0
    stability: float = 0.0
    additional_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.additional_metrics is None:
            self.additional_metrics = {}


class Environment(ABC):
    """Abstract base class for task environments.
    
    Provides a standard interface for environments that the neural network
    can interact with. Similar to OpenAI Gym interface.
    """
    
    def __init__(self, seed: int = None):
        """Initialize the environment.
        
        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.current_step = 0
        self.episode_reward = 0.0
        
    @abstractmethod
    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment to initial state.
        
        Returns:
            observation: Dictionary mapping sense names to input arrays
            info: Additional information dictionary
        """
        pass
        
    @abstractmethod
    def step(
        self, 
        action: Optional[np.ndarray] = None
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Execute one step in the environment.
        
        Args:
            action: Action from the neural network (optional for some tasks)
            
        Returns:
            observation: Dictionary mapping sense names to input arrays
            reward: Reward signal
            done: Whether episode is finished
            info: Additional information dictionary
        """
        pass
        
    @abstractmethod
    def render(self) -> Optional[np.ndarray]:
        """Render the environment state.
        
        Returns:
            Optional visualization as numpy array
        """
        pass
        
    def get_info(self) -> Dict[str, Any]:
        """Get current environment information.
        
        Returns:
            Dictionary with environment metadata
        """
        return {
            'step': self.current_step,
            'episode_reward': self.episode_reward,
        }


class Task(ABC):
    """Abstract base class for tasks/benchmarks.
    
    A task wraps an environment and provides evaluation metrics.
    """
    
    def __init__(self, seed: int = None):
        """Initialize the task.
        
        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        self.env: Optional[Environment] = None
        
    @abstractmethod
    def get_name(self) -> str:
        """Get the task name."""
        pass
        
    @abstractmethod
    def get_description(self) -> str:
        """Get a description of the task."""
        pass
        
    @abstractmethod
    def evaluate(
        self, 
        simulation, 
        num_episodes: int = 10,
        max_steps: int = 1000
    ) -> TaskResult:
        """Evaluate a simulation on this task.
        
        Args:
            simulation: The simulation instance to evaluate
            num_episodes: Number of episodes to run
            max_steps: Maximum steps per episode
            
        Returns:
            TaskResult with evaluation metrics
        """
        pass
        
    @abstractmethod
    def get_metrics(self) -> Dict[str, str]:
        """Get description of metrics this task provides.
        
        Returns:
            Dictionary mapping metric names to descriptions
        """
        pass


class PatternClassificationTask(Task):
    """Simple pattern classification task.
    
    The network receives visual patterns and must classify them.
    Tests basic sensory processing and pattern recognition.
    """
    
    def __init__(
        self, 
        num_classes: int = 4,
        pattern_size: Tuple[int, int] = (20, 20),
        noise_level: float = 0.1,
        seed: int = None
    ):
        """Initialize pattern classification task.
        
        Args:
            num_classes: Number of pattern classes
            pattern_size: Size of input patterns
            noise_level: Amount of noise to add to patterns
            seed: Random seed
        """
        super().__init__(seed)
        self.num_classes = num_classes
        self.pattern_size = pattern_size
        self.noise_level = noise_level
        self.env = PatternClassificationEnvironment(
            num_classes=num_classes,
            pattern_size=pattern_size,
            noise_level=noise_level,
            seed=seed
        )
        
    def get_name(self) -> str:
        return f"PatternClassification-{self.num_classes}class"
        
    def get_description(self) -> str:
        return (
            f"Classify visual patterns into {self.num_classes} classes. "
            f"Pattern size: {self.pattern_size}, noise: {self.noise_level}"
        )
        
    def evaluate(
        self, 
        simulation, 
        num_episodes: int = 10,
        max_steps: int = 100
    ) -> TaskResult:
        """Evaluate simulation on pattern classification.
        
        Args:
            simulation: The simulation to evaluate
            num_episodes: Number of patterns to classify
            max_steps: Steps to run per pattern
            
        Returns:
            TaskResult with accuracy and other metrics
        """
        from .senses import feed_sense_input
        
        correct = 0
        total_reward = 0.0
        reaction_times = []
        
        for episode in range(num_episodes):
            observation, info = self.env.reset()
            target_class = info['target_class']
            
            # Feed pattern to vision
            feed_sense_input(simulation.model, 'vision', observation['vision'])
            
            # Run simulation and monitor output
            first_response_step = None
            predicted_class = None
            
            for step in range(max_steps):
                stats = simulation.step()
                
                # Check for output activity (simplified - would need proper output layer)
                if len(stats['spikes']) > 0 and first_response_step is None:
                    first_response_step = step
                    # Simplified classification - would need proper output decoding
                    predicted_class = self.rng.integers(0, self.num_classes)
                    
            # Evaluate response
            if predicted_class == target_class:
                correct += 1
                reward = 1.0
            else:
                reward = 0.0
                
            total_reward += reward
            
            if first_response_step is not None:
                reaction_times.append(first_response_step)
        
        accuracy = correct / num_episodes
        avg_reward = total_reward / num_episodes
        avg_reaction_time = np.mean(reaction_times) if reaction_times else max_steps
        
        return TaskResult(
            accuracy=accuracy,
            reward=avg_reward,
            reaction_time=avg_reaction_time,
            stability=1.0 - (np.std(reaction_times) / max_steps if reaction_times else 1.0),
            additional_metrics={
                'num_episodes': num_episodes,
                'num_responses': len(reaction_times)
            }
        )
        
    def get_metrics(self) -> Dict[str, str]:
        return {
            'accuracy': 'Classification accuracy (0-1)',
            'reward': 'Average reward per episode',
            'reaction_time': 'Average steps to first response',
            'stability': 'Response time stability (1 - normalized std)',
        }


class PatternClassificationEnvironment(Environment):
    """Environment for pattern classification task."""
    
    def __init__(
        self,
        num_classes: int = 4,
        pattern_size: Tuple[int, int] = (20, 20),
        noise_level: float = 0.1,
        seed: int = None
    ):
        super().__init__(seed)
        self.num_classes = num_classes
        self.pattern_size = pattern_size
        self.noise_level = noise_level
        
        # Generate base patterns for each class
        self.base_patterns = self._generate_base_patterns()
        self.current_pattern = None
        self.current_class = None
        
    def _generate_base_patterns(self) -> np.ndarray:
        """Generate distinct base patterns for each class."""
        patterns = np.zeros((self.num_classes, *self.pattern_size))
        
        for i in range(self.num_classes):
            # Create simple geometric patterns
            if i == 0:
                # Horizontal stripes
                patterns[i, ::2, :] = 1.0
            elif i == 1:
                # Vertical stripes
                patterns[i, :, ::2] = 1.0
            elif i == 2:
                # Diagonal pattern
                for j in range(min(self.pattern_size)):
                    if j < self.pattern_size[0] and j < self.pattern_size[1]:
                        patterns[i, j, j] = 1.0
            else:
                # Checkerboard
                patterns[i, ::2, ::2] = 1.0
                patterns[i, 1::2, 1::2] = 1.0
                
        return patterns
        
    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset with a new random pattern."""
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Select random class
        self.current_class = self.rng.integers(0, self.num_classes)
        
        # Get base pattern and add noise
        pattern = self.base_patterns[self.current_class].copy()
        noise = self.rng.normal(0, self.noise_level, pattern.shape)
        self.current_pattern = np.clip(pattern + noise, 0, 1)
        
        observation = {'vision': self.current_pattern}
        info = {'target_class': self.current_class}
        
        return observation, info
        
    def step(
        self, 
        action: Optional[np.ndarray] = None
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Step the environment (pattern classification is single-step)."""
        self.current_step += 1
        
        # Pattern classification is typically single-step
        done = True
        reward = 0.0  # Reward determined by task evaluator
        
        observation = {'vision': self.current_pattern}
        info = self.get_info()
        info['target_class'] = self.current_class
        
        return observation, reward, done, info
        
    def render(self) -> Optional[np.ndarray]:
        """Render current pattern."""
        return self.current_pattern


class TemporalSequenceTask(Task):
    """Temporal sequence learning task.
    
    The network receives sequences and must predict the next element.
    Tests temporal processing and memory capabilities.
    """
    
    def __init__(
        self,
        sequence_length: int = 5,
        vocabulary_size: int = 8,
        seed: int = None
    ):
        """Initialize temporal sequence task.
        
        Args:
            sequence_length: Length of sequences to learn
            vocabulary_size: Number of distinct elements
            seed: Random seed
        """
        super().__init__(seed)
        self.sequence_length = sequence_length
        self.vocabulary_size = vocabulary_size
        self.env = TemporalSequenceEnvironment(
            sequence_length=sequence_length,
            vocabulary_size=vocabulary_size,
            seed=seed
        )
        
    def get_name(self) -> str:
        return f"TemporalSequence-L{self.sequence_length}-V{self.vocabulary_size}"
        
    def get_description(self) -> str:
        return (
            f"Learn and predict temporal sequences of length {self.sequence_length} "
            f"from vocabulary of {self.vocabulary_size} elements"
        )
        
    def evaluate(
        self, 
        simulation, 
        num_episodes: int = 20,
        max_steps: int = 200
    ) -> TaskResult:
        """Evaluate simulation on sequence prediction.
        
        Args:
            simulation: The simulation to evaluate
            num_episodes: Number of sequences to test
            max_steps: Maximum steps per sequence
            
        Returns:
            TaskResult with prediction accuracy
        """
        from .senses import feed_sense_input
        
        correct_predictions = 0
        total_predictions = 0
        total_reward = 0.0
        
        for episode in range(num_episodes):
            observation, info = self.env.reset()
            done = False
            
            while not done and total_predictions < max_steps:
                # Feed current element via digital sense
                feed_sense_input(simulation.model, 'digital', observation['digital'])
                
                # Run simulation steps
                for _ in range(10):  # Process current input
                    simulation.step()
                
                # Get next observation
                observation, reward, done, info = self.env.step()
                
                # Simplified prediction evaluation
                # Would need proper output decoding in real implementation
                prediction_correct = self.rng.random() < 0.5  # Placeholder
                
                if prediction_correct:
                    correct_predictions += 1
                    
                total_predictions += 1
                total_reward += reward
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        avg_reward = total_reward / num_episodes
        
        return TaskResult(
            accuracy=accuracy,
            reward=avg_reward,
            reaction_time=0.0,
            stability=1.0,
            additional_metrics={
                'num_episodes': num_episodes,
                'total_predictions': total_predictions
            }
        )
        
    def get_metrics(self) -> Dict[str, str]:
        return {
            'accuracy': 'Sequence prediction accuracy (0-1)',
            'reward': 'Average reward per episode',
        }


class TemporalSequenceEnvironment(Environment):
    """Environment for temporal sequence task."""
    
    def __init__(
        self,
        sequence_length: int = 5,
        vocabulary_size: int = 8,
        seed: int = None
    ):
        super().__init__(seed)
        self.sequence_length = sequence_length
        self.vocabulary_size = vocabulary_size
        self.current_sequence = None
        self.sequence_position = 0
        
    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset with a new random sequence."""
        self.current_step = 0
        self.episode_reward = 0.0
        self.sequence_position = 0
        
        # Generate random sequence
        self.current_sequence = self.rng.integers(
            0, self.vocabulary_size, size=self.sequence_length
        )
        
        # Encode first element as one-hot vector
        element_encoding = np.zeros((20, 20))
        element_encoding[:self.vocabulary_size, 0] = 0.0
        element_encoding[self.current_sequence[0], 0] = 1.0
        
        observation = {'digital': element_encoding}
        info = {'sequence': self.current_sequence, 'position': 0}
        
        return observation, info
        
    def step(
        self, 
        action: Optional[np.ndarray] = None
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Step to next element in sequence."""
        self.current_step += 1
        self.sequence_position += 1
        
        done = self.sequence_position >= self.sequence_length
        reward = 0.0  # Would be based on prediction accuracy
        
        if not done:
            # Encode current element
            element_encoding = np.zeros((20, 20))
            element_encoding[:self.vocabulary_size, 0] = 0.0
            element_encoding[self.current_sequence[self.sequence_position], 0] = 1.0
            observation = {'digital': element_encoding}
        else:
            observation = {'digital': np.zeros((20, 20))}
        
        info = self.get_info()
        info['sequence'] = self.current_sequence
        info['position'] = self.sequence_position
        
        return observation, reward, done, info
        
    def render(self) -> Optional[np.ndarray]:
        """Render current sequence state."""
        return None
