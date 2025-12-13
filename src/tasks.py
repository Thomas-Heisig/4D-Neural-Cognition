"""Task and environment framework for 4D Neural Cognition.

This module provides a standardized interface for defining tasks and environments
that can be used to evaluate and train the neural network model.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class TaskResult:
    """Result of a task evaluation."""

    accuracy: float = 0.0
    reward: float = 0.0
    reaction_time: float = 0.0
    stability: float = 0.0
    additional_metrics: Dict[str, float] = None

    def __post_init__(self) -> None:
        """Initialize additional_metrics dict if not provided."""
        if self.additional_metrics is None:
            self.additional_metrics = {}


class Environment(ABC):
    """Abstract base class for task environments.

    Provides a standard interface for environments that the neural network
    can interact with. Similar to OpenAI Gym interface.
    """

    def __init__(self, seed: int = None) -> None:
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
    def step(self, action: Optional[np.ndarray] = None) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
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
            "step": self.current_step,
            "episode_reward": self.episode_reward,
        }


class Task(ABC):
    """Abstract base class for tasks/benchmarks.

    A task wraps an environment and provides evaluation metrics.
    """

    def __init__(self, seed: int = None) -> None:
        """Initialize the task.

        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
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
    def evaluate(self, simulation, num_episodes: int = 10, max_steps: int = 1000) -> TaskResult:
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

    NOTE: Current implementation uses placeholder prediction logic (random).
    A full implementation would require dedicated output neurons and proper
    decoding logic. This serves as a framework demonstration and can be
    extended with actual classification when output layers are implemented.
    """

    def __init__(
        self, num_classes: int = 4, pattern_size: Tuple[int, int] = (20, 20), noise_level: float = 0.1, seed: int = None
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
            num_classes=num_classes, pattern_size=pattern_size, noise_level=noise_level, seed=seed
        )

    def get_name(self) -> str:
        """Get the name of the task.

        Returns:
            str: Task name with number of classes
        """
        return f"PatternClassification-{self.num_classes}class"

    def get_description(self) -> str:
        """Get a human-readable description of the task.

        Returns:
            str: Description including task parameters
        """
        return (
            f"Classify visual patterns into {self.num_classes} classes. "
            f"Pattern size: {self.pattern_size}, noise: {self.noise_level}"
        )

    def _decode_output_from_firing_rates(self, simulation, num_classes: int, 
                                          observation_window: int = 10) -> Optional[int]:
        """Decode output class from neural firing rates.
        
        Uses firing rates of neurons to determine predicted class. This looks at
        recent spike history and assigns neurons to output classes based on their
        position, then uses winner-takes-all to determine the prediction.
        
        Args:
            simulation: The simulation to decode output from
            num_classes: Number of output classes
            observation_window: Number of recent timesteps to consider
            
        Returns:
            Predicted class (0 to num_classes-1) or None if no activity
        """
        model = simulation.model
        
        # Count recent spikes per neuron
        spike_counts = {}
        for neuron_id, times in simulation.spike_history.items():
            if neuron_id in model.neurons:
                # Count spikes in recent observation window
                recent_spikes = sum(1 for t in times if t >= simulation.current_step - observation_window)
                if recent_spikes > 0:
                    spike_counts[neuron_id] = recent_spikes
        
        if not spike_counts:
            return None
        
        # Assign neurons to output classes based on neuron ID modulo num_classes
        # This creates a distributed output representation
        class_firing_rates = [0.0] * num_classes
        for neuron_id, count in spike_counts.items():
            class_idx = neuron_id % num_classes
            class_firing_rates[class_idx] += count
        
        # Winner-takes-all: class with highest firing rate
        predicted_class = int(np.argmax(class_firing_rates))
        
        return predicted_class
    
    def evaluate(self, simulation, num_episodes: int = 10, max_steps: int = 100) -> TaskResult:
        """Evaluate simulation on pattern classification.

        Args:
            simulation: The simulation to evaluate
            num_episodes: Number of patterns to classify
            max_steps: Steps to run per pattern

        Returns:
            TaskResult with accuracy and other metrics
        """
        try:
            from .senses import feed_sense_input
        except ImportError:
            from senses import feed_sense_input

        correct = 0
        total_reward = 0.0
        reaction_times = []

        for episode in range(num_episodes):
            observation, info = self.env.reset()
            target_class = info["target_class"]

            # Feed pattern to vision
            feed_sense_input(simulation.model, "vision", observation["vision"])

            # Run simulation and monitor output
            first_response_step = None
            predicted_class = None

            for step in range(max_steps):
                stats = simulation.step()

                # Decode output from neural firing rates using winner-takes-all
                # Output is decoded from distributed neural activity
                if len(stats["spikes"]) > 0 and first_response_step is None:
                    first_response_step = step
                    # Decode prediction from neural firing rates
                    predicted_class = self._decode_output_from_firing_rates(
                        simulation, self.num_classes, observation_window=10
                    )
                    
                    # Fallback to random if no clear output
                    if predicted_class is None:
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
            additional_metrics={"num_episodes": num_episodes, "num_responses": len(reaction_times)},
        )

    def get_metrics(self) -> Dict[str, str]:
        """Get descriptions of the metrics used in this task.

        Returns:
            Dict[str, str]: Dictionary mapping metric names to descriptions
        """
        return {
            "accuracy": "Classification accuracy (0-1)",
            "reward": "Average reward per episode",
            "reaction_time": "Average steps to first response",
            "stability": "Response time stability (1 - normalized std)",
        }


class PatternClassificationEnvironment(Environment):
    """Environment for pattern classification task."""

    def __init__(
        self, num_classes: int = 4, pattern_size: Tuple[int, int] = (20, 20), noise_level: float = 0.1, seed: int = None
    ):
        """Initialize pattern classification environment.

        Args:
            num_classes: Number of pattern classes to classify.
            pattern_size: Size of patterns as (height, width).
            noise_level: Amount of noise to add to patterns (0.0 to 1.0).
            seed: Random seed for reproducibility.
        """
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

        observation = {"vision": self.current_pattern}
        info = {"target_class": self.current_class}

        return observation, info

    def step(self, action: Optional[np.ndarray] = None) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Step the environment (pattern classification is single-step)."""
        self.current_step += 1

        # Pattern classification is typically single-step
        done = True
        reward = 0.0  # Reward determined by task evaluator

        observation = {"vision": self.current_pattern}
        info = self.get_info()
        info["target_class"] = self.current_class

        return observation, reward, done, info

    def render(self) -> Optional[np.ndarray]:
        """Render current pattern."""
        return self.current_pattern


class TemporalSequenceTask(Task):
    """Temporal sequence learning task.

    The network receives sequences and must predict the next element.
    Tests temporal processing and memory capabilities.

    NOTE: Current implementation uses placeholder prediction evaluation (random).
    A full implementation would require output neurons that predict the next
    sequence element and comparison logic. This serves as a framework
    demonstration and can be extended with actual prediction logic.
    """

    def __init__(self, sequence_length: int = 5, vocabulary_size: int = 8, seed: int = None) -> None:
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
            sequence_length=sequence_length, vocabulary_size=vocabulary_size, seed=seed
        )

    def get_name(self) -> str:
        """Get the name of the task.

        Returns:
            str: Task name with sequence length and vocabulary size
        """
        return f"TemporalSequence-L{self.sequence_length}-V{self.vocabulary_size}"

    def get_description(self) -> str:
        """Get a human-readable description of the task.

        Returns:
            str: Description including task parameters
        """
        return (
            f"Learn and predict temporal sequences of length {self.sequence_length} "
            f"from vocabulary of {self.vocabulary_size} elements"
        )

    def _predict_next_element(self, simulation, vocabulary_size: int, 
                              observation_window: int = 10) -> Optional[int]:
        """Predict next sequence element from neural activity.
        
        Uses recent firing patterns to predict the next element in the sequence.
        Neurons are assigned to vocabulary elements and the element with highest
        associated firing rate is predicted.
        
        Args:
            simulation: The simulation to decode output from
            vocabulary_size: Size of the vocabulary
            observation_window: Number of recent timesteps to consider
            
        Returns:
            Predicted element (0 to vocabulary_size-1) or None if no activity
        """
        model = simulation.model
        
        # Count recent spikes per neuron
        spike_counts = {}
        for neuron_id, times in simulation.spike_history.items():
            if neuron_id in model.neurons:
                # Count spikes in recent observation window
                recent_spikes = sum(1 for t in times if t >= simulation.current_step - observation_window)
                if recent_spikes > 0:
                    spike_counts[neuron_id] = recent_spikes
        
        if not spike_counts:
            return None
        
        # Assign neurons to vocabulary elements based on neuron ID
        element_scores = [0.0] * vocabulary_size
        for neuron_id, count in spike_counts.items():
            element_idx = neuron_id % vocabulary_size
            element_scores[element_idx] += count
        
        # Return element with highest score
        predicted_element = int(np.argmax(element_scores))
        
        return predicted_element
    
    def evaluate(self, simulation, num_episodes: int = 20, max_steps: int = 200) -> TaskResult:
        """Evaluate simulation on sequence prediction.

        Args:
            simulation: The simulation to evaluate
            num_episodes: Number of sequences to test
            max_steps: Maximum steps per sequence

        Returns:
            TaskResult with prediction accuracy
        """
        try:
            from .senses import feed_sense_input
        except ImportError:
            from senses import feed_sense_input

        correct_predictions = 0
        total_predictions = 0
        total_reward = 0.0
        previous_element = None

        for episode in range(num_episodes):
            observation, info = self.env.reset()
            done = False
            previous_element = None

            while not done and total_predictions < max_steps:
                # Feed current element via digital sense
                feed_sense_input(simulation.model, "digital", observation["digital"])

                # Run simulation steps
                for _ in range(10):  # Process current input
                    simulation.step()

                # Predict next element based on neural activity
                if previous_element is not None:
                    predicted_next = self._predict_next_element(
                        simulation, self.vocabulary_size, observation_window=10
                    )
                    
                    # Fallback if no clear prediction
                    if predicted_next is None:
                        predicted_next = self.rng.integers(0, self.vocabulary_size)
                    
                    # Get actual next element from info
                    current_element = info.get("current_element", 0)
                    
                    # Compare prediction with actual next element
                    prediction_correct = (predicted_next == current_element)
                    
                    if prediction_correct:
                        correct_predictions += 1
                    
                    total_predictions += 1

                # Store current element for next prediction
                previous_element = info.get("current_element", 0)

                # Get next observation
                observation, reward, done, info = self.env.step()
                total_reward += reward

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        avg_reward = total_reward / num_episodes

        return TaskResult(
            accuracy=accuracy,
            reward=avg_reward,
            reaction_time=0.0,
            stability=1.0,
            additional_metrics={"num_episodes": num_episodes, "total_predictions": total_predictions},
        )

    def get_metrics(self) -> Dict[str, str]:
        """Get descriptions of the metrics used in this task.

        Returns:
            Dict[str, str]: Dictionary mapping metric names to descriptions
        """
        return {
            "accuracy": "Sequence prediction accuracy (0-1)",
            "reward": "Average reward per episode",
        }


class TemporalSequenceEnvironment(Environment):
    """Environment for temporal sequence task."""

    def __init__(self, sequence_length: int = 5, vocabulary_size: int = 8, seed: int = None) -> None:
        """Initialize temporal sequence environment.

        Args:
            sequence_length: Length of sequences to generate.
            vocabulary_size: Number of unique symbols in vocabulary.
            seed: Random seed for reproducibility.
        """
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
        self.current_sequence = self.rng.integers(0, self.vocabulary_size, size=self.sequence_length)

        # Encode first element as one-hot vector
        element_encoding = np.zeros((20, 20))
        element_encoding[self.current_sequence[0], 0] = 1.0

        observation = {"digital": element_encoding}
        info = {"sequence": self.current_sequence, "position": 0}

        return observation, info

    def step(self, action: Optional[np.ndarray] = None) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Step to next element in sequence."""
        self.current_step += 1
        self.sequence_position += 1

        done = self.sequence_position >= self.sequence_length
        reward = 0.0  # Would be based on prediction accuracy

        if not done:
            # Encode current element
            element_encoding = np.zeros((20, 20))
            element_encoding[self.current_sequence[self.sequence_position], 0] = 1.0
            observation = {"digital": element_encoding}
        else:
            observation = {"digital": np.zeros((20, 20))}

        info = self.get_info()
        info["sequence"] = self.current_sequence
        info["position"] = self.sequence_position

        return observation, reward, done, info

    def render(self) -> Optional[np.ndarray]:
        """Render current sequence state."""
        return None


class SensorimotorControlTask(Task):
    """Sensorimotor control task - pendulum stabilization.
    
    The network must learn to balance a pendulum by applying appropriate forces.
    Tests continuous control, sensorimotor integration, and real-time learning.
    """

    def __init__(
        self, 
        max_angle: float = np.pi / 4,
        control_interval: int = 10,
        seed: int = None
    ) -> None:
        """Initialize sensorimotor control task.
        
        Args:
            max_angle: Maximum allowed angle deviation (radians)
            control_interval: Steps between control actions
            seed: Random seed
        """
        super().__init__(seed)
        self.max_angle = max_angle
        self.control_interval = control_interval
        self.env = SensorimotorControlEnvironment(
            max_angle=max_angle,
            seed=seed
        )

    def get_name(self) -> str:
        """Get the task name."""
        return f"SensorimotorControl-Pendulum"

    def get_description(self) -> str:
        """Get task description."""
        return (
            f"Balance a pendulum within {self.max_angle:.2f} radians. "
            f"Control interval: {self.control_interval} steps"
        )

    def evaluate(self, simulation, num_episodes: int = 10, max_steps: int = 500) -> TaskResult:
        """Evaluate simulation on pendulum control.
        
        Args:
            simulation: The simulation to evaluate
            num_episodes: Number of trials
            max_steps: Maximum steps per trial
            
        Returns:
            TaskResult with control performance metrics
        """
        try:
            from .senses import feed_sense_input
        except ImportError:
            from senses import feed_sense_input

        total_reward = 0.0
        total_time_balanced = 0
        successful_episodes = 0
        reaction_times = []

        for episode in range(num_episodes):
            observation, info = self.env.reset()
            done = False
            episode_reward = 0.0
            balanced_start_step = None
            steps = 0

            while not done and steps < max_steps:
                # Feed pendulum state via vision (angle and velocity visualization)
                feed_sense_input(simulation.model, "vision", observation["vision"])
                
                # Run simulation to generate control signal
                for _ in range(self.control_interval):
                    stats = simulation.step()
                    
                # Extract control action from network activity (simplified)
                # In real implementation, would use dedicated motor output neurons
                if len(stats["spikes"]) > 0:
                    action = np.tanh(len(stats["spikes"]) / 10.0 - 0.5)
                else:
                    action = 0.0
                
                observation, reward, done, info = self.env.step(np.array([action]))
                episode_reward += reward
                steps += 1
                
                # Track when pendulum becomes balanced
                if info.get("balanced", False) and balanced_start_step is None:
                    balanced_start_step = steps
                
                if info.get("balanced", False):
                    total_time_balanced += 1

            total_reward += episode_reward
            
            if episode_reward > 0:
                successful_episodes += 1
                
            if balanced_start_step is not None:
                reaction_times.append(balanced_start_step)

        avg_reward = total_reward / num_episodes
        success_rate = successful_episodes / num_episodes
        avg_reaction_time = np.mean(reaction_times) if reaction_times else max_steps
        stability = total_time_balanced / (num_episodes * max_steps)

        return TaskResult(
            accuracy=success_rate,
            reward=avg_reward,
            reaction_time=avg_reaction_time,
            stability=stability,
            additional_metrics={
                "num_episodes": num_episodes,
                "successful_episodes": successful_episodes,
                "time_balanced": total_time_balanced,
            },
        )

    def get_metrics(self) -> Dict[str, str]:
        """Get metric descriptions."""
        return {
            "accuracy": "Success rate (pendulum kept balanced)",
            "reward": "Average reward per episode",
            "reaction_time": "Steps to achieve balance",
            "stability": "Fraction of time balanced",
        }


class SensorimotorControlEnvironment(Environment):
    """Environment for pendulum control task."""

    def __init__(self, max_angle: float = np.pi / 4, seed: int = None) -> None:
        """Initialize pendulum environment.
        
        Args:
            max_angle: Maximum allowed angle deviation
            seed: Random seed
        """
        super().__init__(seed)
        self.max_angle = max_angle
        self.angle = 0.0
        self.velocity = 0.0
        self.gravity = 9.81
        self.length = 1.0
        self.dt = 0.05

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset pendulum to random initial state."""
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Random initial angle and velocity
        self.angle = self.rng.uniform(-self.max_angle / 2, self.max_angle / 2)
        self.velocity = self.rng.uniform(-0.5, 0.5)
        
        observation = self._get_observation()
        info = self._get_state_info()
        
        return observation, info

    def step(self, action: Optional[np.ndarray] = None) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Update pendulum physics."""
        self.current_step += 1
        
        # Apply control force
        force = action[0] if action is not None else 0.0
        force = np.clip(force, -1.0, 1.0)
        
        # Update physics
        acceleration = -(self.gravity / self.length) * np.sin(self.angle) + force
        self.velocity += acceleration * self.dt
        self.velocity *= 0.99  # Damping
        self.angle += self.velocity * self.dt
        
        # Check if balanced
        balanced = abs(self.angle) < self.max_angle
        
        # Calculate reward
        if balanced:
            reward = 1.0 - abs(self.angle) / self.max_angle
        else:
            reward = -1.0
            
        self.episode_reward += reward
        
        done = not balanced or self.current_step > 1000
        
        observation = self._get_observation()
        info = self._get_state_info()
        info["balanced"] = balanced
        
        return observation, reward, done, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Create visual representation of pendulum state."""
        # Create 20x20 visualization
        vis = np.zeros((20, 20))
        
        # Draw pendulum angle (position indicator)
        angle_pos = int(10 + 8 * np.sin(self.angle))
        angle_pos = np.clip(angle_pos, 0, 19)
        vis[10, angle_pos] = 1.0
        
        # Draw velocity indicator
        vel_pos = int(10 + 8 * np.tanh(self.velocity))
        vel_pos = np.clip(vel_pos, 0, 19)
        vis[15, vel_pos] = 0.5
        
        return {"vision": vis}

    def _get_state_info(self) -> Dict[str, Any]:
        """Get current state information."""
        info = self.get_info()
        info["angle"] = self.angle
        info["velocity"] = self.velocity
        return info

    def render(self) -> Optional[np.ndarray]:
        """Render pendulum visualization."""
        return self._get_observation()["vision"]


class MultiModalIntegrationTask(Task):
    """Multi-modal integration task.
    
    The network receives information from multiple sensory modalities
    and must integrate them to make correct decisions.
    Tests cross-modal processing and sensory fusion.
    """

    def __init__(
        self, 
        num_classes: int = 4,
        modality_noise: float = 0.2,
        seed: int = None
    ) -> None:
        """Initialize multi-modal integration task.
        
        Args:
            num_classes: Number of classes to distinguish
            modality_noise: Noise level for individual modalities
            seed: Random seed
        """
        super().__init__(seed)
        self.num_classes = num_classes
        self.modality_noise = modality_noise
        self.env = MultiModalIntegrationEnvironment(
            num_classes=num_classes,
            modality_noise=modality_noise,
            seed=seed
        )

    def get_name(self) -> str:
        """Get task name."""
        return f"MultiModalIntegration-{self.num_classes}class"

    def get_description(self) -> str:
        """Get task description."""
        return (
            f"Integrate {self.num_classes} classes across vision and audio. "
            f"Individual modality noise: {self.modality_noise}"
        )

    def evaluate(self, simulation, num_episodes: int = 20, max_steps: int = 100) -> TaskResult:
        """Evaluate multi-modal integration.
        
        Args:
            simulation: The simulation to evaluate
            num_episodes: Number of trials
            max_steps: Steps per trial
            
        Returns:
            TaskResult with integration performance
        """
        try:
            from .senses import feed_sense_input
        except ImportError:
            from senses import feed_sense_input

        correct = 0
        total_reward = 0.0
        reaction_times = []

        for episode in range(num_episodes):
            observation, info = self.env.reset()
            target_class = info["target_class"]

            # Feed both modalities
            feed_sense_input(simulation.model, "vision", observation["vision"])
            # Use digital sense for audio (audio is already 2D)
            feed_sense_input(simulation.model, "digital", observation["audio"])

            first_response_step = None
            predicted_class = None

            for step in range(max_steps):
                stats = simulation.step()

                # Simplified prediction from network activity
                if len(stats["spikes"]) > 0 and first_response_step is None:
                    first_response_step = step
                    # Placeholder: Use spike pattern for prediction
                    predicted_class = self.rng.integers(0, self.num_classes)

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
                "num_episodes": num_episodes,
                "num_responses": len(reaction_times)
            },
        )

    def get_metrics(self) -> Dict[str, str]:
        """Get metric descriptions."""
        return {
            "accuracy": "Multi-modal classification accuracy",
            "reward": "Average reward per episode",
            "reaction_time": "Steps to decision",
            "stability": "Response consistency",
        }


class MultiModalIntegrationEnvironment(Environment):
    """Environment for multi-modal integration task."""

    def __init__(
        self,
        num_classes: int = 4,
        modality_noise: float = 0.2,
        seed: int = None
    ) -> None:
        """Initialize multi-modal environment.
        
        Args:
            num_classes: Number of classes
            modality_noise: Noise level for modalities
            seed: Random seed
        """
        super().__init__(seed)
        self.num_classes = num_classes
        self.modality_noise = modality_noise
        self.current_class = None

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset with new multi-modal stimulus."""
        self.current_step = 0
        self.episode_reward = 0.0
        
        self.current_class = self.rng.integers(0, self.num_classes)
        
        # Create complementary patterns in vision and audio
        vision_pattern = np.zeros((20, 20))
        audio_pattern = np.zeros((20, 20))
        
        # Visual component
        if self.current_class % 2 == 0:
            vision_pattern[::2, :] = 1.0
        else:
            vision_pattern[:, ::2] = 1.0
            
        # Audio component
        if self.current_class < self.num_classes // 2:
            audio_pattern[:5, :] = 1.0
        else:
            audio_pattern[15:, :] = 1.0
        
        # Add noise
        vision_pattern += self.rng.normal(0, self.modality_noise, vision_pattern.shape)
        audio_pattern += self.rng.normal(0, self.modality_noise, audio_pattern.shape)
        
        vision_pattern = np.clip(vision_pattern, 0, 1)
        audio_pattern = np.clip(audio_pattern, 0, 1)
        
        observation = {
            "vision": vision_pattern,
            "audio": audio_pattern
        }
        info = {"target_class": self.current_class}
        
        return observation, info

    def step(self, action: Optional[np.ndarray] = None) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Step environment (single-step task)."""
        self.current_step += 1
        done = True
        reward = 0.0
        
        observation = {
            "vision": np.zeros((20, 20)),
            "audio": np.zeros((20, 20))
        }
        info = self.get_info()
        info["target_class"] = self.current_class
        
        return observation, reward, done, info

    def render(self) -> Optional[np.ndarray]:
        """Render multi-modal stimulus."""
        return None


class ContinuousLearningTask(Task):
    """Continuous learning task.
    
    The network faces a sequence of different sub-tasks and must
    learn continuously without catastrophic forgetting.
    Tests plasticity, memory retention, and adaptation.
    """

    def __init__(
        self,
        num_phases: int = 3,
        steps_per_phase: int = 200,
        seed: int = None
    ) -> None:
        """Initialize continuous learning task.
        
        Args:
            num_phases: Number of learning phases
            steps_per_phase: Steps in each phase
            seed: Random seed
        """
        super().__init__(seed)
        self.num_phases = num_phases
        self.steps_per_phase = steps_per_phase
        self.env = ContinuousLearningEnvironment(
            num_phases=num_phases,
            steps_per_phase=steps_per_phase,
            seed=seed
        )

    def get_name(self) -> str:
        """Get task name."""
        return f"ContinuousLearning-{self.num_phases}phases"

    def get_description(self) -> str:
        """Get task description."""
        return (
            f"Learn {self.num_phases} sequential tasks without forgetting. "
            f"Steps per phase: {self.steps_per_phase}"
        )

    def evaluate(self, simulation, num_episodes: int = 1, max_steps: int = 1000) -> TaskResult:
        """Evaluate continuous learning capability.
        
        Args:
            simulation: The simulation to evaluate
            num_episodes: Number of full sequences
            max_steps: Maximum total steps
            
        Returns:
            TaskResult with learning metrics
        """
        try:
            from .senses import feed_sense_input
        except ImportError:
            from senses import feed_sense_input

        phase_accuracies = [[] for _ in range(self.num_phases)]
        total_reward = 0.0

        for episode in range(num_episodes):
            observation, info = self.env.reset()
            done = False
            steps = 0

            while not done and steps < max_steps:
                current_phase = info["current_phase"]
                
                # Feed phase-specific input
                feed_sense_input(simulation.model, "vision", observation["vision"])
                
                # Run simulation
                for _ in range(10):
                    simulation.step()
                    steps += 1
                
                # Simplified evaluation: track phase performance
                # Real implementation would evaluate on held-out test set
                phase_correct = self.rng.random() < 0.6  # Placeholder
                phase_accuracies[current_phase].append(1.0 if phase_correct else 0.0)
                
                observation, reward, done, info = self.env.step()
                total_reward += reward

        # Calculate metrics
        avg_accuracies = [
            np.mean(acc) if acc else 0.0 
            for acc in phase_accuracies
        ]
        overall_accuracy = np.mean([a for a in avg_accuracies if a > 0]) if any(avg_accuracies) else 0.0
        
        # Measure forgetting (compare final phase to early phases)
        forgetting = 0.0
        if len(avg_accuracies) > 1:
            forgetting = max(0, avg_accuracies[0] - avg_accuracies[-1])

        avg_reward = total_reward / max(num_episodes, 1)

        return TaskResult(
            accuracy=overall_accuracy,
            reward=avg_reward,
            reaction_time=0.0,
            stability=1.0 - forgetting,
            additional_metrics={
                "phase_accuracies": avg_accuracies,
                "forgetting": forgetting,
                "num_phases": self.num_phases,
            },
        )

    def get_metrics(self) -> Dict[str, str]:
        """Get metric descriptions."""
        return {
            "accuracy": "Average accuracy across all phases",
            "reward": "Average reward",
            "stability": "1 - forgetting (retention of early learning)",
        }


class ContinuousLearningEnvironment(Environment):
    """Environment for continuous learning task."""

    def __init__(
        self,
        num_phases: int = 3,
        steps_per_phase: int = 200,
        seed: int = None
    ) -> None:
        """Initialize continuous learning environment.
        
        Args:
            num_phases: Number of learning phases
            steps_per_phase: Steps per phase
            seed: Random seed
        """
        super().__init__(seed)
        self.num_phases = num_phases
        self.steps_per_phase = steps_per_phase
        self.current_phase = 0
        self.phase_step = 0

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset to first phase."""
        self.current_step = 0
        self.episode_reward = 0.0
        self.current_phase = 0
        self.phase_step = 0
        
        observation = self._generate_phase_input(self.current_phase)
        info = {
            "current_phase": self.current_phase,
            "phase_step": self.phase_step
        }
        
        return observation, info

    def step(self, action: Optional[np.ndarray] = None) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Step through continuous learning phases."""
        self.current_step += 1
        self.phase_step += 1
        
        # Check if moving to next phase
        if self.phase_step >= self.steps_per_phase:
            self.current_phase += 1
            self.phase_step = 0
        
        done = self.current_phase >= self.num_phases
        reward = 0.0
        
        if not done:
            observation = self._generate_phase_input(self.current_phase)
        else:
            observation = {"vision": np.zeros((20, 20))}
        
        info = self.get_info()
        info["current_phase"] = self.current_phase
        info["phase_step"] = self.phase_step
        
        return observation, reward, done, info

    def _generate_phase_input(self, phase: int) -> Dict[str, np.ndarray]:
        """Generate phase-specific input pattern."""
        pattern = np.zeros((20, 20))
        
        # Different pattern for each phase
        if phase == 0:
            pattern[:, ::2] = 1.0  # Vertical stripes
        elif phase == 1:
            pattern[::2, :] = 1.0  # Horizontal stripes
        else:
            pattern[::3, ::3] = 1.0  # Grid pattern
        
        # Add some noise
        pattern += self.rng.normal(0, 0.1, pattern.shape)
        pattern = np.clip(pattern, 0, 1)
        
        return {"vision": pattern}

    def render(self) -> Optional[np.ndarray]:
        """Render current phase input."""
        return self._generate_phase_input(self.current_phase)["vision"]


class TransferLearningTask(Task):
    """Transfer learning task.
    
    The network learns a source task and must transfer knowledge
    to a related but different target task.
    Tests generalization and knowledge transfer.
    """

    def __init__(
        self,
        source_classes: int = 4,
        target_classes: int = 4,
        similarity: float = 0.7,
        seed: int = None
    ) -> None:
        """Initialize transfer learning task.
        
        Args:
            source_classes: Number of source task classes
            target_classes: Number of target task classes
            similarity: Similarity between tasks (0-1)
            seed: Random seed
        """
        super().__init__(seed)
        self.source_classes = source_classes
        self.target_classes = target_classes
        self.similarity = similarity
        self.env = TransferLearningEnvironment(
            source_classes=source_classes,
            target_classes=target_classes,
            similarity=similarity,
            seed=seed
        )

    def get_name(self) -> str:
        """Get task name."""
        return f"TransferLearning-S{self.source_classes}T{self.target_classes}"

    def get_description(self) -> str:
        """Get task description."""
        return (
            f"Transfer from {self.source_classes} source classes to "
            f"{self.target_classes} target classes. Similarity: {self.similarity}"
        )

    def evaluate(self, simulation, num_episodes: int = 40, max_steps: int = 100) -> TaskResult:
        """Evaluate transfer learning.
        
        First half of episodes are source task, second half are target task.
        
        Args:
            simulation: The simulation to evaluate
            num_episodes: Total episodes (split between source/target)
            max_steps: Steps per episode
            
        Returns:
            TaskResult with transfer performance
        """
        try:
            from .senses import feed_sense_input
        except ImportError:
            from senses import feed_sense_input

        source_episodes = num_episodes // 2
        target_episodes = num_episodes - source_episodes
        
        source_correct = 0
        target_correct = 0
        total_reward = 0.0

        # Source task phase
        for episode in range(source_episodes):
            observation, info = self.env.reset(target_task=False)
            target_class = info["target_class"]

            feed_sense_input(simulation.model, "vision", observation["vision"])

            for _ in range(max_steps):
                simulation.step()

            # Placeholder prediction
            predicted = self.rng.integers(0, self.source_classes)
            if predicted == target_class:
                source_correct += 1
                total_reward += 1.0

        # Target task phase (transfer)
        for episode in range(target_episodes):
            observation, info = self.env.reset(target_task=True)
            target_class = info["target_class"]

            feed_sense_input(simulation.model, "vision", observation["vision"])

            for _ in range(max_steps):
                simulation.step()

            # Placeholder prediction
            predicted = self.rng.integers(0, self.target_classes)
            if predicted == target_class:
                target_correct += 1
                total_reward += 1.0

        source_accuracy = source_correct / source_episodes if source_episodes > 0 else 0.0
        target_accuracy = target_correct / target_episodes if target_episodes > 0 else 0.0
        
        # Transfer effectiveness: how much better than random
        random_performance = 1.0 / self.target_classes
        transfer_gain = max(0, target_accuracy - random_performance) / max(0.01, source_accuracy - random_performance)

        avg_reward = total_reward / num_episodes

        return TaskResult(
            accuracy=target_accuracy,
            reward=avg_reward,
            reaction_time=0.0,
            stability=transfer_gain,
            additional_metrics={
                "source_accuracy": source_accuracy,
                "target_accuracy": target_accuracy,
                "transfer_gain": transfer_gain,
            },
        )

    def get_metrics(self) -> Dict[str, str]:
        """Get metric descriptions."""
        return {
            "accuracy": "Target task accuracy",
            "reward": "Average reward",
            "stability": "Transfer effectiveness gain",
        }


class TransferLearningEnvironment(Environment):
    """Environment for transfer learning task."""

    def __init__(
        self,
        source_classes: int = 4,
        target_classes: int = 4,
        similarity: float = 0.7,
        seed: int = None
    ) -> None:
        """Initialize transfer learning environment.
        
        Args:
            source_classes: Number of source classes
            target_classes: Number of target classes
            similarity: Task similarity (0-1)
            seed: Random seed
        """
        super().__init__(seed)
        self.source_classes = source_classes
        self.target_classes = target_classes
        self.similarity = similarity
        
        # Generate base patterns
        self.source_patterns = self._generate_patterns(source_classes)
        self.target_patterns = self._generate_related_patterns(target_classes)
        
        self.current_class = None
        self.is_target_task = False

    def _generate_patterns(self, num_classes: int) -> np.ndarray:
        """Generate base patterns for classes."""
        patterns = np.zeros((num_classes, 20, 20))
        
        for i in range(num_classes):
            if i % 4 == 0:
                patterns[i, ::2, :] = 1.0
            elif i % 4 == 1:
                patterns[i, :, ::2] = 1.0
            elif i % 4 == 2:
                for j in range(min(20, 20)):
                    patterns[i, j, j] = 1.0
            else:
                patterns[i, ::2, ::2] = 1.0
                patterns[i, 1::2, 1::2] = 1.0
        
        return patterns

    def _generate_related_patterns(self, num_classes: int) -> np.ndarray:
        """Generate related patterns based on similarity."""
        patterns = self._generate_patterns(num_classes).copy()
        
        # Add variation based on similarity
        noise_level = 1.0 - self.similarity
        noise = self.rng.normal(0, noise_level, patterns.shape)
        patterns = np.clip(patterns + noise, 0, 1)
        
        return patterns

    def reset(self, target_task: bool = False) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset with source or target task pattern.
        
        Args:
            target_task: If True, use target task; otherwise source task
        """
        self.current_step = 0
        self.episode_reward = 0.0
        self.is_target_task = target_task
        
        if target_task:
            self.current_class = self.rng.integers(0, self.target_classes)
            pattern = self.target_patterns[self.current_class].copy()
        else:
            self.current_class = self.rng.integers(0, self.source_classes)
            pattern = self.source_patterns[self.current_class].copy()
        
        # Add noise
        pattern += self.rng.normal(0, 0.1, pattern.shape)
        pattern = np.clip(pattern, 0, 1)
        
        observation = {"vision": pattern}
        info = {
            "target_class": self.current_class,
            "is_target_task": target_task
        }
        
        return observation, info

    def step(self, action: Optional[np.ndarray] = None) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Step environment (single-step task)."""
        self.current_step += 1
        done = True
        reward = 0.0
        
        observation = {"vision": np.zeros((20, 20))}
        info = self.get_info()
        info["target_class"] = self.current_class
        info["is_target_task"] = self.is_target_task
        
        return observation, reward, done, info

    def render(self) -> Optional[np.ndarray]:
        """Render current pattern."""
        return None
