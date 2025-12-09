"""Motor output module for action generation and control.

This module provides motor output capabilities including:
- Motor cortex area management
- Action selection mechanisms
- Continuous control outputs
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    try:
        from .brain_model import BrainModel
    except ImportError:
        from brain_model import BrainModel


class MotorCortexArea:
    """Represents a motor cortex area for action generation."""

    def __init__(
        self,
        name: str,
        area_coords: Dict[str, Tuple[int, int]],
        action_space_dim: int = 2,
        activation_threshold: float = 0.5,
    ):
        """Initialize motor cortex area.

        Args:
            name: Name of the motor area.
            area_coords: Dictionary with coordinate ranges (x, y, z, w).
            action_space_dim: Dimension of action space.
            activation_threshold: Threshold for neuron activation.
        """
        self.name = name
        self.area_coords = area_coords
        self.action_space_dim = action_space_dim
        self.activation_threshold = activation_threshold

    def extract_motor_output(self, model: "BrainModel") -> np.ndarray:
        """Extract motor output from neurons in this area.

        Args:
            model: Brain model to extract output from.

        Returns:
            Motor output vector.
        """
        # Get neurons in motor area
        neurons = self._get_area_neurons(model)

        if not neurons:
            return np.zeros(self.action_space_dim)

        # Collect activation values
        activations = []
        for neuron in neurons:
            # Use membrane potential as activation measure
            activation = neuron.v_membrane if hasattr(neuron, 'v_membrane') else 0.0
            activations.append(activation)

        activations = np.array(activations)

        # Normalize activations
        if len(activations) > 0 and np.max(activations) > 0:
            activations = activations / np.max(activations)

        # Map to action space dimension
        if len(activations) >= self.action_space_dim:
            # Use first action_space_dim neurons
            output = activations[:self.action_space_dim]
        else:
            # Pad with zeros if not enough neurons
            output = np.zeros(self.action_space_dim)
            output[:len(activations)] = activations

        return output

    def _get_area_neurons(self, model: "BrainModel") -> List[Any]:
        """Get neurons within the motor area coordinates.

        Args:
            model: Brain model.

        Returns:
            List of neurons in the area.
        """
        neurons = []
        x_range = self.area_coords.get('x', (0, 0))
        y_range = self.area_coords.get('y', (0, 0))
        z_range = self.area_coords.get('z', (0, 0))
        w_range = self.area_coords.get('w', (0, 0))

        for neuron in model.neurons.values():
            if (
                x_range[0] <= neuron.x <= x_range[1]
                and y_range[0] <= neuron.y <= y_range[1]
                and z_range[0] <= neuron.z <= z_range[1]
                and w_range[0] <= neuron.w <= w_range[1]
            ):
                neurons.append(neuron)

        return neurons


class ActionSelector:
    """Action selection mechanism for decision-making."""

    def __init__(self, num_actions: int, selection_method: str = 'softmax'):
        """Initialize action selector.

        Args:
            num_actions: Number of discrete actions.
            selection_method: Method for action selection ('softmax', 'argmax', 'epsilon_greedy').
        """
        self.num_actions = num_actions
        self.selection_method = selection_method
        self.epsilon = 0.1  # For epsilon-greedy

    def select_action(self, action_values: np.ndarray, temperature: float = 1.0) -> int:
        """Select an action based on action values.

        Args:
            action_values: Array of action values/preferences.
            temperature: Temperature parameter for softmax.

        Returns:
            Selected action index.
        """
        if len(action_values) != self.num_actions:
            raise ValueError(
                f"Expected {self.num_actions} action values, got {len(action_values)}"
            )

        if self.selection_method == 'softmax':
            return self._softmax_selection(action_values, temperature)
        elif self.selection_method == 'argmax':
            return self._argmax_selection(action_values)
        elif self.selection_method == 'epsilon_greedy':
            return self._epsilon_greedy_selection(action_values)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")

    def _softmax_selection(self, values: np.ndarray, temperature: float) -> int:
        """Select action using softmax distribution.

        Args:
            values: Action values.
            temperature: Temperature parameter.

        Returns:
            Selected action index.
        """
        # Apply temperature scaling
        scaled_values = values / (temperature + 1e-8)

        # Compute softmax probabilities
        exp_values = np.exp(scaled_values - np.max(scaled_values))  # Numerical stability
        probabilities = exp_values / np.sum(exp_values)

        # Sample from distribution
        action = np.random.choice(self.num_actions, p=probabilities)
        return int(action)

    def _argmax_selection(self, values: np.ndarray) -> int:
        """Select action with highest value.

        Args:
            values: Action values.

        Returns:
            Selected action index.
        """
        return int(np.argmax(values))

    def _epsilon_greedy_selection(self, values: np.ndarray) -> int:
        """Select action using epsilon-greedy strategy.

        Args:
            values: Action values.

        Returns:
            Selected action index.
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            return int(np.random.randint(self.num_actions))
        else:
            # Exploit: best action
            return int(np.argmax(values))

    def set_epsilon(self, epsilon: float) -> None:
        """Set epsilon parameter for epsilon-greedy.

        Args:
            epsilon: Exploration rate (0 to 1).
        """
        self.epsilon = max(0.0, min(1.0, epsilon))

    def get_action_probabilities(self, action_values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Get probability distribution over actions.

        Args:
            action_values: Array of action values.
            temperature: Temperature parameter for softmax.

        Returns:
            Probability distribution over actions.
        """
        scaled_values = action_values / (temperature + 1e-8)
        exp_values = np.exp(scaled_values - np.max(scaled_values))
        probabilities = exp_values / np.sum(exp_values)
        return probabilities


class ContinuousController:
    """Continuous control output for motor actions."""

    def __init__(self, output_dim: int, output_range: Tuple[float, float] = (-1.0, 1.0)):
        """Initialize continuous controller.

        Args:
            output_dim: Dimension of output space.
            output_range: Range of output values (min, max).
        """
        self.output_dim = output_dim
        self.output_range = output_range
        self.output_history: List[np.ndarray] = []
        self.max_history = 100

    def generate_output(self, neural_activity: np.ndarray, smoothing: float = 0.0) -> np.ndarray:
        """Generate continuous control output from neural activity.

        Args:
            neural_activity: Array of neural activity values.
            smoothing: Smoothing factor (0 = no smoothing, 1 = max smoothing).

        Returns:
            Continuous control output.
        """
        # Ensure correct dimension
        if len(neural_activity) < self.output_dim:
            # Pad with zeros
            output = np.zeros(self.output_dim)
            output[:len(neural_activity)] = neural_activity
        else:
            # Take first output_dim values
            output = neural_activity[:self.output_dim]

        # Scale to output range
        output = self._scale_to_range(output)

        # Apply smoothing with history
        if smoothing > 0 and len(self.output_history) > 0:
            previous_output = self.output_history[-1]
            output = (1 - smoothing) * output + smoothing * previous_output

        # Store in history
        self.output_history.append(output.copy())
        if len(self.output_history) > self.max_history:
            self.output_history.pop(0)

        return output

    def _scale_to_range(self, values: np.ndarray) -> np.ndarray:
        """Scale values to output range.

        Args:
            values: Input values.

        Returns:
            Scaled values.
        """
        # Normalize to [0, 1]
        min_val = np.min(values)
        max_val = np.max(values)

        if max_val == min_val:
            normalized = np.ones_like(values) * 0.5
        else:
            normalized = (values - min_val) / (max_val - min_val)

        # Scale to output range
        output_min, output_max = self.output_range
        scaled = normalized * (output_max - output_min) + output_min

        return scaled

    def reset_history(self) -> None:
        """Clear output history."""
        self.output_history.clear()

    def get_output_statistics(self) -> Dict[str, float]:
        """Get statistics about recent outputs.

        Returns:
            Dictionary of statistics.
        """
        if not self.output_history:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}

        history_array = np.array(self.output_history)

        stats = {
            'mean': float(np.mean(history_array)),
            'std': float(np.std(history_array)),
            'min': float(np.min(history_array)),
            'max': float(np.max(history_array)),
        }

        return stats


class ReinforcementLearningIntegrator:
    """Integration with reinforcement learning for motor control."""

    def __init__(self, learning_rate: float = 0.01, discount_factor: float = 0.99):
        """Initialize RL integrator.

        Args:
            learning_rate: Learning rate for value updates.
            discount_factor: Discount factor for future rewards.
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.value_estimates: Dict[str, float] = {}
        self.reward_history: List[float] = []

    def update_values(
        self,
        state_key: str,
        reward: float,
        next_state_key: Optional[str] = None,
    ) -> None:
        """Update value estimates using TD learning.

        Args:
            state_key: Key identifying current state.
            reward: Reward received.
            next_state_key: Key identifying next state.
        """
        # Get current value estimate
        current_value = self.value_estimates.get(state_key, 0.0)

        # Get next state value
        if next_state_key is not None:
            next_value = self.value_estimates.get(next_state_key, 0.0)
        else:
            next_value = 0.0

        # TD error
        td_error = reward + self.discount_factor * next_value - current_value

        # Update value
        new_value = current_value + self.learning_rate * td_error
        self.value_estimates[state_key] = new_value

        # Record reward
        self.reward_history.append(reward)

    def get_value(self, state_key: str) -> float:
        """Get value estimate for a state.

        Args:
            state_key: State identifier.

        Returns:
            Value estimate.
        """
        return self.value_estimates.get(state_key, 0.0)

    def get_average_reward(self, window: int = 100) -> float:
        """Get average reward over recent history.

        Args:
            window: Number of recent rewards to average.

        Returns:
            Average reward.
        """
        if not self.reward_history:
            return 0.0

        recent_rewards = self.reward_history[-window:]
        return float(np.mean(recent_rewards))

    def reset(self) -> None:
        """Reset learning state."""
        self.value_estimates.clear()
        self.reward_history.clear()


def extract_motor_commands(
    model: "BrainModel",
    motor_area_name: str = "motor_cortex",
    control_type: str = "continuous",
    num_actions: int = 4,
) -> np.ndarray:
    """Extract motor commands from brain model.

    Args:
        model: Brain model to extract commands from.
        motor_area_name: Name of motor cortex area.
        control_type: Type of control ('continuous' or 'discrete').
        num_actions: Number of discrete actions (for discrete control).

    Returns:
        Motor command vector.
    """
    # Get motor area neurons
    areas = model.get_areas()
    motor_area = next((a for a in areas if a["name"] == motor_area_name), None)

    if motor_area is None:
        # Return default output if motor area not found
        if control_type == "continuous":
            return np.zeros(2)
        else:
            return np.zeros(num_actions)

    # Extract neural activity from motor area
    coord_ranges = motor_area["coord_ranges"]
    motor_cortex = MotorCortexArea(
        name=motor_area_name,
        area_coords=coord_ranges,
        action_space_dim=num_actions if control_type == "discrete" else 2,
    )

    neural_output = motor_cortex.extract_motor_output(model)

    if control_type == "continuous":
        controller = ContinuousController(output_dim=len(neural_output))
        motor_commands = controller.generate_output(neural_output, smoothing=0.3)
    else:
        # For discrete control, use neural_output as action values
        motor_commands = neural_output

    return motor_commands
