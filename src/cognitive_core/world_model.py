"""Internal world model for prediction and planning.

This module implements an internal simulation/prediction capability,
allowing the system to predict future states and consequences of actions
without actual interaction with the environment.
"""

from typing import Dict, List, Optional, Tuple, Any, Callable, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    try:
        from ..brain_model import BrainModel
    except ImportError:
        from brain_model import BrainModel


class WorldModel:
    """Internal world model for predictive coding and planning.
    
    Implements:
    - State prediction
    - Action consequence modeling
    - Counterfactual reasoning
    - Mental simulation
    """
    
    def __init__(
        self,
        model: "BrainModel",
        prediction_horizon: int = 5,
        learning_rate: float = 0.01
    ):
        """Initialize world model.
        
        Args:
            model: Brain model with 4D lattice
            prediction_horizon: Number of steps to predict ahead
            learning_rate: Learning rate for model updates
        """
        self.model = model
        self.prediction_horizon = prediction_horizon
        self.learning_rate = learning_rate
        
        # State history
        self.state_history: List[np.ndarray] = []
        self.action_history: List[Optional[np.ndarray]] = []
        
        # Learned transition model (simplified)
        self.transition_matrix: Optional[np.ndarray] = None
        self.state_dim: Optional[int] = None
        
        # Prediction errors for learning
        self.prediction_errors: List[float] = []
    
    def observe_state(
        self,
        state: np.ndarray,
        action: Optional[np.ndarray] = None
    ) -> None:
        """Observe and record a state-action pair.
        
        Args:
            state: Current state observation
            action: Action taken (if any)
        """
        self.state_history.append(state.copy())
        self.action_history.append(action.copy() if action is not None else None)
        
        # Initialize transition matrix if needed
        if self.transition_matrix is None and len(self.state_history) > 1:
            self.state_dim = len(state)
            self.transition_matrix = np.eye(self.state_dim) * 0.9
    
    def predict_next_state(
        self,
        current_state: np.ndarray,
        action: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """Predict the next state given current state and action.
        
        Args:
            current_state: Current state
            action: Action to be taken
            
        Returns:
            Tuple of (predicted next state, confidence)
        """
        if self.transition_matrix is None or len(self.state_history) < 2:
            # No model learned yet, return current state
            return current_state.copy(), 0.0
        
        # Simple linear prediction
        predicted = self.transition_matrix @ current_state
        
        # Add action influence if provided
        if action is not None and len(action) == len(predicted):
            predicted += action * 0.1  # Small action influence
        
        # Compute confidence based on recent prediction accuracy
        if self.prediction_errors:
            recent_errors = self.prediction_errors[-10:]
            confidence = 1.0 / (1.0 + np.mean(recent_errors))
        else:
            confidence = 0.5
        
        return predicted, float(confidence)
    
    def predict_sequence(
        self,
        initial_state: np.ndarray,
        actions: Optional[List[np.ndarray]] = None,
        steps: Optional[int] = None
    ) -> List[Tuple[np.ndarray, float]]:
        """Predict a sequence of future states.
        
        Args:
            initial_state: Starting state
            actions: Sequence of actions (optional)
            steps: Number of steps to predict (uses prediction_horizon if None)
            
        Returns:
            List of (predicted state, confidence) tuples
        """
        if steps is None:
            steps = self.prediction_horizon
        
        predictions = []
        current_state = initial_state.copy()
        
        for i in range(steps):
            action = actions[i] if actions and i < len(actions) else None
            predicted_state, confidence = self.predict_next_state(current_state, action)
            predictions.append((predicted_state, confidence))
            current_state = predicted_state
        
        return predictions
    
    def update_model(
        self,
        previous_state: np.ndarray,
        actual_next_state: np.ndarray,
        action: Optional[np.ndarray] = None
    ) -> float:
        """Update the world model based on actual outcome.
        
        Args:
            previous_state: Previous state
            actual_next_state: Actual next state observed
            action: Action that was taken
            
        Returns:
            Prediction error
        """
        # Predict what should have happened
        predicted_state, _ = self.predict_next_state(previous_state, action)
        
        # Compute prediction error
        error = np.mean((predicted_state - actual_next_state) ** 2)
        self.prediction_errors.append(float(error))
        
        # Update transition matrix
        if (self.transition_matrix is not None and
            len(previous_state) == self.state_dim and
            len(actual_next_state) == self.state_dim):
            
            # Gradient descent update
            # T_new = T_old + lr * (actual - predicted) * prev^T
            delta = (actual_next_state - predicted_state).reshape(-1, 1)
            prev = previous_state.reshape(1, -1)
            self.transition_matrix += self.learning_rate * (delta @ prev)
        
        return float(error)
    
    def plan_actions(
        self,
        current_state: np.ndarray,
        goal_state: np.ndarray,
        max_steps: int = 10
    ) -> List[np.ndarray]:
        """Plan a sequence of actions to reach a goal state.
        
        Args:
            current_state: Current state
            goal_state: Desired goal state
            max_steps: Maximum planning steps
            
        Returns:
            List of planned actions
        """
        actions = []
        state = current_state.copy()
        
        for _ in range(max_steps):
            # Compute direction to goal
            direction = goal_state - state
            distance = np.linalg.norm(direction)
            
            if distance < 0.1:  # Close enough to goal
                break
            
            # Simple action: move toward goal
            action = direction / (distance + 1e-8) * 0.5
            actions.append(action)
            
            # Simulate next state
            state, _ = self.predict_next_state(state, action)
        
        return actions
    
    def counterfactual_reasoning(
        self,
        state: np.ndarray,
        alternative_actions: List[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """Perform counterfactual reasoning on alternative actions.
        
        Args:
            state: Current state
            alternative_actions: List of alternative actions to consider
            
        Returns:
            List of outcomes for each alternative action
        """
        outcomes = []
        
        for action in alternative_actions:
            predicted_state, confidence = self.predict_next_state(state, action)
            outcomes.append({
                "action": action,
                "predicted_state": predicted_state,
                "confidence": confidence
            })
        
        return outcomes
    
    def mental_simulation(
        self,
        scenario: Dict[str, Any],
        simulation_steps: int = 10
    ) -> Dict[str, Any]:
        """Run a mental simulation of a scenario.
        
        Args:
            scenario: Dictionary containing scenario parameters
            simulation_steps: Number of steps to simulate
            
        Returns:
            Dictionary containing simulation results
        """
        initial_state = scenario.get("initial_state", np.zeros(self.state_dim or 10))
        actions = scenario.get("actions", None)
        
        # Run simulation
        predictions = self.predict_sequence(initial_state, actions, simulation_steps)
        
        # Analyze trajectory
        states = [pred[0] for pred in predictions]
        confidences = [pred[1] for pred in predictions]
        
        result = {
            "trajectory": states,
            "average_confidence": float(np.mean(confidences)),
            "final_state": states[-1] if states else initial_state,
            "trajectory_length": len(states)
        }
        
        return result
    
    def get_prediction_accuracy(self) -> Dict[str, float]:
        """Get prediction accuracy statistics.
        
        Returns:
            Dictionary of accuracy metrics
        """
        if not self.prediction_errors:
            return {
                "mean_error": 0.0,
                "std_error": 0.0,
                "accuracy": 0.0
            }
        
        errors = np.array(self.prediction_errors)
        return {
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "accuracy": float(1.0 / (1.0 + np.mean(errors)))  # Normalized accuracy
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about world model.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "state_history_length": len(self.state_history),
            "model_initialized": self.transition_matrix is not None,
            "state_dimension": self.state_dim,
            "prediction_accuracy": self.get_prediction_accuracy()
        }
        
        return stats


__all__ = ["WorldModel"]
