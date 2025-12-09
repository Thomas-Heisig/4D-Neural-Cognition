"""Temporal prediction using the w-dimension for temporal processing.

This module implements temporal prediction capabilities that leverage
the w-dimension as temporal depth for sequence learning and forecasting.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import deque


class TemporalPredictor:
    """Leverages w-dimension for temporal processing and prediction.
    
    This class implements an echo state network-like architecture
    using the w-dimension to represent temporal depth.
    """
    
    def __init__(
        self,
        w_depth: int = 10,
        leak_rate: float = 0.3,
        spectral_radius: float = 0.95
    ):
        """Initialize temporal predictor.
        
        Args:
            w_depth: Number of w-layers for temporal depth
            leak_rate: Leak rate for temporal integration
            spectral_radius: Spectral radius for reservoir stability
        """
        self.w_depth = w_depth
        self.leak_rate = leak_rate
        self.spectral_radius = spectral_radius
        
        # Temporal state for each w-layer
        self.w_states: Dict[int, np.ndarray] = {}
        
        # History buffers for each w-layer
        self.w_histories: Dict[int, deque] = {
            w: deque(maxlen=100) for w in range(w_depth)
        }
    
    def initialize_state(
        self,
        w_layer: int,
        state_size: int
    ) -> None:
        """Initialize state for a w-layer.
        
        Args:
            w_layer: W-dimension index
            state_size: Size of state vector
        """
        if w_layer not in self.w_states:
            self.w_states[w_layer] = np.zeros(state_size)
    
    def update_temporal_state(
        self,
        w_layer: int,
        input_signal: np.ndarray,
        dt: float = 1.0
    ) -> np.ndarray:
        """Update temporal state for a w-layer.
        
        Args:
            w_layer: W-dimension index
            input_signal: Input signal to process
            dt: Time step
            
        Returns:
            Updated state
        """
        if w_layer not in self.w_states:
            self.initialize_state(w_layer, len(input_signal))
        
        current_state = self.w_states[w_layer]
        
        # Leaky integration with input
        new_state = (
            (1 - self.leak_rate) * current_state +
            self.leak_rate * input_signal
        )
        
        # Update state
        self.w_states[w_layer] = new_state
        
        # Store in history
        self.w_histories[w_layer].append(new_state.copy())
        
        return new_state
    
    def propagate_temporal_information(
        self,
        coupling_strength: float = 0.1
    ) -> None:
        """Propagate information across w-layers.
        
        Args:
            coupling_strength: Strength of coupling between layers
        """
        # Forward propagation (present to future)
        for w in range(self.w_depth - 1):
            if w in self.w_states and w + 1 in self.w_states:
                self.w_states[w + 1] += (
                    coupling_strength * self.w_states[w]
                )
        
        # Backward propagation (future to present)
        for w in range(self.w_depth - 1, 0, -1):
            if w in self.w_states and w - 1 in self.w_states:
                self.w_states[w - 1] += (
                    coupling_strength * 0.5 * self.w_states[w]
                )
    
    def predict_sequence(
        self,
        input_sequence: List[np.ndarray],
        prediction_steps: int
    ) -> List[np.ndarray]:
        """Predict future sequence values.
        
        Args:
            input_sequence: Historical input sequence
            prediction_steps: Number of steps to predict
            
        Returns:
            List of predicted values
        """
        # Process input sequence
        for t, input_signal in enumerate(input_sequence):
            w_layer = t % self.w_depth
            self.update_temporal_state(w_layer, input_signal)
            self.propagate_temporal_information()
        
        # Generate predictions
        predictions = []
        
        for step in range(prediction_steps):
            # Use most recent w-layer state as prediction
            w_layer = (len(input_sequence) + step) % self.w_depth
            
            if w_layer in self.w_states:
                prediction = self.w_states[w_layer].copy()
                predictions.append(prediction)
                
                # Update state with prediction for next step
                self.update_temporal_state(w_layer, prediction)
                self.propagate_temporal_information()
        
        return predictions
    
    def get_temporal_context(
        self,
        w_layer: int,
        context_length: int = 10
    ) -> np.ndarray:
        """Get temporal context from a w-layer.
        
        Args:
            w_layer: W-dimension index
            context_length: Length of context to retrieve
            
        Returns:
            Temporal context as flattened array
        """
        if w_layer not in self.w_histories:
            return np.array([])
        
        history = list(self.w_histories[w_layer])
        
        if not history:
            return np.array([])
        
        # Get recent history
        recent = history[-context_length:]
        
        # Flatten to single vector
        return np.concatenate(recent)
    
    def compute_prediction_error(
        self,
        predictions: List[np.ndarray],
        actual: List[np.ndarray]
    ) -> float:
        """Compute prediction error.
        
        Args:
            predictions: Predicted values
            actual: Actual values
            
        Returns:
            Mean squared error
        """
        errors = []
        
        for pred, act in zip(predictions, actual):
            if len(pred) == len(act):
                error = np.mean((pred - act) ** 2)
                errors.append(error)
        
        return np.mean(errors) if errors else 0.0


class EchoStateNetwork:
    """Echo State Network for temporal processing.
    
    This implements a reservoir computing approach using the 4D lattice
    structure with the w-dimension providing temporal organization.
    """
    
    def __init__(
        self,
        reservoir_size: int = 100,
        input_size: int = 10,
        output_size: int = 10,
        spectral_radius: float = 0.95,
        leak_rate: float = 0.3
    ):
        """Initialize echo state network.
        
        Args:
            reservoir_size: Size of reservoir (hidden layer)
            input_size: Size of input
            output_size: Size of output
            spectral_radius: Spectral radius for reservoir dynamics
            leak_rate: Leak rate for temporal integration
        """
        self.reservoir_size = reservoir_size
        self.input_size = input_size
        self.output_size = output_size
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        
        # Initialize weights
        self.W_in = np.random.randn(reservoir_size, input_size) * 0.1
        self.W_res = self._initialize_reservoir()
        self.W_out: Optional[np.ndarray] = None
        
        # Reservoir state
        self.state = np.zeros(reservoir_size)
        
        # Training data collection
        self.collected_states: List[np.ndarray] = []
        self.collected_targets: List[np.ndarray] = []
    
    def _initialize_reservoir(self) -> np.ndarray:
        """Initialize reservoir weight matrix with desired spectral radius."""
        # Random sparse connectivity
        W = np.random.randn(self.reservoir_size, self.reservoir_size)
        W[np.random.rand(*W.shape) > 0.1] = 0  # 10% connectivity
        
        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(W)
        current_radius = np.max(np.abs(eigenvalues))
        
        if current_radius > 0:
            W = W * (self.spectral_radius / current_radius)
        
        return W
    
    def update(
        self,
        input_signal: np.ndarray,
        collect_for_training: bool = False,
        target: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Update reservoir state.
        
        Args:
            input_signal: Input signal
            collect_for_training: Whether to collect state for training
            target: Target output (for training)
            
        Returns:
            Updated reservoir state
        """
        # Compute new state
        input_activation = np.dot(self.W_in, input_signal)
        reservoir_activation = np.dot(self.W_res, self.state)
        
        new_state = (
            (1 - self.leak_rate) * self.state +
            self.leak_rate * np.tanh(input_activation + reservoir_activation)
        )
        
        self.state = new_state
        
        # Collect for training if requested
        if collect_for_training and target is not None:
            self.collected_states.append(self.state.copy())
            self.collected_targets.append(target)
        
        return self.state
    
    def train_readout(self, regularization: float = 1e-6) -> None:
        """Train output weights using ridge regression.
        
        Args:
            regularization: Regularization parameter
        """
        if not self.collected_states:
            raise ValueError("No training data collected")
        
        # Convert to matrices
        X = np.array(self.collected_states)  # States
        Y = np.array(self.collected_targets)  # Targets
        
        # Ridge regression: W_out = (X^T X + λI)^-1 X^T Y
        XtX = np.dot(X.T, X)
        XtX += regularization * np.eye(XtX.shape[0])
        XtY = np.dot(X.T, Y)
        
        self.W_out = np.linalg.solve(XtX, XtY).T
        
        # Clear collected data
        self.collected_states = []
        self.collected_targets = []
    
    def predict(self, input_signal: np.ndarray) -> np.ndarray:
        """Generate prediction.
        
        Args:
            input_signal: Input signal
            
        Returns:
            Predicted output
        """
        # Update reservoir
        self.update(input_signal)
        
        # Generate output
        if self.W_out is not None:
            return np.dot(self.W_out, self.state)
        else:
            raise ValueError("Network not trained. Call train_readout() first.")
    
    def predict_sequence(
        self,
        input_sequence: List[np.ndarray],
        prediction_steps: int,
        closed_loop: bool = False
    ) -> List[np.ndarray]:
        """Predict a sequence.
        
        Args:
            input_sequence: Input sequence for context
            prediction_steps: Number of steps to predict
            closed_loop: If True, feed predictions back as input
            
        Returns:
            List of predictions
        """
        # Process input sequence
        for input_signal in input_sequence:
            self.update(input_signal)
        
        # Generate predictions
        predictions = []
        
        if closed_loop and self.W_out is not None:
            # Feed predictions back as input
            current_input = input_sequence[-1] if input_sequence else np.zeros(self.input_size)
            
            for _ in range(prediction_steps):
                pred = self.predict(current_input)
                predictions.append(pred)
                
                # Use prediction as next input (if sizes match)
                if len(pred) == len(current_input):
                    current_input = pred
        else:
            # Open loop: use zero input
            for _ in range(prediction_steps):
                pred = self.predict(np.zeros(self.input_size))
                predictions.append(pred)
        
        return predictions
    
    def reset_state(self) -> None:
        """Reset reservoir state to zero."""
        self.state = np.zeros(self.reservoir_size)


class SequenceLearner:
    """Learn and predict temporal sequences using 4D structure."""
    
    def __init__(
        self,
        sequence_length: int = 10,
        prediction_horizon: int = 5
    ):
        """Initialize sequence learner.
        
        Args:
            sequence_length: Length of input sequences
            prediction_horizon: How far ahead to predict
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        self.sequence_buffer: deque = deque(maxlen=sequence_length)
        self.learned_patterns: Dict[str, List[np.ndarray]] = {}
    
    def add_observation(self, observation: np.ndarray) -> None:
        """Add an observation to the sequence buffer.
        
        Args:
            observation: New observation
        """
        self.sequence_buffer.append(observation.copy())
    
    def learn_pattern(self, pattern_name: str) -> None:
        """Learn current sequence as a named pattern.
        
        Args:
            pattern_name: Name for this pattern
        """
        if len(self.sequence_buffer) >= self.sequence_length:
            pattern = list(self.sequence_buffer)
            self.learned_patterns[pattern_name] = pattern
    
    def match_pattern(
        self,
        threshold: float = 0.8
    ) -> Optional[Tuple[str, float]]:
        """Match current sequence against learned patterns.
        
        Args:
            threshold: Minimum similarity for match
            
        Returns:
            Tuple of (pattern_name, similarity) or None
        """
        if len(self.sequence_buffer) < self.sequence_length:
            return None
        
        current_sequence = list(self.sequence_buffer)
        best_match = None
        best_similarity = 0.0
        
        for pattern_name, pattern in self.learned_patterns.items():
            # Compute similarity
            similarities = []
            for curr, patt in zip(current_sequence, pattern):
                if len(curr) == len(patt):
                    # Cosine similarity
                    sim = np.dot(curr, patt) / (
                        np.linalg.norm(curr) * np.linalg.norm(patt) + 1e-10
                    )
                    similarities.append(sim)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                
                if avg_similarity > best_similarity:
                    best_similarity = avg_similarity
                    best_match = pattern_name
        
        if best_similarity >= threshold:
            return best_match, best_similarity
        
        return None
    
    def get_statistics(self) -> Dict[str, any]:
        """Get statistics about sequence learning.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'buffer_size': len(self.sequence_buffer),
            'num_patterns': len(self.learned_patterns)
        }
