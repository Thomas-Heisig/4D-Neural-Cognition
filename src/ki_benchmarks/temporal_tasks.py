"""Temporal pattern memory tasks for cognitive benchmarking.

This module implements temporal reasoning benchmarks including:
- Sequence memory and recall
- Pattern prediction
- Temporal order reasoning
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    from ..tasks import Task, Environment, TaskResult
except ImportError:
    # Fallback for direct imports
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from tasks import Task, Environment, TaskResult


class SequenceMemoryEnvironment(Environment):
    """Environment for testing sequence memory.
    
    Presents a sequence of patterns that must be remembered and recalled.
    """
    
    def __init__(
        self,
        sequence_length: int = 5,
        pattern_dim: int = 10,
        delay_steps: int = 10,
        seed: int = None
    ):
        """Initialize sequence memory environment.
        
        Args:
            sequence_length: Number of patterns in sequence
            pattern_dim: Dimensionality of each pattern
            delay_steps: Steps between encoding and recall
            seed: Random seed
        """
        super().__init__(seed)
        self.sequence_length = sequence_length
        self.pattern_dim = pattern_dim
        self.delay_steps = delay_steps
        
        # State
        self.target_sequence: List[np.ndarray] = []
        self.recalled_sequence: List[np.ndarray] = []
        self.phase = "encoding"  # "encoding", "delay", "recall"
        self.sequence_index = 0
    
    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Generate random sequence
        self.target_sequence = []
        for _ in range(self.sequence_length):
            pattern = self.rng.uniform(-1, 1, size=self.pattern_dim)
            self.target_sequence.append(pattern)
        
        self.recalled_sequence = []
        self.phase = "encoding"
        self.sequence_index = 0
        
        # Return first pattern
        observation = self._create_observation()
        info = self.get_info()
        
        return observation, info
    
    def _create_observation(self) -> Dict[str, np.ndarray]:
        """Create observation of current state."""
        if self.phase == "encoding" and self.sequence_index < len(self.target_sequence):
            # Present pattern
            pattern = self.target_sequence[self.sequence_index]
        elif self.phase == "recall":
            # Present recall cue
            pattern = np.zeros(self.pattern_dim)
            pattern[0] = 1.0  # Recall signal
        else:
            # Delay period - no input
            pattern = np.zeros(self.pattern_dim)
        
        return {
            "digital": pattern,
            "vision": np.array([self.current_step, self.sequence_index])
        }
    
    def step(
        self,
        action: Optional[np.ndarray] = None
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Execute one step."""
        self.current_step += 1
        reward = 0.0
        
        # Phase transitions
        if self.phase == "encoding":
            self.sequence_index += 1
            if self.sequence_index >= self.sequence_length:
                self.phase = "delay"
                self.sequence_index = 0
        
        elif self.phase == "delay":
            self.sequence_index += 1
            if self.sequence_index >= self.delay_steps:
                self.phase = "recall"
                self.sequence_index = 0
        
        elif self.phase == "recall":
            # Evaluate recall
            if action is not None and len(action) == self.pattern_dim:
                if self.sequence_index < len(self.target_sequence):
                    target = self.target_sequence[self.sequence_index]
                    similarity = np.corrcoef(action, target)[0, 1]
                    
                    if not np.isnan(similarity):
                        reward = max(0.0, similarity)  # Reward for correct recall
                    
                    self.recalled_sequence.append(action.copy())
            
            self.sequence_index += 1
        
        # Check if done
        done = (self.phase == "recall" and 
                self.sequence_index >= self.sequence_length)
        
        observation = self._create_observation()
        self.episode_reward += reward
        info = self.get_info()
        info["phase"] = self.phase
        
        return observation, reward, done, info
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        # Visualize current pattern
        if self.phase == "encoding" and self.sequence_index < len(self.target_sequence):
            return self.target_sequence[self.sequence_index].reshape(-1, 1)
        return None


class TemporalPatternTask(Task):
    """Temporal pattern memory benchmark task.
    
    Evaluates the ability to:
    - Encode temporal sequences
    - Maintain information over delays
    - Recall sequences in correct order
    """
    
    def __init__(
        self,
        sequence_length: int = 5,
        pattern_dim: int = 10,
        delay_steps: int = 10,
        num_trials: int = 100,
        seed: int = None
    ):
        """Initialize temporal pattern task.
        
        Args:
            sequence_length: Length of sequences to remember
            pattern_dim: Dimensionality of patterns
            delay_steps: Delay between encoding and recall
            num_trials: Number of trials for evaluation
            seed: Random seed
        """
        super().__init__(seed)
        self.env = SequenceMemoryEnvironment(
            sequence_length, pattern_dim, delay_steps, seed
        )
        self.num_trials = num_trials
    
    def evaluate(self, model: Any) -> TaskResult:
        """Evaluate model on temporal pattern memory.
        
        Args:
            model: Neural network model to evaluate
            
        Returns:
            Task results with temporal memory performance
        """
        total_accuracy = 0.0
        total_reward = 0.0
        total_steps = 0
        
        for trial in range(self.num_trials):
            observation, info = self.env.reset()
            done = False
            trial_correct = 0
            trial_total = 0
            
            while not done:
                # Get action from model (placeholder)
                action = self.rng.uniform(-1, 1, size=self.env.pattern_dim)
                
                observation, reward, done, info = self.env.step(action)
                total_reward += reward
                total_steps += 1
                
                if info.get("phase") == "recall":
                    trial_total += 1
                    if reward > 0.5:  # Threshold for "correct"
                        trial_correct += 1
            
            if trial_total > 0:
                trial_accuracy = trial_correct / trial_total
                total_accuracy += trial_accuracy
        
        accuracy = total_accuracy / self.num_trials
        avg_reward = total_reward / self.num_trials
        avg_steps = total_steps / self.num_trials
        
        return TaskResult(
            accuracy=accuracy,
            reward=avg_reward,
            reaction_time=avg_steps,
            stability=accuracy,  # Consistent performance
            additional_metrics={
                "recall_accuracy": accuracy,
                "average_reward": avg_reward
            }
        )


class SequenceMemoryTask(TemporalPatternTask):
    """Sequence memory task (alias for TemporalPatternTask)."""
    pass


__all__ = [
    "SequenceMemoryEnvironment",
    "TemporalPatternTask",
    "SequenceMemoryTask",
]
