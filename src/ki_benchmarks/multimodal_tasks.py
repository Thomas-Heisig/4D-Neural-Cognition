"""Multimodal integration and cross-modal association tasks.

This module implements multimodal reasoning benchmarks including:
- Cross-modal pattern association
- Multimodal integration
- Sensory fusion
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


class CrossModalEnvironment(Environment):
    """Environment for testing cross-modal associations.
    
    Presents patterns in different modalities that must be associated.
    """
    
    def __init__(
        self,
        visual_dim: int = 20,
        digital_dim: int = 10,
        num_pairs: int = 5,
        seed: int = None
    ):
        """Initialize cross-modal environment.
        
        Args:
            visual_dim: Dimensionality of visual patterns
            digital_dim: Dimensionality of digital patterns
            num_pairs: Number of pattern pairs to learn
            seed: Random seed
        """
        super().__init__(seed)
        self.visual_dim = visual_dim
        self.digital_dim = digital_dim
        self.num_pairs = num_pairs
        
        # Generate associated pattern pairs
        self.visual_patterns: List[np.ndarray] = []
        self.digital_patterns: List[np.ndarray] = []
        self._generate_pattern_pairs()
        
        # State
        self.phase = "training"  # "training" or "testing"
        self.current_pair = 0
        self.test_results: List[float] = []
    
    def _generate_pattern_pairs(self) -> None:
        """Generate pairs of associated patterns."""
        self.visual_patterns = []
        self.digital_patterns = []
        
        for _ in range(self.num_pairs):
            # Generate distinct patterns
            visual = self.rng.uniform(-1, 1, size=self.visual_dim)
            digital = self.rng.uniform(-1, 1, size=self.digital_dim)
            
            self.visual_patterns.append(visual)
            self.digital_patterns.append(digital)
    
    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        self.current_step = 0
        self.episode_reward = 0.0
        self.phase = "training"
        self.current_pair = 0
        self.test_results = []
        
        observation = self._create_observation()
        info = self.get_info()
        
        return observation, info
    
    def _create_observation(self) -> Dict[str, np.ndarray]:
        """Create observation of current state."""
        if self.phase == "training":
            # Present both modalities together
            visual = self.visual_patterns[self.current_pair]
            digital = self.digital_patterns[self.current_pair]
        else:  # testing
            # Present only visual, expect digital response
            visual = self.visual_patterns[self.current_pair]
            digital = np.zeros(self.digital_dim)
        
        return {
            "vision": visual,
            "digital": digital
        }
    
    def step(
        self,
        action: Optional[np.ndarray] = None
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Execute one step."""
        self.current_step += 1
        reward = 0.0
        
        if self.phase == "training":
            # Training phase - just present pairs
            self.current_pair = (self.current_pair + 1) % self.num_pairs
            
            # After seeing all pairs, switch to testing
            if self.current_pair == 0 and self.current_step > self.num_pairs:
                self.phase = "testing"
                self.current_pair = 0
        
        elif self.phase == "testing":
            # Testing phase - evaluate recall
            if action is not None and len(action) == self.digital_dim:
                target = self.digital_patterns[self.current_pair]
                similarity = np.corrcoef(action, target)[0, 1]
                
                if not np.isnan(similarity):
                    reward = max(0.0, similarity)
                    self.test_results.append(similarity)
            
            self.current_pair += 1
        
        # Check if done
        done = self.phase == "testing" and self.current_pair >= self.num_pairs
        
        observation = self._create_observation()
        self.episode_reward += reward
        info = self.get_info()
        info["phase"] = self.phase
        
        return observation, reward, done, info
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        # Concatenate visual and digital patterns for visualization
        if self.current_pair < len(self.visual_patterns):
            visual = self.visual_patterns[self.current_pair]
            digital = self.digital_patterns[self.current_pair]
            return np.concatenate([visual, digital]).reshape(-1, 1)
        return None


class CrossModalAssociationTask(Task):
    """Cross-modal association benchmark task.
    
    Evaluates the ability to:
    - Learn associations between modalities
    - Retrieve patterns from one modality given another
    - Generalize across modalities
    """
    
    def __init__(
        self,
        visual_dim: int = 20,
        digital_dim: int = 10,
        num_pairs: int = 5,
        num_trials: int = 100,
        seed: int = None
    ):
        """Initialize cross-modal association task.
        
        Args:
            visual_dim: Dimensionality of visual patterns
            digital_dim: Dimensionality of digital patterns
            num_pairs: Number of pattern pairs
            num_trials: Number of trials for evaluation
            seed: Random seed
        """
        super().__init__(seed)
        self.env = CrossModalEnvironment(
            visual_dim, digital_dim, num_pairs, seed
        )
        self.num_trials = num_trials
    
    def evaluate(self, model: Any) -> TaskResult:
        """Evaluate model on cross-modal association.
        
        Args:
            model: Neural network model to evaluate
            
        Returns:
            Task results with cross-modal performance
        """
        total_accuracy = 0.0
        total_reward = 0.0
        total_steps = 0
        
        for trial in range(self.num_trials):
            observation, info = self.env.reset()
            done = False
            
            while not done:
                # Get action from model (placeholder)
                action = self.rng.uniform(-1, 1, size=self.env.digital_dim)
                
                observation, reward, done, info = self.env.step(action)
                total_reward += reward
                total_steps += 1
            
            # Calculate trial accuracy from test results
            if self.env.test_results:
                trial_accuracy = np.mean([
                    1.0 if r > 0.5 else 0.0 
                    for r in self.env.test_results
                ])
                total_accuracy += trial_accuracy
        
        accuracy = total_accuracy / self.num_trials
        avg_reward = total_reward / self.num_trials
        avg_steps = total_steps / self.num_trials
        
        return TaskResult(
            accuracy=accuracy,
            reward=avg_reward,
            reaction_time=avg_steps,
            stability=accuracy,
            additional_metrics={
                "association_accuracy": accuracy,
                "average_reward": avg_reward
            }
        )


class MultimodalIntegrationTask(Task):
    """Multimodal integration task.
    
    Evaluates the ability to integrate information from multiple
    sensory modalities simultaneously.
    """
    
    def __init__(
        self,
        num_modalities: int = 3,
        pattern_dim: int = 10,
        num_trials: int = 100,
        seed: int = None
    ):
        """Initialize multimodal integration task.
        
        Args:
            num_modalities: Number of sensory modalities
            pattern_dim: Dimensionality of each modality
            num_trials: Number of trials for evaluation
            seed: Random seed
        """
        super().__init__(seed)
        self.num_modalities = num_modalities
        self.pattern_dim = pattern_dim
        self.num_trials = num_trials
    
    def evaluate(self, model: Any) -> TaskResult:
        """Evaluate model on multimodal integration.
        
        Args:
            model: Neural network model to evaluate
            
        Returns:
            Task results with multimodal performance
        """
        successes = 0
        total_reward = 0.0
        
        for trial in range(self.num_trials):
            # Generate target pattern
            target = self.rng.uniform(-1, 1, size=self.pattern_dim)
            
            # Present partial information in each modality
            modality_inputs = []
            for _ in range(self.num_modalities):
                # Each modality gets corrupted version of target
                noise = self.rng.normal(0, 0.3, size=self.pattern_dim)
                modality_input = target + noise
                modality_inputs.append(modality_input)
            
            # Model should integrate to recover target
            # (Placeholder - actual model would process these inputs)
            integrated = np.mean(modality_inputs, axis=0)
            
            # Evaluate integration quality
            similarity = np.corrcoef(integrated, target)[0, 1]
            if not np.isnan(similarity):
                reward = max(0.0, similarity)
                total_reward += reward
                if similarity > 0.7:
                    successes += 1
        
        accuracy = successes / self.num_trials
        avg_reward = total_reward / self.num_trials
        
        return TaskResult(
            accuracy=accuracy,
            reward=avg_reward,
            reaction_time=1.0,  # Single-step task
            stability=accuracy,
            additional_metrics={
                "integration_accuracy": accuracy,
                "average_similarity": avg_reward
            }
        )


__all__ = [
    "CrossModalEnvironment",
    "CrossModalAssociationTask",
    "MultimodalIntegrationTask",
]
