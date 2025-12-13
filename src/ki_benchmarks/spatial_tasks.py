"""Spatial reasoning tasks for cognitive benchmarking.

This module implements spatial reasoning benchmarks including:
- Grid world navigation
- Hidden object location
- Spatial relationship reasoning
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


class GridWorldEnvironment(Environment):
    """Simple grid world environment for spatial reasoning.
    
    The agent must navigate a 2D grid to find hidden objects.
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (10, 10),
        num_objects: int = 3,
        seed: int = None
    ):
        """Initialize grid world.
        
        Args:
            grid_size: Size of the grid (width, height)
            num_objects: Number of hidden objects
            seed: Random seed
        """
        super().__init__(seed)
        self.grid_size = grid_size
        self.num_objects = num_objects
        
        # State
        self.agent_pos = np.array([0, 0])
        self.object_positions: List[np.ndarray] = []
        self.found_objects: List[bool] = []
        self.grid = np.zeros(grid_size)
    
    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Place agent at random position
        self.agent_pos = self.rng.integers(0, self.grid_size, size=2)
        
        # Place objects at random positions
        self.object_positions = []
        self.found_objects = []
        for _ in range(self.num_objects):
            pos = self.rng.integers(0, self.grid_size, size=2)
            self.object_positions.append(pos)
            self.found_objects.append(False)
        
        # Create observation
        observation = self._create_observation()
        info = self.get_info()
        
        return observation, info
    
    def _create_observation(self) -> Dict[str, np.ndarray]:
        """Create observation of current state."""
        # Create grid with agent position
        grid = np.zeros(self.grid_size)
        grid[tuple(self.agent_pos)] = 1.0
        
        # Add partial visibility of objects (if close enough)
        for i, obj_pos in enumerate(self.object_positions):
            if not self.found_objects[i]:
                distance = np.linalg.norm(self.agent_pos - obj_pos)
                if distance < 3.0:  # Visibility radius
                    grid[tuple(obj_pos)] = 0.5
        
        return {
            "vision": grid.flatten(),
            "digital": np.array([*self.agent_pos, len([f for f in self.found_objects if f])])
        }
    
    def step(
        self,
        action: Optional[np.ndarray] = None
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Execute one step."""
        self.current_step += 1
        reward = -0.01  # Small penalty for each step
        
        # Execute action (if provided)
        if action is not None and len(action) >= 2:
            # Action is movement direction
            move = np.round(action[:2]).astype(int)
            new_pos = self.agent_pos + move
            
            # Clip to grid boundaries
            new_pos = np.clip(new_pos, 0, np.array(self.grid_size) - 1)
            self.agent_pos = new_pos
        
        # Check if agent found an object
        for i, obj_pos in enumerate(self.object_positions):
            if not self.found_objects[i]:
                if np.array_equal(self.agent_pos, obj_pos):
                    self.found_objects[i] = True
                    reward = 1.0  # Reward for finding object
        
        # Check if done
        done = all(self.found_objects) or self.current_step >= 100
        
        observation = self._create_observation()
        self.episode_reward += reward
        info = self.get_info()
        
        return observation, reward, done, info
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        grid = np.zeros(self.grid_size)
        
        # Draw agent
        grid[tuple(self.agent_pos)] = 1.0
        
        # Draw objects
        for i, obj_pos in enumerate(self.object_positions):
            if not self.found_objects[i]:
                grid[tuple(obj_pos)] = 0.5
            else:
                grid[tuple(obj_pos)] = 0.3  # Found objects dimmed
        
        return grid


class SpatialReasoningTask(Task):
    """Spatial reasoning benchmark task.
    
    Evaluates the ability to:
    - Navigate in space
    - Remember object locations
    - Infer hidden object positions
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (10, 10),
        num_objects: int = 3,
        num_trials: int = 100,
        seed: int = None
    ):
        """Initialize spatial reasoning task.
        
        Args:
            grid_size: Size of grid world
            num_objects: Number of hidden objects
            num_trials: Number of trials for evaluation
            seed: Random seed
        """
        super().__init__(seed)
        self.env = GridWorldEnvironment(grid_size, num_objects, seed)
        self.num_trials = num_trials
    
    def evaluate(self, model: Any) -> TaskResult:
        """Evaluate model on spatial reasoning.
        
        Args:
            model: Neural network model to evaluate
            
        Returns:
            Task results with spatial reasoning performance
            
        Note:
            This is a baseline implementation. Integration with actual neural
            network models requires implementing a policy network that maps
            observations to actions. See docs/benchmarks/README.md for details.
        """
        successes = 0
        total_steps = 0
        total_reward = 0.0
        
        for trial in range(self.num_trials):
            observation, info = self.env.reset()
            done = False
            steps = 0
            
            while not done and steps < 100:
                # TODO: Integrate with actual model
                # action = model.get_action(observation)
                # Placeholder: random exploration policy
                action = self.rng.uniform(-1, 1, size=2)
                
                observation, reward, done, info = self.env.step(action)
                total_reward += reward
                steps += 1
            
            # Check if all objects were found
            if all(self.env.found_objects):
                successes += 1
            
            total_steps += steps
        
        accuracy = successes / self.num_trials
        avg_steps = total_steps / self.num_trials
        avg_reward = total_reward / self.num_trials
        
        return TaskResult(
            accuracy=accuracy,
            reward=avg_reward,
            reaction_time=avg_steps,
            stability=1.0 - (avg_steps / 100.0),  # Lower steps = more stable
            additional_metrics={
                "success_rate": accuracy,
                "average_steps": avg_steps
            }
        )


class GridWorldTask(SpatialReasoningTask):
    """Grid world navigation task (alias for SpatialReasoningTask)."""
    pass


__all__ = [
    "GridWorldEnvironment",
    "SpatialReasoningTask",
    "GridWorldTask",
]
