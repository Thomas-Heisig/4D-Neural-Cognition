"""Experiment: Sensorimotor Learning with STDP and Reward.

This experiment tests the sensorimotor learning loop by training an agent
to perform simple reaching tasks. It validates that the system can learn
motor skills through reinforcement learning combined with STDP.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain_model import BrainModel
from embodiment.virtual_body import VirtualBody
from embodiment.sensorimotor_learner import SensorimotorReinforcementLearner
from simulation import Simulation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReachTargetTask:
    """Simple reaching task for sensorimotor learning.
    
    The agent must learn to move its arm to reach a target position.
    Reward is based on proximity to target.
    
    Attributes:
        target_position: 3D position of target
        success_threshold: Distance threshold for success
        max_steps: Maximum steps per episode
    """
    
    def __init__(
        self,
        target_position: np.ndarray = None,
        success_threshold: float = 0.1,
        max_steps: int = 50,
    ):
        """Initialize reaching task.
        
        Args:
            target_position: Target 3D position
            success_threshold: Success distance threshold
            max_steps: Max steps per episode
        """
        if target_position is None:
            target_position = np.array([0.5, 0.5, 0.5])
        
        self.target_position = target_position
        self.success_threshold = success_threshold
        self.max_steps = max_steps
        
        self.current_step = 0
        self.episode_rewards: List[float] = []
        
        logger.info(
            f"Initialized ReachTargetTask (target={target_position}, "
            f"threshold={success_threshold})"
        )
    
    def reset(self) -> Dict:
        """Reset task for new episode.
        
        Returns:
            Initial observation
        """
        self.current_step = 0
        
        return {
            'target_position': self.target_position.copy(),
            'step': self.current_step,
        }
    
    def step(self, body_position: np.ndarray) -> Dict:
        """Execute one step of the task.
        
        Args:
            body_position: Current body/end-effector position
            
        Returns:
            Dictionary with reward, done, and info
        """
        self.current_step += 1
        
        # Calculate distance to target
        distance = np.linalg.norm(body_position - self.target_position)
        
        # Reward is inversely proportional to distance
        # Higher reward for being closer
        reward = 1.0 / (1.0 + distance)
        
        # Bonus for reaching target
        if distance < self.success_threshold:
            reward += 10.0
            done = True
        elif self.current_step >= self.max_steps:
            done = True
        else:
            done = False
        
        info = {
            'distance': distance,
            'success': distance < self.success_threshold,
        }
        
        return {
            'reward': reward,
            'done': done,
            'info': info,
        }
    
    def get_statistics(self) -> Dict:
        """Get task statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_episodes': len(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'success_rate': sum(
                1 for r in self.episode_rewards if r > 5.0
            ) / len(self.episode_rewards) if self.episode_rewards else 0.0,
        }


def create_simple_brain_model() -> BrainModel:
    """Create a simple brain model for testing.
    
    Returns:
        BrainModel instance
    """
    config = {
        "lattice_shape": [10, 10, 5, 15],  # Small model
        "neuron_model": {
            "type": "lif",
            "params_default": {
                "threshold": -50.0,
                "reset_potential": -65.0,
                "tau_membrane": 20.0,
                "refractory_period": 2,
            }
        },
        "cell_lifecycle": {
            "neurogenesis_rate": 0.0,
            "apoptosis_threshold": 0.0,
        },
        "plasticity": {
            "stdp_enabled": True,
            "learning_rate": 0.01,
        },
        "senses": {
            "digital": {"areal": "V1"}
        },
        "areas": [
            {
                "name": "M1",  # Motor cortex
                "bounds": {"x": [0, 10], "y": [0, 10], "z": [0, 5], "w": [10, 10]},
                "neuron_type": "excitatory",
            },
            {
                "name": "S1",  # Somatosensory cortex
                "bounds": {"x": [0, 10], "y": [0, 10], "z": [0, 5], "w": [6, 6]},
                "neuron_type": "excitatory",
            },
            {
                "name": "V1",  # Visual cortex
                "bounds": {"x": [0, 10], "y": [0, 10], "z": [0, 5], "w": [0, 2]},
                "neuron_type": "excitatory",
            },
        ]
    }
    
    return BrainModel(config=config)


def run_experiment(
    body_type: str = "simple_arm",
    num_episodes: int = 100,
    learning_algorithm: str = "stdp_plus_reward",
    output_file: str = None,
) -> Dict:
    """Run sensorimotor learning experiment.
    
    Args:
        body_type: Type of body to use
        num_episodes: Number of learning episodes
        learning_algorithm: Learning algorithm to use
        output_file: Optional file to save results
        
    Returns:
        Dictionary with experiment results
    """
    logger.info(
        f"Starting sensorimotor learning experiment: "
        f"body={body_type}, episodes={num_episodes}, "
        f"algorithm={learning_algorithm}"
    )
    
    # Create components
    brain = create_simple_brain_model()
    body = VirtualBody(body_type=body_type, num_joints=6)
    task = ReachTargetTask()
    
    # Create simulation
    sim = Simulation(brain)
    
    # Create learner
    learner = SensorimotorReinforcementLearner(
        virtual_body=body,
        brain_model=brain,
        learning_rate=0.01,
    )
    
    # Training loop
    episode_results = []
    
    for episode in range(num_episodes):
        # Start episode
        learner.start_episode()
        task_state = task.reset()
        body.reset()
        
        episode_reward = 0.0
        episode_steps = 0
        
        # Episode loop
        for step in range(task.max_steps):
            # Generate motor command (random exploration for now)
            # In a full implementation, this would come from brain
            num_joints = len(body.skeleton['joints'])
            motor_command = {
                'motor_neurons': {
                    i: np.random.rand() * 0.5
                    for i in range(num_joints)
                }
            }
            
            # Execute action
            feedback = body.execute_motor_command(motor_command)
            
            # Get task reward
            task_result = task.step(body.position)
            
            # Learn from interaction
            learning_result = learner.learn_from_interaction(
                action=motor_command,
                resulting_feedback=feedback,
                external_reward=task_result['reward'],
            )
            
            episode_reward += task_result['reward']
            episode_steps += 1
            
            if task_result['done']:
                break
        
        # End episode
        episode_summary = learner.end_episode()
        
        episode_results.append({
            'episode': episode,
            'reward': episode_reward,
            'steps': episode_steps,
            'success': task_result['info']['success'],
            'distance': task_result['info']['distance'],
            'learning_progress': episode_summary,
        })
        
        task.episode_rewards.append(episode_reward)
        
        # Log progress
        if (episode + 1) % 10 == 0:
            recent_rewards = [r['reward'] for r in episode_results[-10:]]
            recent_success = sum(r['success'] for r in episode_results[-10:])
            logger.info(
                f"Episode {episode + 1}/{num_episodes}: "
                f"avg_reward={np.mean(recent_rewards):.2f}, "
                f"success_rate={recent_success}/10"
            )
    
    # Compile results
    results = {
        'config': {
            'body_type': body_type,
            'num_episodes': num_episodes,
            'learning_algorithm': learning_algorithm,
        },
        'episode_results': episode_results,
        'task_statistics': task.get_statistics(),
        'learner_statistics': learner.get_statistics(),
        'final_progress': learner.calculate_learning_progress(),
    }
    
    # Save results if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*60)
    logger.info(f"Total episodes: {num_episodes}")
    logger.info(f"Average reward: {results['task_statistics']['avg_reward']:.3f}")
    logger.info(f"Success rate: {results['task_statistics']['success_rate']:.1%}")
    logger.info(f"Learning progress:")
    for key, value in results['final_progress'].items():
        logger.info(f"  {key}: {value}")
    logger.info("="*60)
    
    return results


def main():
    """Main entry point for experiment."""
    parser = argparse.ArgumentParser(
        description='Run sensorimotor learning experiment'
    )
    parser.add_argument(
        '--task',
        type=str,
        default='reach_target',
        choices=['reach_target'],
        help='Task to learn'
    )
    parser.add_argument(
        '--body_type',
        type=str,
        default='simple_arm',
        choices=['simple_arm', 'humanoid', 'quadruped'],
        help='Type of body to use'
    )
    parser.add_argument(
        '--learning_algorithm',
        type=str,
        default='stdp_plus_reward',
        choices=['stdp_plus_reward'],
        help='Learning algorithm'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of episodes to run'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for results (JSON)'
    )
    
    args = parser.parse_args()
    
    # Map body type (simple_arm is same as humanoid for this experiment)
    body_type = 'humanoid' if args.body_type == 'simple_arm' else args.body_type
    
    # Run experiment
    results = run_experiment(
        body_type=body_type,
        num_episodes=args.episodes,
        learning_algorithm=args.learning_algorithm,
        output_file=args.output,
    )
    
    return results


if __name__ == '__main__':
    main()
