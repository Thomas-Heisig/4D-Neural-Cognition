"""Experiment: Autonomous Exploration with Intrinsic Motivation.

This experiment tests the autonomous learning loop by letting the agent
explore an environment without external instructions. Success is measured
by coverage of the space and emergent learning patterns.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain_model import BrainModel
from embodiment.virtual_body import VirtualBody
from embodiment.sensorimotor_learner import SensorimotorReinforcementLearner
from consciousness.self_perception_stream import SelfPerceptionStream
from autonomous_learning_loop import AutonomousLearningAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleRoom:
    """Simple room environment for exploration.
    
    The room contains objects that the agent can discover and interact with.
    
    Attributes:
        size: Size of the room (x, y, z)
        objects: List of objects in the room
        agent_position: Current agent position
    """
    
    def __init__(self, size: tuple = (5, 5, 3), num_objects: int = 10):
        """Initialize room environment.
        
        Args:
            size: Room dimensions (x, y, z)
            num_objects: Number of objects to place
        """
        self.size = np.array(size)
        self.objects = self._place_objects(num_objects)
        self.agent_position = np.array([2.5, 2.5, 0.5])  # Start in center
        
        # Track discovered objects
        self.discovered_objects = set()
        self.discovery_threshold = 0.5  # Distance threshold for discovery
        
        logger.info(
            f"Initialized SimpleRoom with size {size} and {num_objects} objects"
        )
    
    def _place_objects(self, num_objects: int) -> List[Dict]:
        """Place objects randomly in the room.
        
        Args:
            num_objects: Number of objects
            
        Returns:
            List of object dictionaries
        """
        objects = []
        for i in range(num_objects):
            position = np.array([
                np.random.rand() * self.size[0],
                np.random.rand() * self.size[1],
                np.random.rand() * self.size[2],
            ])
            
            objects.append({
                'id': i,
                'position': position,
                'type': np.random.choice(['cube', 'sphere', 'cylinder']),
                'discovered': False,
            })
        
        return objects
    
    def update_agent_position(self, movement: np.ndarray) -> Dict:
        """Update agent position based on movement.
        
        Args:
            movement: Movement vector
            
        Returns:
            Environment feedback
        """
        # Apply movement with bounds checking
        new_position = self.agent_position + movement
        new_position = np.clip(new_position, [0, 0, 0], self.size)
        
        self.agent_position = new_position
        
        # Check for object discoveries
        newly_discovered = []
        for obj in self.objects:
            if obj['id'] not in self.discovered_objects:
                distance = np.linalg.norm(obj['position'] - self.agent_position)
                if distance < self.discovery_threshold:
                    self.discovered_objects.add(obj['id'])
                    obj['discovered'] = True
                    newly_discovered.append(obj)
        
        # Calculate sensory feedback
        nearby_objects = self._get_nearby_objects(radius=2.0)
        
        return {
            'position': self.agent_position.copy(),
            'velocity': movement,
            'nearby_objects': nearby_objects,
            'newly_discovered': newly_discovered,
            'total_discovered': len(self.discovered_objects),
            'discovery_rate': len(self.discovered_objects) / len(self.objects),
            'joint_angles': {},  # Placeholder
            'timestamp': time.time(),
            'body_health': 1.0,
        }
    
    def _get_nearby_objects(self, radius: float) -> List[Dict]:
        """Get objects within radius of agent.
        
        Args:
            radius: Search radius
            
        Returns:
            List of nearby objects
        """
        nearby = []
        for obj in self.objects:
            distance = np.linalg.norm(obj['position'] - self.agent_position)
            if distance < radius:
                nearby.append({
                    'id': obj['id'],
                    'distance': distance,
                    'type': obj['type'],
                    'discovered': obj['discovered'],
                })
        
        return nearby
    
    def get_exploration_metrics(self) -> Dict:
        """Get exploration statistics.
        
        Returns:
            Dictionary with metrics
        """
        return {
            'total_objects': len(self.objects),
            'discovered_objects': len(self.discovered_objects),
            'discovery_rate': len(self.discovered_objects) / len(self.objects),
            'undiscovered_objects': len(self.objects) - len(self.discovered_objects),
        }


def create_simple_brain_model() -> BrainModel:
    """Create a simple brain model for autonomous learning.
    
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


def run_autonomous_exploration(
    environment: str = "simple_room",
    duration: int = 1000,
    motivation: str = "curiosity_and_competence",
    output_file: str = None,
) -> Dict:
    """Run autonomous exploration experiment.
    
    Args:
        environment: Environment type
        duration: Number of cycles to run
        motivation: Type of motivation to use
        output_file: Optional file to save results
        
    Returns:
        Dictionary with experiment results
    """
    logger.info(
        f"Starting autonomous exploration experiment: "
        f"env={environment}, duration={duration}, motivation={motivation}"
    )
    
    # Create components
    brain = create_simple_brain_model()
    body = VirtualBody(body_type="humanoid", num_joints=6)
    self_stream = SelfPerceptionStream(
        update_frequency_hz=100.0,
        buffer_duration_seconds=10.0
    )
    learner = SensorimotorReinforcementLearner(
        virtual_body=body,
        brain_model=brain,
        learning_rate=0.01,
    )
    
    # Create autonomous agent
    agent = AutonomousLearningAgent(
        embodiment=body,
        brain=brain,
        self_stream=self_stream,
        learner=learner,
        state_dim=10,
        action_dim=6,
    )
    
    # Create environment
    room = SimpleRoom(size=(5, 5, 3), num_objects=10)
    
    # Exploration loop
    cycle_results = []
    exploration_metrics_history = []
    
    logger.info("Starting autonomous exploration...")
    start_time = time.time()
    
    for cycle in range(duration):
        # Get environment context
        environment_context = {
            'position': room.agent_position,
            'nearby_objects': room._get_nearby_objects(radius=2.0),
            'timestamp': cycle,
            'body_health': 1.0,
        }
        
        # Run autonomous cycle
        cycle_result = agent.run_autonomous_cycle(environment_context)
        
        # Extract movement from cycle result
        # (In real implementation, this would come from motor output)
        movement = np.random.randn(3) * 0.1  # Small random movement
        
        # Update environment
        env_feedback = room.update_agent_position(movement)
        
        # Record metrics
        cycle_results.append({
            'cycle': cycle,
            'agent_position': room.agent_position.tolist(),
            'goal_type': cycle_result['goal']['type'] if cycle_result['goal'] else None,
            'strategy': cycle_result['strategy'],
            'prediction_error': cycle_result['prediction_error'],
            'world_model_accuracy': cycle_result['world_model_accuracy'],
            'newly_discovered': len(env_feedback['newly_discovered']),
            'total_discovered': env_feedback['total_discovered'],
        })
        
        exploration_metrics = room.get_exploration_metrics()
        exploration_metrics_history.append(exploration_metrics)
        
        # Log progress periodically
        if (cycle + 1) % 100 == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"Cycle {cycle + 1}/{duration}: "
                f"discovered {exploration_metrics['discovered_objects']}/{exploration_metrics['total_objects']} objects "
                f"({exploration_metrics['discovery_rate']:.1%}), "
                f"strategy={cycle_result['strategy']}, "
                f"elapsed={elapsed:.1f}s"
            )
    
    elapsed_time = time.time() - start_time
    
    # Compile results
    final_metrics = room.get_exploration_metrics()
    agent_stats = agent.get_statistics()
    
    results = {
        'config': {
            'environment': environment,
            'duration': duration,
            'motivation': motivation,
        },
        'cycle_results': cycle_results,
        'exploration_metrics': final_metrics,
        'agent_statistics': agent_stats,
        'elapsed_time_seconds': elapsed_time,
        'cycles_per_second': duration / elapsed_time,
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
    logger.info("AUTONOMOUS EXPLORATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total cycles: {duration}")
    logger.info(f"Elapsed time: {elapsed_time:.1f}s ({duration/elapsed_time:.1f} cycles/s)")
    logger.info(f"Objects discovered: {final_metrics['discovered_objects']}/{final_metrics['total_objects']} "
                f"({final_metrics['discovery_rate']:.1%})")
    logger.info(f"Goals pursued: {agent_stats['goal_history_length']}")
    logger.info(f"Strategy changes: {agent_stats['strategy_changes']}")
    logger.info(f"Final world model accuracy: {agent_stats['world_model_accuracy']:.3f}")
    logger.info(f"Current strategy: {agent_stats['current_strategy']}")
    logger.info("="*60)
    
    # Success criteria: discovered 80% of objects in 1000 cycles
    success = final_metrics['discovery_rate'] >= 0.8
    logger.info(f"SUCCESS CRITERION: {'✓ PASSED' if success else '✗ FAILED'} "
                f"(target: 80% discovery rate)")
    
    return results


def main():
    """Main entry point for experiment."""
    parser = argparse.ArgumentParser(
        description='Run autonomous exploration experiment'
    )
    parser.add_argument(
        '--environment',
        type=str,
        default='simple_room',
        choices=['simple_room'],
        help='Environment to explore'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=1000,
        help='Number of cycles to run'
    )
    parser.add_argument(
        '--motivation',
        type=str,
        default='curiosity_and_competence',
        choices=['curiosity_and_competence', 'exploration', 'competence'],
        help='Type of intrinsic motivation'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for results (JSON)'
    )
    
    args = parser.parse_args()
    
    # Run experiment
    results = run_autonomous_exploration(
        environment=args.environment,
        duration=args.duration,
        motivation=args.motivation,
        output_file=args.output,
    )
    
    return results


if __name__ == '__main__':
    main()
