"""Virtual Body for embodied AI with sensorimotor learning.

This module implements a virtual body that enables the neural network to
interact with a simulated environment through motor commands and proprioception.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ProprioceptiveSensor:
    """Sensor for body self-awareness (proprioception).
    
    Tracks joint angles, muscle tensions, and body configuration to provide
    internal awareness of the body state.
    
    Attributes:
        joint_angles: Current angles of all joints
        muscle_tensions: Current tension levels in muscles
        body_velocity: Current velocity of body center
    """
    
    def __init__(self):
        """Initialize proprioceptive sensor."""
        self.joint_angles: Dict[str, float] = {}
        self.muscle_tensions: Dict[str, float] = {}
        self.body_velocity: np.ndarray = np.zeros(3)
        self.body_acceleration: np.ndarray = np.zeros(3)
    
    def update(self, skeleton_state: Dict) -> Dict:
        """Update proprioceptive state from skeleton.
        
        Args:
            skeleton_state: Current state of the skeleton
            
        Returns:
            Dictionary of proprioceptive signals
        """
        # Update joint angles
        self.joint_angles = skeleton_state.get('joint_angles', {})
        
        # Calculate muscle tensions from joint forces
        joint_forces = skeleton_state.get('joint_forces', {})
        for joint_id, force in joint_forces.items():
            self.muscle_tensions[joint_id] = np.linalg.norm(force)
        
        # Update velocity and acceleration
        position = skeleton_state.get('center_position', np.zeros(3))
        if hasattr(self, '_last_position'):
            velocity = position - self._last_position
            if hasattr(self, '_last_velocity'):
                self.body_acceleration = velocity - self._last_velocity
            self._last_velocity = self.body_velocity
            self.body_velocity = velocity
        self._last_position = position
        
        return {
            'joint_angles': self.joint_angles,
            'muscle_tensions': self.muscle_tensions,
            'velocity': self.body_velocity,
            'acceleration': self.body_acceleration,
        }


class VirtualBody:
    """Virtual body for embodied neural network.
    
    Provides a simulated physical body that the neural network can control
    through motor commands and sense through proprioception. Enables
    sensorimotor learning and embodied cognition.
    
    Attributes:
        body_type: Type of body (e.g., "humanoid", "quadruped")
        skeleton: Skeletal structure with joints and links
        muscles: Virtual muscles for actuation
        proprioception: Proprioceptive sensor for self-awareness
        position: Current position in 3D space
        orientation: Current orientation (quaternion)
    """
    
    # Muscle dynamics constants
    FATIGUE_ACCUMULATION_RATE = 0.001  # Fatigue per activation unit
    FATIGUE_RECOVERY_RATE = 0.01       # Recovery per timestep
    
    def __init__(
        self,
        body_type: str = "humanoid",
        num_joints: int = 12,
        max_force: float = 100.0,
    ):
        """Initialize virtual body.
        
        Args:
            body_type: Type of body skeleton
            num_joints: Number of controllable joints
            max_force: Maximum force per muscle/joint
        """
        self.body_type = body_type
        self.skeleton = self._load_skeleton(body_type, num_joints)
        self.muscles: Dict[str, Dict] = {}
        self.proprioception = ProprioceptiveSensor()
        
        # Initialize muscles for each joint
        for joint_id in self.skeleton['joints'].keys():
            self.muscles[joint_id] = {
                'max_force': max_force,
                'current_activation': 0.0,
                'fatigue': 0.0,
            }
        
        # Body state in 3D space
        self.position = np.zeros(3)
        self.orientation = np.array([0, 0, 0, 1])  # Quaternion (x, y, z, w)
        
        logger.info(f"Initialized {body_type} body with {num_joints} joints")
    
    def _load_skeleton(self, body_type: str, num_joints: int) -> Dict:
        """Load skeleton configuration for body type.
        
        Args:
            body_type: Type of skeleton to load
            num_joints: Number of joints to create
            
        Returns:
            Dictionary defining skeleton structure
        """
        # Simple skeleton definition
        joints = {}
        links = []
        
        if body_type == "humanoid":
            # Create hierarchical joint structure
            joint_names = [
                "spine", "neck", "head",
                "left_shoulder", "left_elbow", "left_wrist",
                "right_shoulder", "right_elbow", "right_wrist",
                "left_hip", "left_knee", "left_ankle",
                "right_hip", "right_knee", "right_ankle",
            ][:num_joints]
            
            for i, name in enumerate(joint_names):
                joints[name] = {
                    'id': i,
                    'position': np.random.randn(3) * 0.1,  # Random initial position
                    'angle': 0.0,
                    'force': np.zeros(3),
                    'limits': (-np.pi, np.pi),
                }
        
        elif body_type == "quadruped":
            # Four-legged body
            joint_names = [
                "spine", "neck", "head",
                "front_left_shoulder", "front_left_elbow",
                "front_right_shoulder", "front_right_elbow",
                "rear_left_hip", "rear_left_knee",
                "rear_right_hip", "rear_right_knee",
            ][:num_joints]
            
            for i, name in enumerate(joint_names):
                joints[name] = {
                    'id': i,
                    'position': np.random.randn(3) * 0.1,
                    'angle': 0.0,
                    'force': np.zeros(3),
                    'limits': (-np.pi, np.pi),
                }
        
        else:
            # Generic skeleton
            for i in range(num_joints):
                joints[f"joint_{i}"] = {
                    'id': i,
                    'position': np.random.randn(3) * 0.1,
                    'angle': 0.0,
                    'force': np.zeros(3),
                    'limits': (-np.pi, np.pi),
                }
        
        return {
            'joints': joints,
            'links': links,
            'center_position': np.zeros(3),
        }
    
    def execute_motor_command(self, neural_output: Dict) -> Dict:
        """Execute motor commands from neural network.
        
        Translates neural activity into muscle activations and updates
        body state through simplified physics simulation.
        
        Args:
            neural_output: Dictionary containing motor neuron activities
            
        Returns:
            Kinematic feedback for self-observation
        """
        # Decode muscle activations from neural output
        muscle_activations = self.decode_motor_pattern(neural_output)
        
        # Apply forces to joints
        for muscle_id, activation in muscle_activations.items():
            if muscle_id in self.muscles:
                muscle = self.muscles[muscle_id]
                
                # Calculate force considering fatigue
                force_magnitude = activation * muscle['max_force'] * (1.0 - muscle['fatigue'])
                
                # Apply force to corresponding joint
                if muscle_id in self.skeleton['joints']:
                    joint = self.skeleton['joints'][muscle_id]
                    
                    # Simplified: force rotates joint
                    joint['angle'] += force_magnitude * 0.01
                    
                    # Clamp to joint limits
                    min_angle, max_angle = joint['limits']
                    joint['angle'] = np.clip(joint['angle'], min_angle, max_angle)
                    
                    joint['force'] = np.array([0, force_magnitude, 0])
                
                # Update muscle state
                muscle['current_activation'] = activation
                muscle['fatigue'] = min(1.0, muscle['fatigue'] + self.FATIGUE_ACCUMULATION_RATE * activation)
        
        # Update body position based on joint configuration (simplified)
        self._update_physics()
        
        # Get proprioceptive feedback
        kinematic_feedback = self.get_kinematic_feedback()
        
        return kinematic_feedback
    
    def decode_motor_pattern(self, neural_output: Dict) -> Dict[str, float]:
        """Decode neural activity pattern into muscle activations.
        
        Args:
            neural_output: Neural activity from motor cortex
            
        Returns:
            Dictionary mapping muscle IDs to activation levels (0-1)
        """
        muscle_activations = {}
        
        # Extract motor neuron activities
        motor_activities = neural_output.get('motor_neurons', {})
        
        # Map to muscles (simple 1:1 mapping)
        for neuron_id, activity in motor_activities.items():
            # Normalize activity to [0, 1]
            activation = max(0.0, min(1.0, activity))
            
            # Map to muscle (using modulo for simplicity)
            muscle_ids = list(self.muscles.keys())
            if muscle_ids:
                muscle_id = muscle_ids[neuron_id % len(muscle_ids)]
                muscle_activations[muscle_id] = activation
        
        return muscle_activations
    
    def _update_physics(self) -> None:
        """Update body physics based on joint configuration.
        
        This is a simplified physics simulation. In a full implementation,
        this would use a physics engine like PyBullet or MuJoCo.
        """
        # Calculate center of mass from joint positions
        joint_positions = [j['position'] for j in self.skeleton['joints'].values()]
        if joint_positions:
            self.skeleton['center_position'] = np.mean(joint_positions, axis=0)
            self.position = self.skeleton['center_position'].copy()
        
        # Apply gravity and ground contact (simplified)
        if self.position[2] > 0:  # Above ground
            self.position[2] -= 0.01  # Gravity
        else:
            self.position[2] = 0  # Ground contact
        
        # Recover from fatigue
        for muscle in self.muscles.values():
            muscle['fatigue'] = max(0.0, muscle['fatigue'] - self.FATIGUE_RECOVERY_RATE)
    
    def get_kinematic_feedback(self) -> Dict:
        """Get kinematic feedback for self-observation.
        
        Returns:
            Dictionary containing body state information
        """
        # Update proprioception
        skeleton_state = {
            'joint_angles': {jid: j['angle'] for jid, j in self.skeleton['joints'].items()},
            'joint_forces': {jid: j['force'] for jid, j in self.skeleton['joints'].items()},
            'center_position': self.skeleton['center_position'],
        }
        
        proprio_feedback = self.proprioception.update(skeleton_state)
        
        return {
            'position': self.position.copy(),
            'orientation': self.orientation.copy(),
            'joint_angles': proprio_feedback['joint_angles'],
            'muscle_tensions': proprio_feedback['muscle_tensions'],
            'velocity': proprio_feedback['velocity'],
            'touch_sensors': {},  # Placeholder for tactile feedback
        }
    
    def get_state(self) -> Dict:
        """Get complete body state.
        
        Returns:
            Dictionary with all body state information
        """
        return {
            'position': self.position.tolist(),
            'orientation': self.orientation.tolist(),
            'joints': {
                jid: {
                    'angle': j['angle'],
                    'force': j['force'].tolist(),
                }
                for jid, j in self.skeleton['joints'].items()
            },
            'muscles': {
                mid: {
                    'activation': m['current_activation'],
                    'fatigue': m['fatigue'],
                }
                for mid, m in self.muscles.items()
            },
        }
    
    def reset(self) -> None:
        """Reset body to initial state."""
        self.position = np.zeros(3)
        self.orientation = np.array([0, 0, 0, 1])
        
        for joint in self.skeleton['joints'].values():
            joint['angle'] = 0.0
            joint['force'] = np.zeros(3)
        
        for muscle in self.muscles.values():
            muscle['current_activation'] = 0.0
            muscle['fatigue'] = 0.0
        
        logger.info("Virtual body reset to initial state")
