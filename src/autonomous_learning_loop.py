"""Autonomous Learning Loop for self-directed cognitive development.

This module implements the autonomous learning cycle that closes the loop
between perception, action, and adaptive learning. The system can:
1. Set intrinsic goals based on curiosity and competence
2. Simulate actions in an internal world model
3. Learn from prediction errors and adapt its self-model
4. Switch learning strategies based on success/failure patterns

This represents the transition from reactive to autonomous intelligence.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
import numpy as np
import logging

if TYPE_CHECKING:
    from .embodiment.virtual_body import VirtualBody
    from .brain_model import BrainModel
    from .consciousness.self_perception_stream import SelfPerceptionStream
    from .embodiment.sensorimotor_learner import SensorimotorReinforcementLearner

logger = logging.getLogger(__name__)


class LearningStrategy(Enum):
    """Available learning strategies."""
    EXPLORE = "explore"  # Random exploration
    EXPLOIT = "exploit"  # Use learned policies
    IMITATE = "imitate"  # Learn from demonstrations
    CURIOUS = "curious"  # Seek novelty
    CONSOLIDATE = "consolidate"  # Practice and refine


class GoalType(Enum):
    """Types of intrinsic goals."""
    REDUCE_PREDICTION_ERROR = "reduce_prediction_error"  # Neugierde/Curiosity
    MAXIMIZE_NOVEL_SENSATIONS = "maximize_novel_sensations"  # Erkundung/Exploration
    MASTER_MOTOR_SKILL = "master_motor_skill"  # Kompetenzstreben/Competence
    MAINTAIN_BODY_INTEGRITY = "maintain_body_integrity"  # Homöostase/Homeostasis


class IntrinsicMotivationEngine:
    """Engine for generating intrinsic goals and motivation.
    
    Implements autonomous goal generation based on:
    1. Curiosity (prediction error reduction)
    2. Exploration (novel sensation seeking)
    3. Competence (motor skill mastery)
    4. Homeostasis (body integrity maintenance)
    
    Attributes:
        goal_priorities: Weight for each goal type
        current_goal: Currently active goal
        goal_history: History of pursued goals
    """
    
    def __init__(
        self,
        curiosity_weight: float = 0.3,
        exploration_weight: float = 0.3,
        competence_weight: float = 0.3,
        homeostasis_weight: float = 0.1,
    ):
        """Initialize intrinsic motivation engine.
        
        Args:
            curiosity_weight: Priority for curiosity-driven goals
            exploration_weight: Priority for exploration goals
            competence_weight: Priority for competence goals
            homeostasis_weight: Priority for homeostasis
        """
        self.goal_priorities = {
            GoalType.REDUCE_PREDICTION_ERROR: curiosity_weight,
            GoalType.MAXIMIZE_NOVEL_SENSATIONS: exploration_weight,
            GoalType.MASTER_MOTOR_SKILL: competence_weight,
            GoalType.MAINTAIN_BODY_INTEGRITY: homeostasis_weight,
        }
        
        self.current_goal: Optional[Dict] = None
        self.goal_history: List[Dict] = []
        
        # State tracking for goal generation
        self.recent_prediction_errors: List[float] = []
        self.visited_states: Dict[str, int] = {}
        self.skill_competence: Dict[str, float] = {}
        self.body_health_history: List[float] = []
        
        logger.info(
            f"Initialized IntrinsicMotivationEngine with weights: "
            f"curiosity={curiosity_weight}, exploration={exploration_weight}, "
            f"competence={competence_weight}, homeostasis={homeostasis_weight}"
        )
    
    def generate_goal(
        self,
        current_state: Dict,
        prediction_error: float,
        body_health: float,
    ) -> Dict:
        """Generate a new intrinsic goal based on current state.
        
        Args:
            current_state: Current agent state
            prediction_error: Recent prediction error magnitude
            body_health: Current body integrity (0-1)
            
        Returns:
            Dictionary describing the new goal
        """
        # Update state tracking
        self.recent_prediction_errors.append(prediction_error)
        if len(self.recent_prediction_errors) > 100:
            self.recent_prediction_errors.pop(0)
        
        self.body_health_history.append(body_health)
        if len(self.body_health_history) > 100:
            self.body_health_history.pop(0)
        
        # Calculate urgency for each goal type
        urgencies = self._calculate_goal_urgencies(
            current_state,
            prediction_error,
            body_health
        )
        
        # Select goal type based on weighted urgencies
        goal_type = self._select_goal_type(urgencies)
        
        # Generate specific goal for the selected type
        goal = self._generate_specific_goal(goal_type, current_state)
        
        # Record goal
        self.current_goal = goal
        self.goal_history.append(goal)
        
        logger.info(f"Generated new goal: {goal['type']} - {goal['description']}")
        
        return goal
    
    def _calculate_goal_urgencies(
        self,
        current_state: Dict,
        prediction_error: float,
        body_health: float,
    ) -> Dict[GoalType, float]:
        """Calculate urgency for each goal type.
        
        Args:
            current_state: Current state
            prediction_error: Recent prediction error
            body_health: Body health level
            
        Returns:
            Dictionary mapping goal types to urgency scores
        """
        urgencies = {}
        
        # Curiosity urgency (high when prediction errors are high)
        avg_pred_error = np.mean(self.recent_prediction_errors) if self.recent_prediction_errors else 0.5
        urgencies[GoalType.REDUCE_PREDICTION_ERROR] = (
            self.goal_priorities[GoalType.REDUCE_PREDICTION_ERROR] * avg_pred_error
        )
        
        # Exploration urgency (high when visiting known states)
        state_key = self._state_to_key(current_state)
        visit_count = self.visited_states.get(state_key, 0)
        self.visited_states[state_key] = visit_count + 1
        novelty = 1.0 / (1.0 + visit_count)
        urgencies[GoalType.MAXIMIZE_NOVEL_SENSATIONS] = (
            self.goal_priorities[GoalType.MAXIMIZE_NOVEL_SENSATIONS] * (1.0 - novelty)
        )
        
        # Competence urgency (high when skills are low)
        avg_competence = np.mean(list(self.skill_competence.values())) if self.skill_competence else 0.3
        urgencies[GoalType.MASTER_MOTOR_SKILL] = (
            self.goal_priorities[GoalType.MASTER_MOTOR_SKILL] * (1.0 - avg_competence)
        )
        
        # Homeostasis urgency (high when body health is low)
        urgencies[GoalType.MAINTAIN_BODY_INTEGRITY] = (
            self.goal_priorities[GoalType.MAINTAIN_BODY_INTEGRITY] * (1.0 - body_health)
        )
        
        return urgencies
    
    def _select_goal_type(self, urgencies: Dict[GoalType, float]) -> GoalType:
        """Select goal type based on urgencies.
        
        Args:
            urgencies: Urgency scores for each goal type
            
        Returns:
            Selected goal type
        """
        # Add some randomness for exploration
        urgency_values = np.array(list(urgencies.values()))
        urgency_values += np.random.rand(len(urgency_values)) * 0.1
        
        goal_types = list(urgencies.keys())
        selected_idx = np.argmax(urgency_values)
        
        return goal_types[selected_idx]
    
    def _generate_specific_goal(
        self,
        goal_type: GoalType,
        current_state: Dict,
    ) -> Dict:
        """Generate a specific goal for the given type.
        
        Args:
            goal_type: Type of goal to generate
            current_state: Current state
            
        Returns:
            Specific goal dictionary
        """
        goal = {
            'type': goal_type.value,
            'timestamp': current_state.get('timestamp', 0),
            'priority': self.goal_priorities[goal_type],
        }
        
        if goal_type == GoalType.REDUCE_PREDICTION_ERROR:
            goal['description'] = "Explore sensorimotor contingencies to reduce prediction errors"
            goal['target_error_reduction'] = 0.3
            
        elif goal_type == GoalType.MAXIMIZE_NOVEL_SENSATIONS:
            goal['description'] = "Seek novel sensory experiences"
            goal['target_novelty_score'] = 0.7
            
        elif goal_type == GoalType.MASTER_MOTOR_SKILL:
            # Pick a random motor skill to practice
            skills = ['reach', 'grasp', 'rotate', 'push']
            skill = np.random.choice(skills)
            goal['description'] = f"Practice and master {skill} motor skill"
            goal['target_skill'] = skill
            goal['target_competence'] = 0.8
            
        elif goal_type == GoalType.MAINTAIN_BODY_INTEGRITY:
            goal['description'] = "Maintain body integrity and avoid damage"
            goal['target_health'] = 0.95
        
        return goal
    
    def goal_achieved(self, goal: Dict, current_metrics: Dict) -> bool:
        """Check if a goal has been achieved.
        
        Args:
            goal: Goal to check
            current_metrics: Current performance metrics
            
        Returns:
            True if goal is achieved
        """
        goal_type = GoalType(goal['type'])
        
        if goal_type == GoalType.REDUCE_PREDICTION_ERROR:
            avg_error = np.mean(self.recent_prediction_errors[-20:]) if len(self.recent_prediction_errors) >= 20 else 1.0
            return avg_error < goal.get('target_error_reduction', 0.3)
        
        elif goal_type == GoalType.MAXIMIZE_NOVEL_SENSATIONS:
            novelty = current_metrics.get('novelty_score', 0.0)
            return novelty > goal.get('target_novelty_score', 0.7)
        
        elif goal_type == GoalType.MASTER_MOTOR_SKILL:
            skill = goal.get('target_skill', '')
            competence = self.skill_competence.get(skill, 0.0)
            return competence > goal.get('target_competence', 0.8)
        
        elif goal_type == GoalType.MAINTAIN_BODY_INTEGRITY:
            health = current_metrics.get('body_health', 0.0)
            return health > goal.get('target_health', 0.95)
        
        return False
    
    def _state_to_key(self, state: Dict) -> str:
        """Convert state to hashable key.
        
        Args:
            state: State dictionary
            
        Returns:
            String key
        """
        position = state.get('position', np.zeros(3))
        discretized = np.round(np.array(position) * 10).astype(int)
        return str(discretized.tolist())


class PredictiveWorldModel:
    """Internal model for simulating action outcomes.
    
    Implements "what-if" simulation using learned dynamics model.
    The agent can mentally simulate actions before executing them.
    
    Attributes:
        transition_model: Learned state transition dynamics
        accuracy_history: History of prediction accuracy
    """
    
    def __init__(self, state_dim: int = 10, action_dim: int = 6):
        """Initialize predictive world model.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Simple linear transition model (can be made more sophisticated)
        self.transition_matrix = np.random.randn(state_dim, state_dim + action_dim) * 0.1
        self.transition_bias = np.zeros(state_dim)
        
        self.accuracy_history: List[float] = []
        
        logger.info(
            f"Initialized PredictiveWorldModel "
            f"(state_dim={state_dim}, action_dim={action_dim})"
        )
    
    def simulate(
        self,
        initial_state: np.ndarray,
        action_sequence: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Simulate a sequence of actions from initial state.
        
        Args:
            initial_state: Starting state
            action_sequence: Sequence of actions to simulate
            
        Returns:
            List of predicted states
        """
        predicted_states = [initial_state]
        current_state = initial_state.copy()
        
        for action in action_sequence:
            # Predict next state
            state_action = np.concatenate([current_state, action])
            next_state = self.transition_matrix @ state_action + self.transition_bias
            
            # Add some noise for uncertainty
            next_state += np.random.randn(self.state_dim) * 0.05
            
            predicted_states.append(next_state)
            current_state = next_state
        
        return predicted_states
    
    def update_from_experience(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        learning_rate: float = 0.01,
    ) -> float:
        """Update world model from real experience.
        
        Args:
            state: Previous state
            action: Action taken
            next_state: Resulting state
            learning_rate: Learning rate for update
            
        Returns:
            Prediction error magnitude
        """
        # Predict what we thought would happen
        state_action = np.concatenate([state, action])
        predicted_next_state = self.transition_matrix @ state_action + self.transition_bias
        
        # Calculate prediction error
        prediction_error = next_state - predicted_next_state
        error_magnitude = np.linalg.norm(prediction_error)
        
        # Update transition model using gradient descent
        self.transition_matrix += learning_rate * np.outer(prediction_error, state_action)
        self.transition_bias += learning_rate * prediction_error
        
        # Track accuracy
        self.accuracy_history.append(error_magnitude)
        if len(self.accuracy_history) > 1000:
            self.accuracy_history.pop(0)
        
        return error_magnitude
    
    def get_accuracy(self) -> float:
        """Get current model accuracy.
        
        Returns:
            Average prediction accuracy (lower error = higher accuracy)
        """
        if not self.accuracy_history:
            return 0.0
        
        recent_errors = self.accuracy_history[-100:]
        avg_error = np.mean(recent_errors)
        
        # Convert error to accuracy score (0-1)
        accuracy = 1.0 / (1.0 + avg_error)
        return accuracy


class MetaLearningController:
    """Controller for adapting learning strategy.
    
    Implements "learning to learn" by monitoring success/failure
    patterns and switching between strategies.
    
    Attributes:
        current_strategy: Currently active learning strategy
        strategy_history: History of strategy switches
        strategy_performance: Performance metrics for each strategy
    """
    
    def __init__(self):
        """Initialize meta-learning controller."""
        self.current_strategy = LearningStrategy.EXPLORE
        self.strategy_history: List[Tuple[int, LearningStrategy]] = []
        self.strategy_performance: Dict[LearningStrategy, List[float]] = {
            strategy: [] for strategy in LearningStrategy
        }
        
        self.steps_since_strategy_change = 0
        self.min_steps_before_change = 20  # Minimum steps before considering strategy change
        
        logger.info(
            f"Initialized MetaLearningController with strategy: {self.current_strategy.value}"
        )
    
    def adapt_strategy(
        self,
        learning_history: List[Dict],
        current_performance: Dict,
    ) -> LearningStrategy:
        """Adapt learning strategy based on performance.
        
        Args:
            learning_history: Recent learning outcomes
            current_performance: Current performance metrics
            
        Returns:
            Selected learning strategy
        """
        self.steps_since_strategy_change += 1
        
        # Don't change strategy too frequently
        if self.steps_since_strategy_change < self.min_steps_before_change:
            return self.current_strategy
        
        # Analyze recent performance
        if len(learning_history) >= 10:
            recent_rewards = [h.get('reward', 0.0) for h in learning_history[-10:]]
            avg_reward = np.mean(recent_rewards)
            reward_trend = np.mean(recent_rewards[-5:]) - np.mean(recent_rewards[:5])
            
            success_rate = sum(1 for h in learning_history[-10:] if h.get('success', False)) / 10.0
        else:
            avg_reward = 0.0
            reward_trend = 0.0
            success_rate = 0.0
        
        # Record performance for current strategy
        self.strategy_performance[self.current_strategy].append(avg_reward)
        
        new_strategy = self.current_strategy
        
        # Decision rules for strategy switching
        if self.current_strategy == LearningStrategy.EXPLORE:
            # Switch to exploit if we're making progress
            if reward_trend > 0 and success_rate > 0.3:
                new_strategy = LearningStrategy.EXPLOIT
                logger.info("Switching to EXPLOIT: Sufficient exploration, time to leverage knowledge")
        
        elif self.current_strategy == LearningStrategy.EXPLOIT:
            # Switch to explore if performance plateaus or drops
            if reward_trend < -0.1 or success_rate < 0.2:
                new_strategy = LearningStrategy.EXPLORE
                logger.info("Switching to EXPLORE: Performance declined, need more exploration")
            # Switch to consolidate if doing well
            elif success_rate > 0.7:
                new_strategy = LearningStrategy.CONSOLIDATE
                logger.info("Switching to CONSOLIDATE: High performance, time to refine")
        
        elif self.current_strategy == LearningStrategy.CONSOLIDATE:
            # Switch to curious if we've mastered current skills
            if success_rate > 0.9:
                new_strategy = LearningStrategy.CURIOUS
                logger.info("Switching to CURIOUS: Mastered current skills, seek new challenges")
            # Switch back to exploit if performance drops
            elif success_rate < 0.6:
                new_strategy = LearningStrategy.EXPLOIT
                logger.info("Switching to EXPLOIT: Performance unstable, return to practice")
        
        elif self.current_strategy == LearningStrategy.CURIOUS:
            # Switch to explore if curiosity reveals interesting areas
            if avg_reward > 0.5:
                new_strategy = LearningStrategy.EXPLORE
                logger.info("Switching to EXPLORE: Found interesting area, explore systematically")
        
        # Check if we should try imitation learning (after repeated failures)
        if success_rate < 0.1 and len(learning_history) > 50:
            consecutive_failures = sum(
                1 for h in learning_history[-20:]
                if not h.get('success', False)
            )
            if consecutive_failures > 15:
                new_strategy = LearningStrategy.IMITATE
                logger.info("Switching to IMITATE: Repeated failures, would benefit from demonstration")
        
        # Update strategy if changed
        if new_strategy != self.current_strategy:
            self.strategy_history.append((len(learning_history), new_strategy))
            self.current_strategy = new_strategy
            self.steps_since_strategy_change = 0
        
        return self.current_strategy


class AutonomousLearningAgent:
    """Autonomous learning agent with intrinsic motivation.
    
    Implements the complete autonomous learning cycle:
    1. Goal generation based on intrinsic motivation
    2. Planning with internal world model
    3. Action execution through embodiment
    4. Learning from prediction errors
    5. Self-model adaptation
    6. Meta-learning strategy adaptation
    
    This closes the loop from reactive to truly autonomous intelligence.
    
    Attributes:
        embodiment: Virtual body for interaction
        brain: Neural substrate
        self_stream: Self-perception and monitoring
        learner: Sensorimotor learning system
        motivation_engine: Intrinsic goal generator
        world_model: Predictive model of environment
        meta_controller: Learning strategy controller
    """
    
    def __init__(
        self,
        embodiment: VirtualBody,
        brain: BrainModel,
        self_stream: SelfPerceptionStream,
        learner: SensorimotorReinforcementLearner,
        state_dim: int = 10,
        action_dim: int = 6,
    ):
        """Initialize autonomous learning agent.
        
        Args:
            embodiment: Virtual body
            brain: Brain model
            self_stream: Self-perception stream
            learner: Sensorimotor learner
            state_dim: State space dimension
            action_dim: Action space dimension
        """
        self.embodiment = embodiment
        self.brain = brain
        self.self_stream = self_stream
        self.learner = learner
        
        # Initialize autonomous learning components
        self.motivation_engine = IntrinsicMotivationEngine()
        self.world_model = PredictiveWorldModel(state_dim, action_dim)
        self.meta_controller = MetaLearningController()
        
        # Current state
        self.current_goal: Optional[Dict] = None
        self.current_strategy = LearningStrategy.EXPLORE
        
        # Learning history
        self.cycle_count = 0
        self.learning_history: List[Dict] = []
        
        logger.info(
            "Initialized AutonomousLearningAgent with intrinsic motivation, "
            "world model, and meta-learning"
        )
    
    def run_autonomous_cycle(
        self,
        environment_context: Dict,
    ) -> Dict:
        """Execute one autonomous learning cycle.
        
        This is the main loop that integrates all components:
        Goal → Plan → Act → Learn → Adapt
        
        Args:
            environment_context: Current environment state
            
        Returns:
            Dictionary with cycle results and metrics
        """
        self.cycle_count += 1
        
        # Get current state
        current_state = self._get_current_state(environment_context)
        body_health = environment_context.get('body_health', 1.0)
        
        # 1. GOAL SETTING
        if self.current_goal is None or self._goal_achieved():
            prediction_error = self._get_recent_prediction_error()
            self.current_goal = self.motivation_engine.generate_goal(
                current_state,
                prediction_error,
                body_health
            )
        
        # 2. PLANNING with world model
        action_candidates = self._generate_action_candidates()
        simulated_outcomes = []
        
        for action_seq in action_candidates:
            predicted_states = self.world_model.simulate(
                current_state.get('state_vector', np.zeros(10)),
                action_seq
            )
            simulated_outcomes.append({
                'actions': action_seq,
                'predicted_states': predicted_states,
                'goal_alignment': self._evaluate_goal_alignment(
                    predicted_states[-1],
                    self.current_goal
                )
            })
        
        # Select best action sequence
        best_outcome = max(simulated_outcomes, key=lambda x: x['goal_alignment'])
        selected_action = best_outcome['actions'][0] if best_outcome['actions'] else np.zeros(6)
        
        # 3. EXECUTION
        motor_output = self._decode_action(selected_action)
        feedback = self.embodiment.execute_motor_command(motor_output)
        
        # 4. LEARNING from prediction error
        next_state = self._get_current_state(feedback)
        prediction_error = self.world_model.update_from_experience(
            current_state.get('state_vector', np.zeros(10)),
            selected_action,
            next_state.get('state_vector', np.zeros(10))
        )
        
        # Update learner
        learning_result = self.learner.learn_from_interaction(
            action=motor_output,
            resulting_feedback=feedback,
            external_reward=0.0  # Purely intrinsic
        )
        
        # 5. SELF-MODEL ADAPTATION
        if prediction_error > 0.5:  # Threshold for model adaptation
            self._adjust_self_body_model(feedback)
        
        # 6. META-LEARNING strategy adaptation
        self.current_strategy = self.meta_controller.adapt_strategy(
            self.learning_history,
            {
                'reward': learning_result.get('total_reward', 0.0),
                'success': prediction_error < 0.3,
                'body_health': body_health,
            }
        )
        
        # Record cycle
        cycle_result = {
            'cycle': self.cycle_count,
            'goal': self.current_goal,
            'strategy': self.current_strategy.value,
            'prediction_error': prediction_error,
            'learning_result': learning_result,
            'body_health': body_health,
            'world_model_accuracy': self.world_model.get_accuracy(),
        }
        
        self.learning_history.append(cycle_result)
        
        return cycle_result
    
    def _get_current_state(self, context: Dict) -> Dict:
        """Extract current state from context.
        
        Args:
            context: Environment/feedback context
            
        Returns:
            State dictionary
        """
        # Extract position and other state variables
        position = context.get('position', np.zeros(3))
        velocity = context.get('velocity', np.zeros(3))
        joint_angles = context.get('joint_angles', {})
        
        # Create state vector
        state_vector = np.concatenate([
            position,
            velocity,
            list(joint_angles.values())[:4] if joint_angles else [0, 0, 0, 0]
        ])
        
        return {
            'position': position,
            'velocity': velocity,
            'joint_angles': joint_angles,
            'state_vector': state_vector,
            'timestamp': context.get('timestamp', self.cycle_count),
        }
    
    def _goal_achieved(self) -> bool:
        """Check if current goal is achieved.
        
        Returns:
            True if goal is achieved
        """
        if self.current_goal is None:
            return True
        
        # Get current metrics
        current_metrics = {
            'prediction_error': self._get_recent_prediction_error(),
            'body_health': 1.0,  # Would come from embodiment
            'novelty_score': 0.5,  # Would be calculated
        }
        
        return self.motivation_engine.goal_achieved(
            self.current_goal,
            current_metrics
        )
    
    def _get_recent_prediction_error(self) -> float:
        """Get recent average prediction error.
        
        Returns:
            Average prediction error
        """
        if not self.learning_history:
            return 0.5
        
        recent_errors = [
            h.get('prediction_error', 0.5)
            for h in self.learning_history[-10:]
        ]
        
        return np.mean(recent_errors)
    
    def _generate_action_candidates(self) -> List[List[np.ndarray]]:
        """Generate candidate action sequences.
        
        Returns:
            List of action sequences to evaluate
        """
        # Generate based on current strategy
        if self.current_strategy == LearningStrategy.EXPLORE:
            # Random exploration actions
            num_candidates = 5
            sequence_length = 3
            
            candidates = []
            for _ in range(num_candidates):
                sequence = [
                    np.random.randn(6) * 0.5
                    for _ in range(sequence_length)
                ]
                candidates.append(sequence)
            
            return candidates
        
        elif self.current_strategy == LearningStrategy.EXPLOIT:
            # Use learned policies (simplified: small variations)
            num_candidates = 3
            sequence_length = 3
            
            # Get best known action (simplified)
            base_action = np.zeros(6)
            
            candidates = []
            for _ in range(num_candidates):
                sequence = [
                    base_action + np.random.randn(6) * 0.1
                    for _ in range(sequence_length)
                ]
                candidates.append(sequence)
            
            return candidates
        
        else:
            # Default: simple exploration
            return [[np.random.randn(6) * 0.3] for _ in range(3)]
    
    def _evaluate_goal_alignment(
        self,
        predicted_state: np.ndarray,
        goal: Dict,
    ) -> float:
        """Evaluate how well a predicted state aligns with the goal.
        
        Args:
            predicted_state: Predicted future state
            goal: Current goal
            
        Returns:
            Alignment score (higher is better)
        """
        goal_type = GoalType(goal['type'])
        
        # Simplified goal alignment evaluation
        if goal_type == GoalType.REDUCE_PREDICTION_ERROR:
            # Prefer states that lead to learning
            return np.random.rand()  # Placeholder
        
        elif goal_type == GoalType.MAXIMIZE_NOVEL_SENSATIONS:
            # Prefer states we haven't visited
            state_key = str(np.round(predicted_state[:3] * 10).astype(int).tolist())
            visit_count = self.motivation_engine.visited_states.get(state_key, 0)
            return 1.0 / (1.0 + visit_count)
        
        elif goal_type == GoalType.MASTER_MOTOR_SKILL:
            # Prefer states that exercise motor skills
            return np.linalg.norm(predicted_state[:3])  # Movement magnitude
        
        elif goal_type == GoalType.MAINTAIN_BODY_INTEGRITY:
            # Prefer safe states
            return 1.0 - np.linalg.norm(predicted_state[:3]) * 0.1
        
        return 0.5
    
    def _decode_action(self, action: np.ndarray) -> Dict:
        """Decode action vector to motor command.
        
        Args:
            action: Action vector
            
        Returns:
            Motor command dictionary
        """
        # Convert action vector to motor neuron activations
        motor_command = {
            'motor_neurons': {
                i: max(0.0, min(1.0, action[i]))
                for i in range(min(len(action), 6))
            }
        }
        
        return motor_command
    
    def _adjust_self_body_model(self, feedback: Dict) -> None:
        """Adjust self-model based on unexpected feedback.
        
        Args:
            feedback: Unexpected feedback from environment
        """
        # Update self-perception stream with anomaly
        self.self_stream.update(
            sensor_data=feedback,
            motor_commands={},
            internal_state={'model_updating': True}
        )
        
        logger.info("Adjusting self-model based on prediction error")
    
    def get_statistics(self) -> Dict:
        """Get agent statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_cycles': self.cycle_count,
            'current_goal': self.current_goal,
            'current_strategy': self.current_strategy.value,
            'world_model_accuracy': self.world_model.get_accuracy(),
            'goal_history_length': len(self.motivation_engine.goal_history),
            'strategy_changes': len(self.meta_controller.strategy_history),
        }
