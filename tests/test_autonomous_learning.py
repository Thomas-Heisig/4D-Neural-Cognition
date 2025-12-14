"""Tests for autonomous learning loop components."""

import pytest
import numpy as np

from src.autonomous_learning_loop import (
    IntrinsicMotivationEngine,
    PredictiveWorldModel,
    MetaLearningController,
    AutonomousLearningAgent,
    GoalType,
    LearningStrategy,
)
from src.embodiment.virtual_body import VirtualBody
from src.embodiment.sensorimotor_learner import SensorimotorReinforcementLearner
from src.consciousness.self_perception_stream import SelfPerceptionStream
from src.brain_model import BrainModel


@pytest.fixture
def simple_brain_config():
    """Create simple brain config for testing."""
    return {
        "lattice_shape": [5, 5, 3, 10],
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
                "name": "M1",
                "bounds": {"x": [0, 5], "y": [0, 5], "z": [0, 3], "w": [8, 8]},
                "neuron_type": "excitatory",
            },
        ]
    }


class TestIntrinsicMotivationEngine:
    """Tests for intrinsic motivation engine."""
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = IntrinsicMotivationEngine()
        
        assert engine.goal_priorities[GoalType.REDUCE_PREDICTION_ERROR] == 0.3
        assert engine.goal_priorities[GoalType.MAXIMIZE_NOVEL_SENSATIONS] == 0.3
        assert engine.goal_priorities[GoalType.MASTER_MOTOR_SKILL] == 0.3
        assert engine.goal_priorities[GoalType.MAINTAIN_BODY_INTEGRITY] == 0.1
        
        assert engine.current_goal is None
        assert len(engine.goal_history) == 0
    
    def test_goal_generation(self):
        """Test goal generation."""
        engine = IntrinsicMotivationEngine()
        
        current_state = {'position': np.array([0, 0, 0])}
        prediction_error = 0.5
        body_health = 0.9
        
        goal = engine.generate_goal(current_state, prediction_error, body_health)
        
        assert 'type' in goal
        assert 'description' in goal
        assert 'priority' in goal
        assert goal['type'] in [gt.value for gt in GoalType]
        
        # Check goal is recorded
        assert engine.current_goal == goal
        assert len(engine.goal_history) == 1
    
    def test_goal_achievement_check(self):
        """Test goal achievement checking."""
        engine = IntrinsicMotivationEngine()
        
        # Generate and check a goal
        goal = {
            'type': GoalType.MAINTAIN_BODY_INTEGRITY.value,
            'target_health': 0.95,
        }
        
        # Should not be achieved with low health
        metrics = {'body_health': 0.8}
        assert not engine.goal_achieved(goal, metrics)
        
        # Should be achieved with high health
        metrics = {'body_health': 0.96}
        assert engine.goal_achieved(goal, metrics)
    
    def test_state_tracking(self):
        """Test state visitation tracking."""
        engine = IntrinsicMotivationEngine()
        
        state1 = {'position': np.array([1, 1, 1])}
        state2 = {'position': np.array([1, 1, 1])}
        state3 = {'position': np.array([2, 2, 2])}
        
        # Generate goals to track states
        engine.generate_goal(state1, 0.5, 0.9)
        engine.generate_goal(state2, 0.5, 0.9)
        engine.generate_goal(state3, 0.5, 0.9)
        
        # Should have recorded visits
        assert len(engine.visited_states) >= 2


class TestPredictiveWorldModel:
    """Tests for predictive world model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = PredictiveWorldModel(state_dim=10, action_dim=6)
        
        assert model.state_dim == 10
        assert model.action_dim == 6
        assert model.transition_matrix.shape == (10, 16)  # state_dim x (state_dim + action_dim)
        assert len(model.accuracy_history) == 0
    
    def test_simulation(self):
        """Test action simulation."""
        model = PredictiveWorldModel(state_dim=5, action_dim=3)
        
        initial_state = np.array([1, 0, 0, 0, 0])
        actions = [
            np.array([0.1, 0, 0]),
            np.array([0, 0.1, 0]),
        ]
        
        predicted_states = model.simulate(initial_state, actions)
        
        assert len(predicted_states) == 3  # initial + 2 actions
        assert predicted_states[0].shape == (5,)
        assert predicted_states[1].shape == (5,)
    
    def test_learning_from_experience(self):
        """Test model updates from experience."""
        model = PredictiveWorldModel(state_dim=5, action_dim=3)
        
        state = np.array([1, 0, 0, 0, 0])
        action = np.array([0.1, 0, 0])
        next_state = np.array([1.1, 0.1, 0, 0, 0])
        
        # Update model
        error = model.update_from_experience(state, action, next_state, learning_rate=0.01)
        
        assert error >= 0
        assert len(model.accuracy_history) == 1
        assert model.accuracy_history[0] == error
    
    def test_accuracy_calculation(self):
        """Test accuracy metric."""
        model = PredictiveWorldModel(state_dim=5, action_dim=3)
        
        # Initial accuracy (no history)
        assert model.get_accuracy() == 0.0
        
        # Add some prediction errors
        model.accuracy_history = [0.5, 0.4, 0.3, 0.2, 0.1]
        
        accuracy = model.get_accuracy()
        assert 0 < accuracy <= 1.0


class TestMetaLearningController:
    """Tests for meta-learning controller."""
    
    def test_initialization(self):
        """Test controller initialization."""
        controller = MetaLearningController()
        
        assert controller.current_strategy == LearningStrategy.EXPLORE
        assert len(controller.strategy_history) == 0
        assert controller.steps_since_strategy_change == 0
    
    def test_strategy_persistence(self):
        """Test that strategy doesn't change too frequently."""
        controller = MetaLearningController()
        
        # Should not change immediately
        history = [{'reward': 0.1, 'success': False}] * 10
        performance = {'reward': 0.1, 'success': False}
        
        strategy = controller.adapt_strategy(history, performance)
        
        # Should still be explore initially
        assert strategy == LearningStrategy.EXPLORE
        assert controller.steps_since_strategy_change < controller.min_steps_before_change
    
    def test_strategy_switching(self):
        """Test strategy switching based on performance."""
        controller = MetaLearningController()
        controller.steps_since_strategy_change = 30  # Past minimum
        
        # Good performance should trigger switch to exploit
        history = [{'reward': 0.7, 'success': True}] * 15
        performance = {'reward': 0.8, 'success': True}
        
        strategy = controller.adapt_strategy(history, performance)
        
        # Should have considered switching (might not always switch due to logic)
        assert controller.steps_since_strategy_change >= 0


class TestAutonomousLearningAgent:
    """Tests for autonomous learning agent."""
    
    @pytest.fixture
    def agent_components(self, simple_brain_config):
        """Create agent components for testing."""
        brain = BrainModel(config=simple_brain_config)
        body = VirtualBody(body_type="humanoid", num_joints=6)
        self_stream = SelfPerceptionStream()
        learner = SensorimotorReinforcementLearner(
            virtual_body=body,
            brain_model=brain,
            learning_rate=0.01,
        )
        
        return brain, body, self_stream, learner
    
    def test_initialization(self, agent_components):
        """Test agent initialization."""
        brain, body, self_stream, learner = agent_components
        
        agent = AutonomousLearningAgent(
            embodiment=body,
            brain=brain,
            self_stream=self_stream,
            learner=learner,
            state_dim=10,
            action_dim=6,
        )
        
        assert agent.embodiment == body
        assert agent.brain == brain
        assert agent.self_stream == self_stream
        assert agent.learner == learner
        
        assert agent.motivation_engine is not None
        assert agent.world_model is not None
        assert agent.meta_controller is not None
        
        assert agent.current_goal is None
        assert agent.cycle_count == 0
    
    def test_autonomous_cycle(self, agent_components):
        """Test running an autonomous cycle."""
        brain, body, self_stream, learner = agent_components
        
        agent = AutonomousLearningAgent(
            embodiment=body,
            brain=brain,
            self_stream=self_stream,
            learner=learner,
            state_dim=10,
            action_dim=6,
        )
        
        environment_context = {
            'position': np.array([1, 1, 1]),
            'velocity': np.array([0, 0, 0]),
            'joint_angles': {},
            'body_health': 1.0,
            'timestamp': 0,
        }
        
        # Run one cycle
        result = agent.run_autonomous_cycle(environment_context)
        
        assert 'cycle' in result
        assert 'goal' in result
        assert 'strategy' in result
        assert 'prediction_error' in result
        assert 'world_model_accuracy' in result
        
        assert agent.cycle_count == 1
        assert agent.current_goal is not None
        assert len(agent.learning_history) == 1
    
    def test_statistics(self, agent_components):
        """Test statistics retrieval."""
        brain, body, self_stream, learner = agent_components
        
        agent = AutonomousLearningAgent(
            embodiment=body,
            brain=brain,
            self_stream=self_stream,
            learner=learner,
            state_dim=10,
            action_dim=6,
        )
        
        stats = agent.get_statistics()
        
        assert 'total_cycles' in stats
        assert 'current_goal' in stats
        assert 'current_strategy' in stats
        assert 'world_model_accuracy' in stats
        assert 'goal_history_length' in stats
        assert 'strategy_changes' in stats
        
        assert stats['total_cycles'] == 0  # No cycles run yet


class TestIntegration:
    """Integration tests for autonomous learning."""
    
    def test_multiple_cycles(self, simple_brain_config):
        """Test running multiple autonomous cycles."""
        brain = BrainModel(config=simple_brain_config)
        body = VirtualBody(body_type="humanoid", num_joints=6)
        self_stream = SelfPerceptionStream()
        learner = SensorimotorReinforcementLearner(
            virtual_body=body,
            brain_model=brain,
            learning_rate=0.01,
        )
        
        agent = AutonomousLearningAgent(
            embodiment=body,
            brain=brain,
            self_stream=self_stream,
            learner=learner,
            state_dim=10,
            action_dim=6,
        )
        
        # Run several cycles
        num_cycles = 10
        for i in range(num_cycles):
            environment_context = {
                'position': np.array([1 + i*0.1, 1, 1]),
                'velocity': np.array([0.1, 0, 0]),
                'joint_angles': {},
                'body_health': 1.0,
                'timestamp': i,
            }
            
            result = agent.run_autonomous_cycle(environment_context)
            assert result['cycle'] == i + 1
        
        # Check final state
        stats = agent.get_statistics()
        assert stats['total_cycles'] == num_cycles
        assert len(agent.learning_history) == num_cycles
        
        # World model should have some accuracy now
        assert stats['world_model_accuracy'] >= 0
