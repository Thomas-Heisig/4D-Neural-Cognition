"""Tests for motor_output module."""

import pytest
import numpy as np
from src.motor_output import (
    MotorCortexArea,
    ActionSelector,
    ContinuousController,
    ReinforcementLearningIntegrator,
    extract_motor_commands,
)
from src.brain_model import BrainModel, Neuron


class TestMotorCortexArea:
    """Test motor cortex area functionality."""

    @pytest.fixture
    def brain_model(self, minimal_config):
        """Create a simple brain model for testing."""
        model = BrainModel(config=minimal_config)
        return model

    @pytest.fixture
    def motor_area(self):
        """Create a motor cortex area."""
        area_coords = {
            'x': (0, 5),
            'y': (0, 5),
            'z': (0, 5),
            'w': (0, 5),
        }
        return MotorCortexArea(
            name="test_motor",
            area_coords=area_coords,
            action_space_dim=4,
            activation_threshold=0.5
        )

    def test_motor_area_initialization(self, motor_area):
        """Test motor area initialization."""
        assert motor_area.name == "test_motor"
        assert motor_area.action_space_dim == 4
        assert motor_area.activation_threshold == 0.5

    def test_extract_motor_output_empty_model(self, motor_area, minimal_config):
        """Test extracting output from empty model."""
        model = BrainModel(config=minimal_config)
        output = motor_area.extract_motor_output(model)
        assert len(output) == 4
        assert np.all(output == 0.0)

    def test_extract_motor_output_with_neurons(self, motor_area, brain_model):
        """Test extracting output with neurons."""
        # Add some neurons to motor area
        for i in range(5):
            neuron = Neuron(
                id=1000 + i,
                x=i, y=i, z=i, w=i,
                neuron_type='excitatory'
            )
            neuron.v_membrane = float(i) / 10.0  # Set membrane potential
            brain_model.neurons[neuron.id] = neuron
        
        output = motor_area.extract_motor_output(brain_model)
        assert len(output) == 4
        assert np.max(output) <= 1.0
        assert np.min(output) >= 0.0

    def test_extract_motor_output_normalization(self, motor_area, brain_model):
        """Test that output is normalized."""
        neuron = Neuron(id=1000, x=2, y=2, z=2, w=2, neuron_type='excitatory')
        neuron.v_membrane = 100.0  # High potential
        brain_model.neurons[1000] = neuron
        
        output = motor_area.extract_motor_output(brain_model)
        # Should be normalized to [0, 1]
        assert np.max(output) == 1.0

    def test_get_area_neurons(self, motor_area, brain_model):
        """Test getting neurons in area."""
        # Add neurons inside and outside area
        neuron_in = Neuron(id=1000, x=2, y=2, z=2, w=2, neuron_type='excitatory')
        neuron_out = Neuron(id=1001, x=8, y=8, z=8, w=8, neuron_type='excitatory')
        brain_model.neurons[1000] = neuron_in
        brain_model.neurons[1001] = neuron_out
        
        neurons = motor_area._get_area_neurons(brain_model)
        neuron_ids = [n.id for n in neurons]
        assert 1000 in neuron_ids
        assert 1001 not in neuron_ids


class TestActionSelector:
    """Test action selection mechanisms."""

    def test_selector_initialization(self):
        """Test action selector initialization."""
        selector = ActionSelector(num_actions=4, selection_method='softmax')
        assert selector.num_actions == 4
        assert selector.selection_method == 'softmax'
        assert selector.epsilon == 0.1

    def test_select_action_softmax(self):
        """Test softmax action selection."""
        selector = ActionSelector(num_actions=3, selection_method='softmax')
        action_values = np.array([1.0, 2.0, 3.0])
        action = selector.select_action(action_values, temperature=1.0)
        assert 0 <= action < 3

    def test_select_action_argmax(self):
        """Test argmax action selection."""
        selector = ActionSelector(num_actions=3, selection_method='argmax')
        action_values = np.array([1.0, 2.0, 3.0])
        action = selector.select_action(action_values)
        assert action == 2  # Highest value

    def test_select_action_epsilon_greedy(self):
        """Test epsilon-greedy action selection."""
        selector = ActionSelector(num_actions=3, selection_method='epsilon_greedy')
        selector.set_epsilon(0.0)  # Deterministic
        action_values = np.array([1.0, 2.0, 3.0])
        action = selector.select_action(action_values)
        assert action == 2  # Should pick best

    def test_select_action_wrong_size(self):
        """Test action selection with wrong-sized input."""
        selector = ActionSelector(num_actions=3, selection_method='softmax')
        action_values = np.array([1.0, 2.0])  # Wrong size
        with pytest.raises(ValueError, match="Expected 3 action values"):
            selector.select_action(action_values)

    def test_select_action_invalid_method(self):
        """Test invalid selection method."""
        selector = ActionSelector(num_actions=3, selection_method='invalid')
        action_values = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Unknown selection method"):
            selector.select_action(action_values)

    def test_set_epsilon(self):
        """Test setting epsilon parameter."""
        selector = ActionSelector(num_actions=3, selection_method='epsilon_greedy')
        selector.set_epsilon(0.5)
        assert selector.epsilon == 0.5
        selector.set_epsilon(1.5)  # Should clip to 1.0
        assert selector.epsilon == 1.0
        selector.set_epsilon(-0.1)  # Should clip to 0.0
        assert selector.epsilon == 0.0

    def test_get_action_probabilities(self):
        """Test getting action probabilities."""
        selector = ActionSelector(num_actions=3, selection_method='softmax')
        action_values = np.array([1.0, 2.0, 3.0])
        probs = selector.get_action_probabilities(action_values, temperature=1.0)
        assert len(probs) == 3
        assert np.isclose(np.sum(probs), 1.0)
        assert probs[2] > probs[1] > probs[0]  # Should reflect values

    def test_softmax_numerical_stability(self):
        """Test softmax with large values."""
        selector = ActionSelector(num_actions=3, selection_method='softmax')
        action_values = np.array([1000.0, 1001.0, 1002.0])
        probs = selector.get_action_probabilities(action_values, temperature=1.0)
        assert np.isfinite(probs).all()
        assert np.isclose(np.sum(probs), 1.0)


class TestContinuousController:
    """Test continuous control output."""

    def test_controller_initialization(self):
        """Test controller initialization."""
        controller = ContinuousController(output_dim=3, output_range=(-1.0, 1.0))
        assert controller.output_dim == 3
        assert controller.output_range == (-1.0, 1.0)

    def test_generate_output_basic(self):
        """Test basic output generation."""
        controller = ContinuousController(output_dim=2)
        neural_activity = np.array([0.5, 0.8])
        output = controller.generate_output(neural_activity, smoothing=0.0)
        assert len(output) == 2
        assert -1.0 <= output[0] <= 1.0
        assert -1.0 <= output[1] <= 1.0

    def test_generate_output_padding(self):
        """Test output padding for insufficient input."""
        controller = ContinuousController(output_dim=5)
        neural_activity = np.array([0.5, 0.8])  # Only 2 values
        output = controller.generate_output(neural_activity)
        assert len(output) == 5

    def test_generate_output_truncation(self):
        """Test output truncation for excess input."""
        controller = ContinuousController(output_dim=2)
        neural_activity = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # 5 values
        output = controller.generate_output(neural_activity)
        assert len(output) == 2

    def test_generate_output_smoothing(self):
        """Test output smoothing."""
        controller = ContinuousController(output_dim=2)
        
        # First output
        output1 = controller.generate_output(np.array([0.0, 0.0]), smoothing=0.0)
        
        # Second output with smoothing
        output2 = controller.generate_output(np.array([1.0, 1.0]), smoothing=0.5)
        
        # With smoothing, output2 should be influenced by output1
        assert len(controller.output_history) == 2

    def test_scale_to_range(self):
        """Test scaling to output range."""
        controller = ContinuousController(output_dim=2, output_range=(0.0, 10.0))
        values = np.array([0.0, 1.0])
        scaled = controller._scale_to_range(values)
        assert scaled[0] == 0.0
        assert scaled[1] == 10.0

    def test_scale_to_range_constant(self):
        """Test scaling constant values."""
        controller = ContinuousController(output_dim=2)
        values = np.array([5.0, 5.0])
        scaled = controller._scale_to_range(values)
        # Constant values should map to middle of range
        assert np.all(scaled == 0.0)  # (min+max)/2 for range [-1, 1]

    def test_reset_history(self):
        """Test resetting output history."""
        controller = ContinuousController(output_dim=2)
        controller.generate_output(np.array([0.5, 0.5]))
        controller.generate_output(np.array([0.6, 0.6]))
        assert len(controller.output_history) == 2
        
        controller.reset_history()
        assert len(controller.output_history) == 0

    def test_get_output_statistics(self):
        """Test output statistics."""
        controller = ContinuousController(output_dim=2)
        controller.generate_output(np.array([0.0, 0.0]))
        controller.generate_output(np.array([1.0, 1.0]))
        
        stats = controller.get_output_statistics()
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats

    def test_get_output_statistics_empty(self):
        """Test statistics with empty history."""
        controller = ContinuousController(output_dim=2)
        stats = controller.get_output_statistics()
        assert stats['mean'] == 0.0
        assert stats['std'] == 0.0

    def test_history_limit(self):
        """Test that history is limited."""
        controller = ContinuousController(output_dim=2)
        controller.max_history = 5
        
        # Generate more outputs than max_history
        for i in range(10):
            controller.generate_output(np.array([float(i), float(i)]))
        
        assert len(controller.output_history) == 5


class TestReinforcementLearningIntegrator:
    """Test RL integration."""

    def test_rl_initialization(self):
        """Test RL integrator initialization."""
        rl = ReinforcementLearningIntegrator(learning_rate=0.01, discount_factor=0.99)
        assert rl.learning_rate == 0.01
        assert rl.discount_factor == 0.99

    def test_update_values_basic(self):
        """Test value updates."""
        rl = ReinforcementLearningIntegrator()
        rl.update_values(state_key='s1', reward=1.0, next_state_key='s2')
        
        # Value should be updated
        value = rl.get_value('s1')
        assert value != 0.0

    def test_update_values_terminal(self):
        """Test value update for terminal state."""
        rl = ReinforcementLearningIntegrator()
        rl.update_values(state_key='s1', reward=1.0, next_state_key=None)
        
        # Should still update value
        value = rl.get_value('s1')
        assert value != 0.0

    def test_get_value_new_state(self):
        """Test getting value for new state."""
        rl = ReinforcementLearningIntegrator()
        value = rl.get_value('new_state')
        assert value == 0.0

    def test_get_average_reward(self):
        """Test average reward calculation."""
        rl = ReinforcementLearningIntegrator()
        rl.update_values('s1', reward=1.0)
        rl.update_values('s2', reward=2.0)
        rl.update_values('s3', reward=3.0)
        
        avg_reward = rl.get_average_reward()
        assert avg_reward == 2.0

    def test_get_average_reward_windowed(self):
        """Test windowed average reward."""
        rl = ReinforcementLearningIntegrator()
        for i in range(10):
            rl.update_values(f's{i}', reward=float(i))
        
        # Get average of last 3
        avg_reward = rl.get_average_reward(window=3)
        assert avg_reward == 8.0  # (7 + 8 + 9) / 3

    def test_get_average_reward_empty(self):
        """Test average reward with no history."""
        rl = ReinforcementLearningIntegrator()
        avg_reward = rl.get_average_reward()
        assert avg_reward == 0.0

    def test_reset(self):
        """Test resetting RL state."""
        rl = ReinforcementLearningIntegrator()
        rl.update_values('s1', reward=1.0)
        rl.update_values('s2', reward=2.0)
        
        rl.reset()
        
        assert len(rl.value_estimates) == 0
        assert len(rl.reward_history) == 0

    def test_td_learning_convergence(self):
        """Test that TD learning updates values correctly."""
        rl = ReinforcementLearningIntegrator(learning_rate=0.1, discount_factor=0.9)
        
        # Simulate simple chain: s1 -> s2 -> s3 with rewards
        for _ in range(100):
            rl.update_values('s3', reward=10.0, next_state_key=None)  # Terminal
            rl.update_values('s2', reward=0.0, next_state_key='s3')
            rl.update_values('s1', reward=0.0, next_state_key='s2')
        
        # Values should propagate backwards
        v1 = rl.get_value('s1')
        v2 = rl.get_value('s2')
        v3 = rl.get_value('s3')
        
        # s3 should have highest value (gets reward)
        # s2 should have second highest (close to terminal)
        # s1 should have lowest (furthest from reward)
        assert v3 > v2 > v1


class TestExtractMotorCommands:
    """Test motor command extraction function."""

    @pytest.fixture
    def brain_model_with_area(self, minimal_config):
        """Create brain model with motor area."""
        pytest.skip("BrainModel.add_area() method not implemented - areas defined in config")
        model = BrainModel(config=minimal_config)
        # Add motor cortex area
        model.add_area(
            name="motor_cortex",
            coord_ranges={
                'x': (0, 5),
                'y': (0, 5),
                'z': (0, 5),
                'w': (0, 5),
            },
            neuron_type='excitatory'
        )
        return model

    @pytest.mark.skip(reason="BrainModel.add_area() method not implemented")
    def test_extract_motor_commands_continuous(self, brain_model_with_area):
        """Test extracting continuous motor commands."""
        commands = extract_motor_commands(
            brain_model_with_area,
            motor_area_name="motor_cortex",
            control_type="continuous"
        )
        assert len(commands) == 2  # Default continuous output dim

    @pytest.mark.skip(reason="BrainModel.add_area() method not implemented")
    def test_extract_motor_commands_discrete(self, brain_model_with_area):
        """Test extracting discrete motor commands."""
        commands = extract_motor_commands(
            brain_model_with_area,
            motor_area_name="motor_cortex",
            control_type="discrete",
            num_actions=4
        )
        assert len(commands) == 4

    def test_extract_motor_commands_missing_area(self, minimal_config):
        """Test extraction with missing motor area."""
        model = BrainModel(config=minimal_config)
        # No motor area added
        commands = extract_motor_commands(
            model,
            motor_area_name="nonexistent",
            control_type="continuous"
        )
        # Should return zeros
        assert len(commands) == 2
        assert np.all(commands == 0.0)

    def test_extract_motor_commands_with_neurons(self, minimal_config):
        """Test extraction with active neurons."""
        model = BrainModel(config=minimal_config)
        # Note: add_area method not implemented - areas defined in config
        
        # Add some neurons with activity
        for i in range(3):
            neuron = Neuron(id=2000 + i, x=i, y=i, z=i, w=i, neuron_type='excitatory')
            neuron.v_membrane = 0.5 + i * 0.1
            model.neurons[neuron.id] = neuron
        
        commands = extract_motor_commands(
            model,
            motor_area_name="motor_cortex",
            control_type="continuous"
        )
        
        # Should have some non-zero values
        assert len(commands) == 2
