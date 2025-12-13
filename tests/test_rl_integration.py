"""Tests for reinforcement learning integration."""

import pytest
import numpy as np
from src.motor_output import ReinforcementLearningIntegrator


class TestReinforcementLearningIntegrator:
    """Test ReinforcementLearningIntegrator class."""

    def test_initialization_td(self):
        """Test TD learning initialization."""
        rl = ReinforcementLearningIntegrator(
            learning_rate=0.01,
            discount_factor=0.99,
            algorithm="td"
        )
        assert rl.learning_rate == 0.01
        assert rl.discount_factor == 0.99
        assert rl.algorithm == "td"
        assert len(rl.value_estimates) == 0
        assert len(rl.reward_history) == 0

    def test_td_learning_update(self):
        """Test TD learning value updates."""
        rl = ReinforcementLearningIntegrator(learning_rate=0.1, algorithm="td")
        
        # Update values
        rl.update_values("state1", reward=1.0, next_state_key="state2")
        rl.update_values("state2", reward=0.5, next_state_key="state3")
        
        # Check values are updated
        assert "state1" in rl.value_estimates
        assert "state2" in rl.value_estimates
        assert len(rl.reward_history) == 2

    def test_qlearning_initialization(self):
        """Test Q-learning initialization."""
        rl = ReinforcementLearningIntegrator(algorithm="qlearning")
        assert rl.algorithm == "qlearning"
        assert len(rl.q_table) == 0
        assert rl.epsilon == 0.1

    def test_qlearning_update(self):
        """Test Q-learning updates."""
        rl = ReinforcementLearningIntegrator(learning_rate=0.1, algorithm="qlearning")
        
        # Update Q-value
        td_error = rl.update_q_value(
            state_key="state1",
            action=0,
            reward=1.0,
            next_state_key="state2",
            done=False
        )
        
        assert isinstance(td_error, float)
        assert ("state1", 0) in rl.q_table
        assert len(rl.reward_history) == 1

    def test_qlearning_action_selection(self):
        """Test Q-learning action selection."""
        rl = ReinforcementLearningIntegrator(algorithm="qlearning")
        
        # Train Q-values
        for _ in range(10):
            rl.update_q_value("state1", 2, 1.0, "state2", False)
        
        # Test exploitation (should prefer action 2)
        rl.epsilon = 0.0
        action = rl.select_action("state1", num_actions=4, exploration=False)
        assert action == 2

    def test_qlearning_exploration(self):
        """Test Q-learning exploration."""
        rl = ReinforcementLearningIntegrator(algorithm="qlearning")
        rl.epsilon = 1.0  # Always explore
        
        actions = [rl.select_action("state1", num_actions=4, exploration=True) 
                  for _ in range(100)]
        
        # Should have variety in actions
        unique_actions = set(actions)
        assert len(unique_actions) > 1

    def test_policy_gradient_initialization(self):
        """Test policy gradient initialization."""
        rl = ReinforcementLearningIntegrator(algorithm="policy_gradient")
        assert rl.algorithm == "policy_gradient"
        assert len(rl.policy_params) == 0
        assert rl.baseline == 0.0

    def test_policy_gradient_update(self):
        """Test policy gradient updates."""
        rl = ReinforcementLearningIntegrator(
            learning_rate=0.01,
            algorithm="policy_gradient"
        )
        
        state_features = np.random.randn(10)
        
        # Update policy
        rl.update_policy_gradient(
            state_key="state1",
            action=2,
            reward=1.0,
            state_features=state_features
        )
        
        assert "state1" in rl.policy_params
        assert len(rl.reward_history) == 1

    def test_policy_gradient_baseline(self):
        """Test baseline adaptation in policy gradient."""
        rl = ReinforcementLearningIntegrator(algorithm="policy_gradient")
        
        # Multiple updates with high rewards
        for _ in range(20):
            rl.update_policy_gradient("state1", 0, 10.0)
        
        # Baseline should increase
        assert rl.baseline > 0

    def test_actor_critic_initialization(self):
        """Test actor-critic initialization."""
        rl = ReinforcementLearningIntegrator(algorithm="actor_critic")
        assert rl.algorithm == "actor_critic"
        assert len(rl.actor_params) == 0
        assert len(rl.critic_values) == 0

    def test_actor_critic_update(self):
        """Test actor-critic updates."""
        rl = ReinforcementLearningIntegrator(
            learning_rate=0.01,
            algorithm="actor_critic"
        )
        
        state_features = np.random.randn(10)
        
        # Update actor-critic
        critic_loss, actor_loss = rl.update_actor_critic(
            state_key="state1",
            action=1,
            reward=0.5,
            next_state_key="state2",
            state_features=state_features,
            done=False
        )
        
        assert isinstance(critic_loss, float)
        assert isinstance(actor_loss, float)
        assert "state1" in rl.critic_values
        assert "state1" in rl.actor_params

    def test_get_value(self):
        """Test getting value estimates."""
        rl = ReinforcementLearningIntegrator(algorithm="td")
        
        rl.update_values("state1", 1.0, "state2")
        value = rl.get_value("state1")
        
        assert isinstance(value, float)
        assert value != 0.0

    def test_get_q_value(self):
        """Test getting Q-values."""
        rl = ReinforcementLearningIntegrator(algorithm="qlearning")
        
        rl.update_q_value("state1", 2, 1.0, "state2", False)
        q_val = rl.get_q_value("state1", 2)
        
        assert isinstance(q_val, float)
        assert q_val != 0.0

    def test_average_reward(self):
        """Test average reward calculation."""
        rl = ReinforcementLearningIntegrator(algorithm="td")
        
        # Add rewards
        for reward in [1.0, 0.5, 0.8, 0.6, 1.2]:
            rl.update_values("state1", reward, "state2")
        
        avg = rl.get_average_reward(window=5)
        expected = (1.0 + 0.5 + 0.8 + 0.6 + 1.2) / 5
        
        assert abs(avg - expected) < 0.01

    def test_average_reward_window(self):
        """Test average reward with window."""
        rl = ReinforcementLearningIntegrator(algorithm="td")
        
        # Add many rewards
        for i in range(20):
            rl.update_values("state1", float(i), "state2")
        
        # Should only average last 10
        avg = rl.get_average_reward(window=10)
        expected = sum(range(10, 20)) / 10
        
        assert abs(avg - expected) < 0.01

    def test_reset(self):
        """Test reset functionality."""
        rl = ReinforcementLearningIntegrator(algorithm="qlearning")
        
        # Populate data
        rl.update_q_value("state1", 0, 1.0, "state2", False)
        rl.update_values("state1", 1.0, "state2")
        
        # Reset
        rl.reset()
        
        # Check everything is cleared
        assert len(rl.value_estimates) == 0
        assert len(rl.reward_history) == 0
        assert len(rl.q_table) == 0
        assert len(rl.policy_params) == 0
        assert len(rl.actor_params) == 0
        assert len(rl.critic_values) == 0
        assert rl.baseline == 0.0

    def test_episode_done(self):
        """Test Q-learning with episode done."""
        rl = ReinforcementLearningIntegrator(algorithm="qlearning")
        
        # Terminal state (done=True)
        td_error = rl.update_q_value(
            state_key="state1",
            action=0,
            reward=10.0,
            next_state_key="terminal",
            done=True
        )
        
        # Q-value should reflect terminal reward only
        q_val = rl.get_q_value("state1", 0)
        assert q_val > 0

    def test_multiple_actions(self):
        """Test learning with multiple actions."""
        rl = ReinforcementLearningIntegrator(learning_rate=0.1, algorithm="qlearning")
        
        # Train multiple actions differently
        for _ in range(10):
            rl.update_q_value("state1", 0, 0.1, "state2", False)
            rl.update_q_value("state1", 1, 1.0, "state2", False)
        
        # Action 1 should have higher Q-value
        q0 = rl.get_q_value("state1", 0)
        q1 = rl.get_q_value("state1", 1)
        
        assert q1 > q0

    def test_discount_factor_effect(self):
        """Test discount factor in Q-learning."""
        # High discount (values future rewards)
        rl_high = ReinforcementLearningIntegrator(
            learning_rate=0.5,
            discount_factor=0.99,
            algorithm="qlearning"
        )
        
        # Low discount (values immediate rewards)
        rl_low = ReinforcementLearningIntegrator(
            learning_rate=0.5,
            discount_factor=0.1,
            algorithm="qlearning"
        )
        
        # Same update
        rl_high.update_q_value("s1", 0, 1.0, "s2", False)
        rl_low.update_q_value("s1", 0, 1.0, "s2", False)
        
        # Both should have learned something
        assert rl_high.get_q_value("s1", 0) > 0
        assert rl_low.get_q_value("s1", 0) > 0

    def test_learning_rate_effect(self):
        """Test learning rate effect."""
        # High learning rate
        rl_fast = ReinforcementLearningIntegrator(
            learning_rate=0.9,
            algorithm="td"
        )
        
        # Low learning rate
        rl_slow = ReinforcementLearningIntegrator(
            learning_rate=0.01,
            algorithm="td"
        )
        
        # Single update
        rl_fast.update_values("state1", 1.0)
        rl_slow.update_values("state1", 1.0)
        
        # Fast should learn more from single update
        assert rl_fast.get_value("state1") > rl_slow.get_value("state1")
