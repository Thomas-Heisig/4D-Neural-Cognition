"""Tests for neuromodulation systems."""

import pytest
from src.neuromodulation import (
    DopamineSystem,
    SerotoninSystem,
    NorepinephrineSystem,
    NeuromodulationSystem,
    create_neuromodulation_system,
    NeuromodulatorState,
)
from src.brain_model import BrainModel, Neuron


class TestNeuromodulatorState:
    """Tests for basic neuromodulator state."""
    
    def test_initialization(self):
        """Test state initialization."""
        state = NeuromodulatorState(level=0.7, baseline=0.5, decay_rate=0.1)
        assert state.level == 0.7
        assert state.baseline == 0.5
        assert state.decay_rate == 0.1
    
    def test_update_baseline(self):
        """Test baseline update."""
        state = NeuromodulatorState(baseline=0.5)
        state.update_baseline(0.7)
        assert state.baseline == 0.7
        
        # Test bounds
        state.update_baseline(1.5)
        assert state.baseline == 1.0
        state.update_baseline(-0.5)
        assert state.baseline == 0.0
    
    def test_decay_toward_baseline_above(self):
        """Test decay when above baseline."""
        state = NeuromodulatorState(level=0.8, baseline=0.5, decay_rate=0.1)
        state.decay_toward_baseline()
        assert state.level == pytest.approx(0.7, abs=1e-10)  # 0.8 - 0.1
        
        # Multiple decays
        for _ in range(3):
            state.decay_toward_baseline()
        assert state.level == pytest.approx(0.5, abs=0.01)
    
    def test_decay_toward_baseline_below(self):
        """Test decay when below baseline."""
        state = NeuromodulatorState(level=0.2, baseline=0.5, decay_rate=0.1)
        state.decay_toward_baseline()
        assert state.level == pytest.approx(0.3, abs=1e-10)  # 0.2 + 0.1
        
        # Multiple decays
        for _ in range(3):
            state.decay_toward_baseline()
        assert state.level == pytest.approx(0.5, abs=0.01)


class TestDopamineSystem:
    """Tests for dopamine system."""
    
    def test_initialization(self):
        """Test dopamine system initialization."""
        da = DopamineSystem()
        assert da.state.level == 0.5
        assert da.state.baseline == 0.5
        assert len(da.reward_history) == 0
    
    def test_positive_prediction_error(self):
        """Test response to positive reward prediction error."""
        da = DopamineSystem()
        initial_level = da.state.level
        
        # Unexpected reward
        rpe = da.update(reward=1.0, expected_reward=0.0)
        
        assert rpe == 1.0  # Positive prediction error
        assert da.state.level > initial_level
        assert len(da.reward_history) == 1
    
    def test_negative_prediction_error(self):
        """Test response to negative reward prediction error."""
        da = DopamineSystem()
        initial_level = da.state.level
        
        # Expected reward not received
        rpe = da.update(reward=0.0, expected_reward=1.0)
        
        assert rpe == -1.0  # Negative prediction error
        assert da.state.level < initial_level
    
    def test_learning_rate_modulation(self):
        """Test learning rate modulation."""
        da = DopamineSystem()
        
        # Low dopamine -> low learning
        da.state.level = 0.1
        multiplier_low = da.get_learning_rate_multiplier()
        assert multiplier_low < 0.5
        
        # High dopamine -> high learning
        da.state.level = 0.9
        multiplier_high = da.get_learning_rate_multiplier()
        assert multiplier_high > 1.5
        assert multiplier_high > multiplier_low
    
    def test_plasticity_modulation(self):
        """Test plasticity modulation."""
        da = DopamineSystem()
        
        # High dopamine amplifies learning
        da.state.level = 0.9
        delta_w = 0.01
        modulated = da.modulate_plasticity(delta_w)
        assert modulated > delta_w
        
        # Low dopamine suppresses learning
        da.state.level = 0.1
        modulated_low = da.modulate_plasticity(delta_w)
        assert modulated_low < delta_w
    
    def test_reward_history_limit(self):
        """Test reward history is limited."""
        da = DopamineSystem(max_history=10)
        
        # Add more rewards than max
        for i in range(20):
            da.update(reward=float(i), expected_reward=0.0)
        
        assert len(da.reward_history) == 10


class TestSerotoninSystem:
    """Tests for serotonin system."""
    
    def test_initialization(self):
        """Test serotonin initialization."""
        serotonin = SerotoninSystem()
        assert serotonin.state.level == 0.5
        assert serotonin.inhibition_strength == 0.5
    
    def test_punishment_response(self):
        """Test response to punishment."""
        serotonin = SerotoninSystem()
        initial_level = serotonin.state.level
        
        serotonin.update(punishment=0.5, stress=0.0)
        
        # Punishment lowers serotonin
        assert serotonin.state.level < initial_level
        assert len(serotonin.punishment_history) == 1
    
    def test_stress_response(self):
        """Test response to stress."""
        serotonin = SerotoninSystem()
        initial_level = serotonin.state.level
        
        serotonin.update(punishment=0.0, stress=0.5)
        
        # Stress lowers serotonin
        assert serotonin.state.level < initial_level
    
    def test_inhibition_factor(self):
        """Test inhibition factor calculation."""
        serotonin = SerotoninSystem()
        
        # High serotonin -> high inhibition
        serotonin.state.level = 0.9
        inhibition = serotonin.get_inhibition_factor()
        assert inhibition > 0.4
        
        # Low serotonin -> low inhibition
        serotonin.state.level = 0.1
        inhibition_low = serotonin.get_inhibition_factor()
        assert inhibition_low < 0.1
    
    def test_threshold_modulation(self):
        """Test threshold modulation."""
        serotonin = SerotoninSystem()
        base_threshold = 10.0
        
        # High serotonin -> higher threshold (harder to fire)
        serotonin.state.level = 0.9
        modulated_high = serotonin.modulate_threshold(base_threshold)
        assert modulated_high > base_threshold
        
        # Low serotonin -> lower threshold (easier to fire)
        serotonin.state.level = 0.1
        modulated_low = serotonin.modulate_threshold(base_threshold)
        assert modulated_low < modulated_high


class TestNorepinephrineSystem:
    """Tests for norepinephrine system."""
    
    def test_initialization(self):
        """Test norepinephrine initialization."""
        ne = NorepinephrineSystem()
        assert ne.state.level == 0.5
        assert ne.gain_modulation == 2.0
    
    def test_uncertainty_response(self):
        """Test response to uncertainty."""
        ne = NorepinephrineSystem()
        initial_level = ne.state.level
        
        ne.update(uncertainty=0.5, novelty=0.0)
        
        # Uncertainty increases norepinephrine
        assert ne.state.level > initial_level
        assert len(ne.uncertainty_history) == 1
    
    def test_novelty_response(self):
        """Test response to novelty."""
        ne = NorepinephrineSystem()
        initial_level = ne.state.level
        
        ne.update(uncertainty=0.0, novelty=0.5)
        
        # Novelty increases norepinephrine
        assert ne.state.level > initial_level
    
    def test_gain_multiplier(self):
        """Test gain multiplier calculation."""
        ne = NorepinephrineSystem()
        
        # High NE -> high gain
        ne.state.level = 0.9
        gain_high = ne.get_gain_multiplier()
        assert gain_high > 2.5
        
        # Low NE -> low gain
        ne.state.level = 0.1
        gain_low = ne.get_gain_multiplier()
        assert gain_low < 1.5
        assert gain_low < gain_high
    
    def test_input_modulation(self):
        """Test input modulation."""
        ne = NorepinephrineSystem()
        base_input = 10.0
        
        # High NE amplifies input
        ne.state.level = 0.9
        modulated_high = ne.modulate_input(base_input)
        assert modulated_high > base_input
        
        # Low NE reduces amplification
        ne.state.level = 0.1
        modulated_low = ne.modulate_input(base_input)
        assert modulated_low < modulated_high


class TestNeuromodulationSystem:
    """Tests for complete neuromodulation system."""
    
    def test_initialization(self):
        """Test system initialization."""
        system = NeuromodulationSystem()
        assert system.dopamine is not None
        assert system.serotonin is not None
        assert system.norepinephrine is not None
    
    def test_step_updates_all(self):
        """Test step updates all systems."""
        system = NeuromodulationSystem()
        
        # Set levels away from baseline
        system.dopamine.state.level = 0.9
        system.serotonin.state.level = 0.2
        system.norepinephrine.state.level = 0.8
        
        # Step should decay toward baseline
        initial_da = system.dopamine.state.level
        initial_5ht = system.serotonin.state.level
        initial_ne = system.norepinephrine.state.level
        
        system.step()
        
        # All should move toward baseline (0.5)
        assert system.dopamine.state.level < initial_da
        assert system.serotonin.state.level > initial_5ht
        assert system.norepinephrine.state.level < initial_ne
    
    def test_get_state(self):
        """Test getting system state."""
        system = NeuromodulationSystem()
        state = system.get_state()
        
        assert "dopamine" in state
        assert "serotonin" in state
        assert "norepinephrine" in state
        assert all(0.0 <= v <= 1.0 for v in state.values())
    
    def test_modulate_learning(self):
        """Test learning modulation."""
        system = NeuromodulationSystem()
        base_delta_w = 0.01
        
        # High dopamine amplifies
        system.dopamine.state.level = 0.9
        modulated = system.modulate_learning(base_delta_w)
        assert modulated > base_delta_w
    
    def test_modulate_neuron_update(self):
        """Test neuron update modulation."""
        system = NeuromodulationSystem()
        
        # Create dummy neuron
        neuron = Neuron(id=0, x=0, y=0, z=0, w=0)
        synaptic_input = 5.0
        threshold = 10.0
        
        # High NE increases gain, high serotonin increases threshold
        system.norepinephrine.state.level = 0.9
        system.serotonin.state.level = 0.9
        
        mod_input, mod_threshold = system.modulate_neuron_update(
            neuron, synaptic_input, threshold
        )
        
        assert mod_input > synaptic_input  # NE increases input
        assert mod_threshold > threshold  # Serotonin increases threshold
    
    def test_process_reward(self):
        """Test reward processing."""
        system = NeuromodulationSystem()
        
        rpe = system.process_reward(reward=1.0, expected_reward=0.0)
        
        assert rpe == 1.0
        assert system.dopamine.state.level > 0.5
    
    def test_process_punishment(self):
        """Test punishment processing."""
        system = NeuromodulationSystem()
        
        system.process_punishment(punishment=0.5, stress=0.0)
        
        assert system.serotonin.state.level < 0.5
    
    def test_process_uncertainty(self):
        """Test uncertainty processing."""
        system = NeuromodulationSystem()
        
        system.process_uncertainty(uncertainty=0.5, novelty=0.0)
        
        assert system.norepinephrine.state.level > 0.5


class TestNeuromodulationIntegration:
    """Integration tests for neuromodulation with brain model."""
    
    def test_create_from_config(self):
        """Test creating system from configuration."""
        config = {
            "dopamine": {
                "baseline": 0.6,
                "decay_rate": 0.05,
                "learning_rate_modulation": 3.0
            },
            "serotonin": {
                "baseline": 0.4,
                "decay_rate": 0.15,
                "inhibition_strength": 0.7
            },
            "norepinephrine": {
                "baseline": 0.5,
                "decay_rate": 0.1,
                "gain_modulation": 1.5
            }
        }
        
        system = create_neuromodulation_system(config)
        
        assert system.dopamine.state.baseline == 0.6
        assert system.dopamine.learning_rate_modulation == 3.0
        assert system.serotonin.inhibition_strength == 0.7
        assert system.norepinephrine.gain_modulation == 1.5
    
    def test_create_default(self):
        """Test creating with default config."""
        system = create_neuromodulation_system()
        
        assert system.dopamine.state.baseline == 0.5
        assert system.serotonin.state.baseline == 0.5
        assert system.norepinephrine.state.baseline == 0.5
    
    def test_realistic_scenario(self):
        """Test realistic learning scenario."""
        system = NeuromodulationSystem()
        
        # Scenario: Unexpected reward during learning
        base_learning = 0.01
        
        # 1. Receive unexpected reward
        system.process_reward(reward=1.0, expected_reward=0.0)
        
        # Dopamine should be high
        assert system.dopamine.state.level > 0.5
        
        # Learning should be enhanced
        enhanced_learning = system.modulate_learning(base_learning)
        assert enhanced_learning > base_learning
        
        # 2. Several steps later, dopamine decays
        for _ in range(10):
            system.step()
        
        # Dopamine should have decayed
        assert system.dopamine.state.level < 1.0
        
        # 3. Punishment occurs
        system.process_punishment(punishment=0.5)
        
        # Serotonin should be low
        assert system.serotonin.state.level < 0.5
