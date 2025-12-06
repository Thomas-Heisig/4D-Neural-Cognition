"""Unit tests for plasticity module."""

import pytest
from src.brain_model import BrainModel, Synapse
from src.plasticity import hebbian_update, apply_weight_decay


@pytest.fixture
def model():
    """Create a test brain model."""
    config = {
        "lattice_shape": [10, 10, 10, 10],
        "neuron_model": {
            "params_default": {
                "v_rest": -65.0,
                "v_threshold": -50.0,
                "v_reset": -70.0,
                "tau_membrane": 20.0,
                "refractory_period": 5
            }
        },
        "cell_lifecycle": {
            "aging_rate": 0.001,
            "death_threshold": 0.1,
            "reproduction_threshold": 0.9,
            "reproduction_probability": 0.1
        },
        "plasticity": {
            "enabled": True,
            "learning_rate": 0.01,
            "weight_min": 0.0,
            "weight_max": 1.0,
            "weight_decay": 0.001
        },
        "senses": {},
        "areas": []
    }
    return BrainModel(config=config)


@pytest.fixture
def synapse():
    """Create a test synapse."""
    return Synapse(pre_id=0, post_id=1, weight=0.5, delay=1, plasticity_tag=0.0)


class TestHebbianUpdate:
    """Tests for Hebbian learning rule."""
    
    def test_both_active_strengthens(self, model, synapse):
        """Test that correlated activity strengthens the connection."""
        initial_weight = synapse.weight
        hebbian_update(synapse, pre_active=True, post_active=True, model=model)
        assert synapse.weight > initial_weight
        
    def test_pre_only_weakens(self, model, synapse):
        """Test that pre-only activity weakens the connection."""
        initial_weight = synapse.weight
        hebbian_update(synapse, pre_active=True, post_active=False, model=model)
        assert synapse.weight < initial_weight
        
    def test_neither_active_no_change(self, model, synapse):
        """Test that no activity causes no change."""
        initial_weight = synapse.weight
        hebbian_update(synapse, pre_active=False, post_active=False, model=model)
        assert synapse.weight == initial_weight
        
    def test_post_only_no_change(self, model, synapse):
        """Test that post-only activity causes no change."""
        initial_weight = synapse.weight
        hebbian_update(synapse, pre_active=False, post_active=True, model=model)
        assert synapse.weight == initial_weight
        
    def test_weight_bounds_max(self, model):
        """Test that weight doesn't exceed maximum."""
        synapse = Synapse(pre_id=0, post_id=1, weight=0.99, delay=1)
        for _ in range(10):
            hebbian_update(synapse, pre_active=True, post_active=True, model=model)
        assert synapse.weight <= model.get_plasticity_config()["weight_max"]
        
    def test_weight_bounds_min(self, model):
        """Test that weight doesn't go below minimum."""
        synapse = Synapse(pre_id=0, post_id=1, weight=0.01, delay=1)
        for _ in range(100):
            hebbian_update(synapse, pre_active=True, post_active=False, model=model)
        assert synapse.weight >= model.get_plasticity_config()["weight_min"]
        
    def test_learning_rate_effect(self, model, synapse):
        """Test that learning rate controls update magnitude."""
        initial_weight = synapse.weight
        learning_rate = model.get_plasticity_config()["learning_rate"]
        hebbian_update(synapse, pre_active=True, post_active=True, model=model)
        weight_change = synapse.weight - initial_weight
        assert abs(weight_change - learning_rate) < 1e-6
        
    def test_ltd_smaller_than_ltp(self, model):
        """Test that LTD is weaker than LTP."""
        synapse1 = Synapse(pre_id=0, post_id=1, weight=0.5, delay=1)
        synapse2 = Synapse(pre_id=0, post_id=1, weight=0.5, delay=1)
        
        hebbian_update(synapse1, pre_active=True, post_active=True, model=model)
        ltp_change = synapse1.weight - 0.5
        
        hebbian_update(synapse2, pre_active=True, post_active=False, model=model)
        ltd_change = abs(synapse2.weight - 0.5)
        
        assert ltp_change > ltd_change


class TestWeightDecay:
    """Tests for weight decay."""
    
    def test_positive_weight_decays(self, model):
        """Test that positive weights decay towards zero."""
        synapse = Synapse(pre_id=0, post_id=1, weight=0.5, delay=1)
        initial_weight = synapse.weight
        apply_weight_decay(synapse, model)
        assert 0 <= synapse.weight < initial_weight
        
    def test_negative_weight_decays(self, model):
        """Test that negative weights decay towards zero."""
        synapse = Synapse(pre_id=0, post_id=1, weight=-0.5, delay=1)
        initial_weight = synapse.weight
        apply_weight_decay(synapse, model)
        assert initial_weight < synapse.weight <= 0
        
    def test_zero_weight_stays_zero(self, model):
        """Test that zero weight remains zero."""
        synapse = Synapse(pre_id=0, post_id=1, weight=0.0, delay=1)
        apply_weight_decay(synapse, model)
        assert synapse.weight == 0.0
        
    def test_decay_rate(self, model):
        """Test that decay follows configured rate."""
        synapse = Synapse(pre_id=0, post_id=1, weight=0.5, delay=1)
        initial_weight = synapse.weight
        decay_rate = model.get_plasticity_config()["weight_decay"]
        apply_weight_decay(synapse, model)
        expected_weight = initial_weight - decay_rate
        assert abs(synapse.weight - expected_weight) < 1e-6
        
    def test_repeated_decay(self, model):
        """Test that repeated decay eventually reaches zero."""
        synapse = Synapse(pre_id=0, post_id=1, weight=0.1, delay=1)
        for _ in range(1000):
            apply_weight_decay(synapse, model)
        assert synapse.weight == 0.0


class TestPlasticityIntegration:
    """Integration tests for plasticity."""
    
    def test_alternating_updates(self, model, synapse):
        """Test alternating hebbian and decay updates."""
        initial_weight = synapse.weight
        
        # Apply several rounds of LTP followed by decay
        for _ in range(5):
            hebbian_update(synapse, pre_active=True, post_active=True, model=model)
            apply_weight_decay(synapse, model)
        
        # Should be higher than initial despite decay
        assert synapse.weight > initial_weight
        
    def test_equilibrium(self, model):
        """Test that LTP and decay can reach equilibrium."""
        synapse = Synapse(pre_id=0, post_id=1, weight=0.5, delay=1)
        
        # Apply many rounds of LTP + decay
        for _ in range(100):
            hebbian_update(synapse, pre_active=True, post_active=True, model=model)
            apply_weight_decay(synapse, model)
        
        # Weight should be stable (near max)
        assert 0.8 < synapse.weight <= 1.0
        
    def test_ltd_decay_to_zero(self, model):
        """Test that LTD plus decay drives weight to minimum."""
        synapse = Synapse(pre_id=0, post_id=1, weight=0.5, delay=1)
        
        # Apply many rounds of LTD + decay
        for _ in range(100):
            hebbian_update(synapse, pre_active=True, post_active=False, model=model)
            apply_weight_decay(synapse, model)
        
        # Weight should reach minimum
        assert synapse.weight == model.get_plasticity_config()["weight_min"]
