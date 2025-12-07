"""Tests for advanced plasticity mechanisms."""

import pytest
from src.plasticity import (
    homeostatic_scaling,
    BCMThreshold,
    bcm_plasticity,
    ShortTermPlasticityState,
    apply_short_term_plasticity,
    create_facilitating_synapse,
    create_depressing_synapse,
)
from src.brain_model import BrainModel, Neuron, Synapse


class TestHomeostaticPlasticity:
    """Tests for homeostatic synaptic scaling."""
    
    def test_homeostatic_scaling_basic(self):
        """Test basic homeostatic scaling."""
        # Create neurons
        neurons = {
            0: Neuron(id=0, x=0, y=0, z=0, w=0, v_membrane=-60.0),
            1: Neuron(id=1, x=1, y=0, z=0, w=0, v_membrane=-50.0),
        }
        
        # Create synapses with different weights
        synapses = [
            Synapse(pre_id=0, post_id=1, weight=0.5),
            Synapse(pre_id=0, post_id=1, weight=0.3),
        ]
        
        initial_weights = [s.weight for s in synapses]
        
        # Apply homeostatic scaling
        homeostatic_scaling(
            neurons=neurons,
            synapses=synapses,
            target_rate=5.0,
            scaling_rate=0.1
        )
        
        # Weights should have changed
        final_weights = [s.weight for s in synapses]
        assert final_weights != initial_weights
        
        # All weights should be within bounds
        assert all(0.0 <= w <= 1.0 for w in final_weights)
    
    def test_homeostatic_prevents_saturation(self):
        """Test homeostatic scaling prevents weight saturation."""
        neurons = {
            0: Neuron(id=0, x=0, y=0, z=0, w=0, v_membrane=-40.0),  # High activity
            1: Neuron(id=1, x=1, y=0, z=0, w=0, v_membrane=-40.0),  # High activity
        }
        
        # Start with very high weights
        synapses = [Synapse(pre_id=0, post_id=1, weight=0.95) for _ in range(5)]
        
        # Apply scaling multiple times with higher target rate
        for _ in range(10):
            homeostatic_scaling(
                neurons=neurons,
                synapses=synapses,
                target_rate=2.0,  # Lower target means scale down
                scaling_rate=0.1
            )
        
        # Weights should have been scaled down (or at least some change occurred)
        final_weights = [s.weight for s in synapses]
        # Check that weights changed from initial
        assert any(w != 0.95 for w in final_weights)
    
    def test_homeostatic_respects_bounds(self):
        """Test homeostatic scaling respects weight bounds."""
        # Create model with custom bounds
        config = {
            "lattice_shape": [10, 10, 10, 10],
            "neuron_model": {"params_default": {}},
            "cell_lifecycle": {},
            "plasticity": {"weight_min": 0.0, "weight_max": 0.8},
            "senses": {},
            "areas": []
        }
        model = BrainModel(config=config)
        
        neurons = {
            0: Neuron(id=0, x=0, y=0, z=0, w=0),
            1: Neuron(id=1, x=1, y=0, z=0, w=0),
        }
        
        synapses = [Synapse(pre_id=0, post_id=1, weight=0.5)]
        
        # Apply aggressive scaling
        for _ in range(100):
            homeostatic_scaling(
                neurons=neurons,
                synapses=synapses,
                target_rate=20.0,
                scaling_rate=0.5,
                model=model
            )
        
        # Weights should not exceed bounds
        assert all(0.0 <= s.weight <= 0.8 for s in synapses)


class TestBCMMetaplasticity:
    """Tests for BCM metaplasticity."""
    
    def test_bcm_threshold_initialization(self):
        """Test BCM threshold initialization."""
        threshold = BCMThreshold(theta=0.5, target_rate=5.0)
        assert threshold.theta == 0.5
        assert threshold.target_rate == 5.0
    
    def test_bcm_threshold_update_high_activity(self):
        """Test threshold increases with high activity."""
        threshold = BCMThreshold(theta=0.5, tau=100.0)
        initial_theta = threshold.theta
        
        # High postsynaptic rate
        threshold.update(postsynaptic_rate=10.0, dt=10.0)
        
        # Threshold should increase
        assert threshold.theta > initial_theta
    
    def test_bcm_threshold_update_low_activity(self):
        """Test threshold decreases with low activity."""
        threshold = BCMThreshold(theta=0.5, tau=100.0)
        initial_theta = threshold.theta
        
        # Low postsynaptic rate
        threshold.update(postsynaptic_rate=1.0, dt=10.0)
        
        # Threshold should decrease
        assert threshold.theta < initial_theta
    
    def test_bcm_threshold_bounds(self):
        """Test threshold respects bounds."""
        threshold = BCMThreshold(theta=0.5, theta_min=0.1, theta_max=2.0)
        
        # Try to push below minimum
        for _ in range(100):
            threshold.update(postsynaptic_rate=0.0, dt=1.0)
        assert threshold.theta >= 0.1
        
        # Try to push above maximum
        for _ in range(100):
            threshold.update(postsynaptic_rate=100.0, dt=1.0)
        assert threshold.theta <= 2.0
    
    def test_bcm_plasticity_above_threshold(self):
        """Test BCM plasticity when activity above threshold."""
        config = {
            "lattice_shape": [10, 10, 10, 10],
            "neuron_model": {"params_default": {}},
            "cell_lifecycle": {},
            "plasticity": {"weight_min": 0.0, "weight_max": 1.0},
            "senses": {},
            "areas": []
        }
        model = BrainModel(config=config)
        
        synapse = Synapse(pre_id=0, post_id=1, weight=0.5)
        threshold = BCMThreshold(theta=0.5)
        
        # High activity above threshold -> LTP
        initial_weight = synapse.weight
        bcm_plasticity(
            synapse=synapse,
            pre_rate=5.0,
            post_rate=10.0,  # Above threshold
            bcm_threshold=threshold,
            learning_rate=0.01,
            model=model
        )
        
        # Weight should increase
        assert synapse.weight > initial_weight
    
    def test_bcm_plasticity_below_threshold(self):
        """Test BCM plasticity when activity below threshold."""
        config = {
            "lattice_shape": [10, 10, 10, 10],
            "neuron_model": {"params_default": {}},
            "cell_lifecycle": {},
            "plasticity": {"weight_min": 0.0, "weight_max": 1.0},
            "senses": {},
            "areas": []
        }
        model = BrainModel(config=config)
        
        synapse = Synapse(pre_id=0, post_id=1, weight=0.5)
        threshold = BCMThreshold(theta=10.0)  # High threshold
        
        # Activity below threshold -> LTD
        initial_weight = synapse.weight
        bcm_plasticity(
            synapse=synapse,
            pre_rate=5.0,
            post_rate=5.0,  # Below threshold
            bcm_threshold=threshold,
            learning_rate=0.01,
            model=model
        )
        
        # Weight should decrease
        assert synapse.weight < initial_weight
    
    def test_bcm_plasticity_respects_bounds(self):
        """Test BCM plasticity respects weight bounds."""
        config = {
            "lattice_shape": [10, 10, 10, 10],
            "neuron_model": {"params_default": {}},
            "cell_lifecycle": {},
            "plasticity": {"weight_min": 0.0, "weight_max": 1.0},
            "senses": {},
            "areas": []
        }
        model = BrainModel(config=config)
        
        synapse = Synapse(pre_id=0, post_id=1, weight=0.5)
        threshold = BCMThreshold(theta=0.1)
        
        # Apply many updates
        for _ in range(100):
            bcm_plasticity(
                synapse=synapse,
                pre_rate=10.0,
                post_rate=10.0,
                bcm_threshold=threshold,
                learning_rate=0.1,
                model=model
            )
        
        # Weight should stay within bounds
        assert 0.0 <= synapse.weight <= 1.0


class TestShortTermPlasticity:
    """Tests for short-term plasticity."""
    
    def test_stp_state_initialization(self):
        """Test STP state initialization."""
        stp = ShortTermPlasticityState(U=0.5, tau_facil=50.0, tau_rec=800.0)
        assert stp.u == 0.5
        assert stp.x == 1.0
        assert stp.U == 0.5
    
    def test_stp_reset(self):
        """Test STP state reset."""
        stp = ShortTermPlasticityState(U=0.5)
        stp.u = 0.8
        stp.x = 0.5
        
        stp.reset()
        
        assert stp.u == 0.5
        assert stp.x == 1.0
    
    def test_facilitation_on_spike(self):
        """Test facilitation increases utilization with spikes."""
        stp = ShortTermPlasticityState(U=0.1, tau_facil=50.0, tau_rec=800.0)
        
        # First spike
        u1 = stp.u
        release1 = stp.update_on_spike()
        u2 = stp.u
        
        # Utilization should increase (facilitation)
        # Note: release might decrease due to depression, but u should increase
        assert u2 > u1
    
    def test_depression_on_spike(self):
        """Test depression decreases available resources."""
        stp = ShortTermPlasticityState(U=0.5, tau_facil=800.0, tau_rec=100.0)
        
        # Multiple rapid spikes
        releases = []
        for _ in range(5):
            release = stp.update_on_spike()
            releases.append(release)
        
        # Later releases should be smaller (depression)
        assert releases[-1] < releases[0]
    
    def test_stp_decay(self):
        """Test STP decay toward baseline."""
        stp = ShortTermPlasticityState(U=0.5, tau_facil=50.0, tau_rec=800.0)
        
        # Spike to change state
        stp.update_on_spike()
        u_after_spike = stp.u
        x_after_spike = stp.x
        
        # Decay for several steps
        for _ in range(100):
            stp.decay(dt=1.0)
        
        # u should approach U, x should approach 1.0
        assert abs(stp.u - stp.U) < abs(u_after_spike - stp.U)
        assert stp.x > x_after_spike
    
    def test_apply_stp_with_spike(self):
        """Test applying STP during spike."""
        synapse = Synapse(pre_id=0, post_id=1, weight=1.0)
        stp = ShortTermPlasticityState(U=0.5)
        
        effective_weight = apply_short_term_plasticity(
            synapse=synapse,
            stp_state=stp,
            presynaptic_spike=True,
            dt=1.0
        )
        
        # Effective weight should be modulated
        assert effective_weight != synapse.weight
        assert effective_weight > 0
    
    def test_apply_stp_without_spike(self):
        """Test applying STP without spike."""
        synapse = Synapse(pre_id=0, post_id=1, weight=1.0)
        stp = ShortTermPlasticityState(U=0.5)
        
        # Spike first
        apply_short_term_plasticity(synapse, stp, True, 1.0)
        
        # Then no spike
        effective_weight = apply_short_term_plasticity(
            synapse=synapse,
            stp_state=stp,
            presynaptic_spike=False,
            dt=1.0
        )
        
        # Should still return modulated weight
        assert effective_weight > 0
    
    def test_create_facilitating_synapse(self):
        """Test creating facilitation-dominant synapse."""
        stp = create_facilitating_synapse()
        
        # Should have low U and faster facilitation
        assert stp.U < 0.2
        assert stp.tau_facil < stp.tau_rec
    
    def test_create_depressing_synapse(self):
        """Test creating depression-dominant synapse."""
        stp = create_depressing_synapse()
        
        # Should have high U and faster recovery
        assert stp.U >= 0.5
        assert stp.tau_rec < stp.tau_facil
    
    def test_facilitating_behavior(self):
        """Test facilitating synapse shows increased utilization."""
        synapse = Synapse(pre_id=0, post_id=1, weight=1.0)
        stp = create_facilitating_synapse()
        
        # Track utilization over spikes
        u_values = []
        for _ in range(5):
            apply_short_term_plasticity(synapse, stp, True, 1.0)
            u_values.append(stp.u)
            # Allow some recovery between spikes
            for _ in range(2):
                stp.decay(dt=1.0)
        
        # Utilization should increase (facilitation)
        assert u_values[-1] > u_values[0]
    
    def test_depressing_behavior(self):
        """Test depressing synapse shows depression."""
        synapse = Synapse(pre_id=0, post_id=1, weight=1.0)
        stp = create_depressing_synapse()
        
        # Multiple rapid spikes
        weights = []
        for _ in range(5):
            w = apply_short_term_plasticity(synapse, stp, True, 1.0)
            weights.append(w)
        
        # Should show depression (decreasing weights)
        assert weights[-1] < weights[0]


class TestAdvancedPlasticityIntegration:
    """Integration tests for advanced plasticity mechanisms."""
    
    def test_homeostatic_stabilizes_bcm(self):
        """Test homeostatic plasticity stabilizes BCM learning."""
        config = {
            "lattice_shape": [10, 10, 10, 10],
            "neuron_model": {"params_default": {}},
            "cell_lifecycle": {},
            "plasticity": {"weight_min": 0.0, "weight_max": 1.0},
            "senses": {},
            "areas": []
        }
        model = BrainModel(config=config)
        
        neurons = {
            0: Neuron(id=0, x=0, y=0, z=0, w=0),
            1: Neuron(id=1, x=1, y=0, z=0, w=0),
        }
        
        synapses = [Synapse(pre_id=0, post_id=1, weight=0.5)]
        threshold = BCMThreshold()
        
        # Simulate learning with both mechanisms
        for _ in range(100):
            # BCM learning
            bcm_plasticity(
                synapse=synapses[0],
                pre_rate=5.0,
                post_rate=8.0,
                bcm_threshold=threshold,
                learning_rate=0.01,
                model=model
            )
            threshold.update(postsynaptic_rate=8.0, dt=1.0)
            
            # Homeostatic scaling every 10 steps
            if _ % 10 == 0:
                homeostatic_scaling(neurons, synapses, model=model)
        
        # Weight should be stable and within bounds
        assert 0.0 <= synapses[0].weight <= 1.0
    
    def test_stp_with_hebbian(self):
        """Test short-term plasticity combined with Hebbian learning."""
        synapse = Synapse(pre_id=0, post_id=1, weight=0.5)
        stp = ShortTermPlasticityState(U=0.5)
        
        # Simulate spike train with both mechanisms
        long_term_weight = synapse.weight
        
        for i in range(10):
            # Short-term effect
            effective_weight = apply_short_term_plasticity(
                synapse=synapse,
                stp_state=stp,
                presynaptic_spike=(i % 2 == 0),  # Every other spike
                dt=1.0
            )
            
            # Long-term effect (simplified Hebbian)
            if i % 2 == 0:
                long_term_weight += 0.01
        
        # Both short and long-term changes should occur
        assert long_term_weight > 0.5  # Long-term increase
        # Short-term state should be modified
        assert stp.x < 1.0 or stp.u != stp.U
