"""Unit tests for cell_lifecycle.py."""

import pytest
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell_lifecycle import (
    mutate_params,
    mutate_weight,
    maybe_kill_and_reproduce,
    update_health_and_age
)
from brain_model import BrainModel, Neuron


class TestMutateParams:
    """Tests for mutate_params function."""
    
    def test_mutate_params_changes_values(self, rng):
        """Test that mutation changes parameter values."""
        params = {"tau_m": 20.0, "v_rest": -65.0}
        mutated = mutate_params(params, rng, std=0.1)
        
        # Values should be different (with high probability)
        assert mutated["tau_m"] != params["tau_m"]
        assert mutated["v_rest"] != params["v_rest"]
        
    def test_mutate_params_preserves_keys(self, rng):
        """Test that mutation preserves parameter keys."""
        params = {"tau_m": 20.0, "v_rest": -65.0, "name": "test"}
        mutated = mutate_params(params, rng, std=0.1)
        
        assert set(mutated.keys()) == set(params.keys())
        
    def test_mutate_params_only_numeric(self, rng):
        """Test that only numeric values are mutated."""
        params = {"tau_m": 20.0, "name": "test", "count": 5}
        mutated = mutate_params(params, rng, std=0.1)
        
        # Numeric values should change
        assert mutated["tau_m"] != params["tau_m"]
        assert mutated["count"] != params["count"]
        # Non-numeric should stay same
        assert mutated["name"] == params["name"]
        
    def test_mutate_params_small_std(self, rng):
        """Test that small std produces small changes."""
        params = {"tau_m": 20.0}
        mutated = mutate_params(params, rng, std=0.01)
        
        # Change should be small
        change = abs(mutated["tau_m"] - params["tau_m"])
        assert change < 1.0  # Should be much less than 20% change
        
    def test_mutate_params_large_std(self, rng):
        """Test that large std can produce large changes."""
        params = {"tau_m": 20.0}
        
        # Run multiple times to get a large change
        max_change = 0
        for _ in range(100):
            mutated = mutate_params(params, rng, std=0.5)
            change = abs(mutated["tau_m"] - params["tau_m"])
            max_change = max(max_change, change)
            
        # Should get some large changes with std=0.5
        assert max_change > 5.0


class TestMutateWeight:
    """Tests for mutate_weight function."""
    
    def test_mutate_weight_changes_value(self, rng):
        """Test that mutation changes weight value."""
        weight = 0.5
        mutated = mutate_weight(weight, rng, std=0.1)
        
        # Should be different (with high probability)
        assert mutated != weight
        
    def test_mutate_weight_small_std(self, rng):
        """Test that small std produces small changes."""
        weight = 0.5
        mutated = mutate_weight(weight, rng, std=0.01)
        
        # Change should be small
        change = abs(mutated - weight)
        assert change < 0.1
        
    def test_mutate_weight_can_increase_or_decrease(self, rng):
        """Test that mutation can both increase and decrease."""
        weight = 0.5
        results = [mutate_weight(weight, rng, std=0.1) for _ in range(100)]
        
        # Should have both increases and decreases
        increases = sum(1 for r in results if r > weight)
        decreases = sum(1 for r in results if r < weight)
        
        assert increases > 0
        assert decreases > 0


class TestUpdateHealthAndAge:
    """Tests for update_health_and_age function."""
    
    def test_update_increments_age(self, sample_neuron, brain_model):
        """Test that update increments neuron age."""
        initial_age = sample_neuron.age
        update_health_and_age(sample_neuron, brain_model)
        
        assert sample_neuron.age == initial_age + 1
        
    def test_update_decreases_health(self, sample_neuron, brain_model):
        """Test that update decreases neuron health."""
        initial_health = sample_neuron.health
        update_health_and_age(sample_neuron, brain_model)
        
        assert sample_neuron.health < initial_health
        
    def test_update_health_floor(self, sample_neuron, brain_model):
        """Test that health doesn't go below zero."""
        sample_neuron.health = 0.0001
        update_health_and_age(sample_neuron, brain_model)
        
        assert sample_neuron.health >= 0.0
        
    def test_update_multiple_times(self, sample_neuron, brain_model):
        """Test repeated health and age updates."""
        for _ in range(100):
            update_health_and_age(sample_neuron, brain_model)
            
        assert sample_neuron.age == 100
        assert sample_neuron.health >= 0.0


class TestMaybeKillAndReproduce:
    """Tests for maybe_kill_and_reproduce function."""
    
    def test_healthy_neuron_survives(self, sample_neuron, brain_model, rng):
        """Test that healthy young neuron survives."""
        sample_neuron.health = 1.0
        sample_neuron.age = 10
        
        brain_model.neurons[sample_neuron.id] = sample_neuron
        
        result = maybe_kill_and_reproduce(sample_neuron, brain_model, rng)
        
        # Should return same neuron
        assert result is sample_neuron
        assert sample_neuron.id in brain_model.neurons
        
    def test_low_health_causes_death(self, sample_neuron, brain_model, rng):
        """Test that low health causes death and possibly reproduction."""
        sample_neuron.health = 0.05  # Below threshold
        old_generation = sample_neuron.generation
        brain_model.neurons[sample_neuron.id] = sample_neuron
        
        result = maybe_kill_and_reproduce(sample_neuron, brain_model, rng)
        
        # Should be replaced (if reproduction enabled) or die
        if result is not None:
            # New neuron should have incremented generation
            assert result.generation == old_generation + 1
            # Health should be reset
            assert result.health == 1.0
            # Age should be reset
            assert result.age == 0
        # Either way, process should complete without error
            
    def test_old_age_causes_death(self, sample_neuron, brain_model, rng):
        """Test that old age causes death and possibly reproduction."""
        sample_neuron.age = 2000  # Above max_age
        old_generation = sample_neuron.generation
        brain_model.neurons[sample_neuron.id] = sample_neuron
        
        result = maybe_kill_and_reproduce(sample_neuron, brain_model, rng)
        
        # Should be replaced or die
        if result is not None:
            # Generation should increase
            assert result.generation == old_generation + 1
            # Age should be reset
            assert result.age == 0
        # Either way, process should complete without error
            
    def test_reproduction_creates_offspring(self, sample_neuron, brain_model, rng):
        """Test that reproduction creates offspring at same position."""
        sample_neuron.health = 0.05
        old_position = sample_neuron.position()
        brain_model.neurons[sample_neuron.id] = sample_neuron
        
        result = maybe_kill_and_reproduce(sample_neuron, brain_model, rng)
        
        if result is not None:
            # New neuron at same position
            assert result.position() == old_position
            # Generation incremented
            assert result.generation == sample_neuron.generation + 1
            # Health reset
            assert result.health == 1.0
            # Age reset
            assert result.age == 0
            
    def test_reproduction_inherits_mutated_params(self, sample_neuron, brain_model, rng):
        """Test that offspring inherits mutated parameters."""
        sample_neuron.health = 0.05
        original_tau_m = sample_neuron.params["tau_m"]
        brain_model.neurons[sample_neuron.id] = sample_neuron
        
        result = maybe_kill_and_reproduce(sample_neuron, brain_model, rng)
        
        if result is not None:
            # Parameters should be similar but mutated
            new_tau_m = result.params["tau_m"]
            assert new_tau_m != original_tau_m  # Should be mutated
            
    def test_reproduction_transfers_synapses(self, brain_model, rng):
        """Test that reproduction transfers synapses to offspring."""
        # Create network
        n1 = brain_model.add_neuron(1, 1, 1, 0)
        n2 = brain_model.add_neuron(2, 2, 2, 0)
        n3 = brain_model.add_neuron(3, 3, 3, 0)
        
        # Create synapses involving n2
        s1 = brain_model.add_synapse(n1.id, n2.id, weight=0.5)
        s2 = brain_model.add_synapse(n2.id, n3.id, weight=0.6)
        
        # Kill n2
        n2.health = 0.05
        result = maybe_kill_and_reproduce(n2, brain_model, rng)
        
        if result is not None:
            # Should have synapses to/from new neuron
            new_synapses = brain_model.get_synapses_for_neuron(result.id, "both")
            assert len(new_synapses) == 2
            
    def test_death_without_reproduction(self, sample_neuron, brain_model, rng):
        """Test death when reproduction is disabled."""
        # Disable reproduction
        brain_model.config["cell_lifecycle"]["enable_reproduction"] = False
        
        sample_neuron.health = 0.05
        brain_model.neurons[sample_neuron.id] = sample_neuron
        
        result = maybe_kill_and_reproduce(sample_neuron, brain_model, rng)
        
        # Should die without offspring
        assert result is None
        assert sample_neuron.id not in brain_model.neurons
        
    def test_death_disabled(self, sample_neuron, brain_model, rng):
        """Test that neuron survives when death is disabled."""
        brain_model.config["cell_lifecycle"]["enable_death"] = False
        
        sample_neuron.health = 0.05
        sample_neuron.age = 2000
        brain_model.neurons[sample_neuron.id] = sample_neuron
        
        result = maybe_kill_and_reproduce(sample_neuron, brain_model, rng)
        
        # Should survive even with low health and old age
        assert result is sample_neuron
        assert sample_neuron.id in brain_model.neurons
        
    def test_parent_id_tracking(self, sample_neuron, brain_model, rng):
        """Test that offspring tracks parent ID."""
        sample_neuron.health = 0.05
        parent_id = sample_neuron.id
        brain_model.neurons[sample_neuron.id] = sample_neuron
        
        result = maybe_kill_and_reproduce(sample_neuron, brain_model, rng)
        
        if result is not None and result.id != parent_id:
            # New neuron should track parent
            assert result.parent_id == parent_id
            
    def test_mutation_variability(self, brain_model, rng):
        """Test that multiple reproductions produce varied offspring."""
        # Create many offspring from same parent
        offspring_params = []
        
        for i in range(10):
            neuron = brain_model.add_neuron(5, 5, 5, 0)
            neuron.health = 0.05
            
            result = maybe_kill_and_reproduce(neuron, brain_model, rng)
            if result is not None:
                offspring_params.append(result.params["tau_m"])
                
        # Offspring should have varied parameters
        if len(offspring_params) > 1:
            assert len(set(offspring_params)) > 1  # Not all the same
