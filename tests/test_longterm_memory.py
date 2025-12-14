"""Tests for long-term memory module."""

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock

from src.longterm_memory import (
    MemoryConsolidation,
    MemoryReplay,
    SleepLikeState,
)


class MockNeuron:
    """Mock neuron for testing."""
    
    def __init__(self, neuron_id, x=0, y=0, z=0, w=0):
        self.id = neuron_id
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.v = 0.0
        self.external_input = 0.0


class MockSynapse:
    """Mock synapse for testing."""
    
    def __init__(self, pre_id, post_id, weight=0.5):
        self.pre_id = pre_id
        self.post_id = post_id
        self.weight = weight


class TestMemoryConsolidation:
    """Tests for MemoryConsolidation class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = Mock()
        self.model.neurons = {
            1: MockNeuron(1, x=0, y=0, z=0, w=0),
            2: MockNeuron(2, x=1, y=0, z=0, w=0),
            3: MockNeuron(3, x=2, y=0, z=0, w=0),
            4: MockNeuron(4, x=5, y=0, z=0, w=0),
            5: MockNeuron(5, x=6, y=0, z=0, w=0),
        }
        self.model.synapses = []
        self.model.add_synapse = Mock()
        self.model.get_areas = Mock(return_value=[
            {
                "name": "hippocampus",
                "coord_ranges": {
                    "x": [0, 2],
                    "y": [0, 2],
                    "z": [0, 2],
                    "w": [0, 2],
                }
            },
            {
                "name": "temporal_cortex",
                "coord_ranges": {
                    "x": [5, 7],
                    "y": [0, 2],
                    "z": [0, 2],
                    "w": [0, 2],
                }
            }
        ])
    
    def test_init(self):
        """Test initialization."""
        consolidator = MemoryConsolidation(self.model)
        
        assert consolidator.model == self.model
        assert consolidator.short_term_area == "hippocampus"
        assert consolidator.long_term_area == "temporal_cortex"
        assert consolidator.consolidation_threshold == 0.5
        assert consolidator.consolidation_rate == 0.01
        assert len(consolidator.short_term_patterns) == 0
        assert len(consolidator.pattern_neurons) == 0
        assert len(consolidator.consolidation_history) == 0
    
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        consolidator = MemoryConsolidation(
            self.model,
            short_term_area="custom_st",
            long_term_area="custom_lt",
            consolidation_threshold=0.7,
            consolidation_rate=0.05
        )
        
        assert consolidator.short_term_area == "custom_st"
        assert consolidator.long_term_area == "custom_lt"
        assert consolidator.consolidation_threshold == 0.7
        assert consolidator.consolidation_rate == 0.05
    
    def test_store_pattern_success(self):
        """Test successful pattern storage."""
        consolidator = MemoryConsolidation(self.model)
        pattern = np.array([1.0, 2.0, 3.0])
        
        result = consolidator.store_pattern(pattern, "test_pattern")
        
        assert result is True
        assert "test_pattern" in consolidator.short_term_patterns
        assert np.array_equal(consolidator.short_term_patterns["test_pattern"], pattern)
        assert "test_pattern" in consolidator.pattern_neurons
        assert len(consolidator.pattern_neurons["test_pattern"]) == 3
    
    def test_store_pattern_insufficient_neurons(self):
        """Test pattern storage with insufficient neurons."""
        consolidator = MemoryConsolidation(self.model)
        # Pattern larger than available neurons in short-term area
        pattern = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        
        result = consolidator.store_pattern(pattern, "large_pattern")
        
        assert result is False
        assert "large_pattern" not in consolidator.short_term_patterns
    
    def test_store_pattern_activates_neurons(self):
        """Test that storing pattern activates neurons."""
        consolidator = MemoryConsolidation(self.model)
        pattern = np.array([1.0, 2.0])
        
        # Store initial external inputs
        initial_inputs = {nid: n.external_input for nid, n in self.model.neurons.items()}
        
        consolidator.store_pattern(pattern, "test")
        
        # Check that some neurons received input
        final_inputs = {nid: n.external_input for nid, n in self.model.neurons.items()}
        assert any(final_inputs[nid] > initial_inputs[nid] for nid in [1, 2, 3])
    
    def test_consolidate_success(self):
        """Test successful consolidation."""
        consolidator = MemoryConsolidation(self.model)
        pattern = np.array([1.0, 2.0])
        
        consolidator.store_pattern(pattern, "test_pattern")
        result = consolidator.consolidate("test_pattern")
        
        assert result is True
        assert self.model.add_synapse.called
        assert len(consolidator.consolidation_history) == 1
        assert consolidator.consolidation_history[0]["pattern_id"] == "test_pattern"
    
    def test_consolidate_nonexistent_pattern(self):
        """Test consolidation of non-existent pattern."""
        consolidator = MemoryConsolidation(self.model)
        
        result = consolidator.consolidate("nonexistent")
        
        assert result is False
        assert len(consolidator.consolidation_history) == 0
    
    def test_consolidate_with_strength_multiplier(self):
        """Test consolidation with strength multiplier."""
        consolidator = MemoryConsolidation(self.model)
        pattern = np.array([1.0, 2.0])
        
        consolidator.store_pattern(pattern, "test")
        result = consolidator.consolidate("test", strength_multiplier=2.0)
        
        assert result is True
        assert consolidator.consolidation_history[0]["strength"] == 2.0
    
    def test_consolidate_existing_synapse(self):
        """Test consolidation when synapse already exists."""
        consolidator = MemoryConsolidation(self.model)
        pattern = np.array([1.0, 2.0])
        
        # Setup existing synapse
        existing_synapse = MockSynapse(1, 4, weight=0.5)
        self.model.synapses = [existing_synapse]
        self.model.add_synapse = Mock(side_effect=ValueError("Synapse exists"))
        
        consolidator.store_pattern(pattern, "test")
        result = consolidator.consolidate("test")
        
        assert result is True
        # Weight should have been increased
        assert existing_synapse.weight > 0.5
    
    def test_consolidate_all_empty(self):
        """Test consolidate_all with no patterns."""
        consolidator = MemoryConsolidation(self.model)
        
        count = consolidator.consolidate_all()
        
        assert count == 0
    
    def test_consolidate_all_multiple_patterns(self):
        """Test consolidate_all with multiple patterns."""
        consolidator = MemoryConsolidation(self.model)
        
        consolidator.store_pattern(np.array([1.0, 2.0]), "pattern1")
        consolidator.store_pattern(np.array([3.0, 4.0]), "pattern2")
        
        count = consolidator.consolidate_all()
        
        assert count == 2
        assert len(consolidator.consolidation_history) == 2
    
    def test_get_area_neurons_valid_area(self):
        """Test getting neurons in a valid area."""
        consolidator = MemoryConsolidation(self.model)
        
        neurons = consolidator._get_area_neurons("hippocampus")
        
        assert len(neurons) == 3  # Neurons 1, 2, 3 are in hippocampus
        assert 1 in neurons
        assert 2 in neurons
        assert 3 in neurons
    
    def test_get_area_neurons_invalid_area(self):
        """Test getting neurons in invalid area."""
        consolidator = MemoryConsolidation(self.model)
        
        neurons = consolidator._get_area_neurons("nonexistent_area")
        
        assert len(neurons) == 0
    
    def test_get_consolidation_stats(self):
        """Test getting consolidation statistics."""
        consolidator = MemoryConsolidation(self.model)
        
        consolidator.store_pattern(np.array([1.0, 2.0]), "pattern1")
        consolidator.store_pattern(np.array([3.0, 4.0]), "pattern2")
        consolidator.consolidate("pattern1")
        
        stats = consolidator.get_consolidation_stats()
        
        assert stats["num_patterns_stored"] == 2
        assert stats["num_consolidations"] == 1
        assert "pattern1" in stats["patterns"]
        assert "pattern2" in stats["patterns"]


class TestMemoryReplay:
    """Tests for MemoryReplay class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = Mock()
        self.model.neurons = {
            1: MockNeuron(1, x=0, y=0, z=0, w=0),
            2: MockNeuron(2, x=1, y=0, z=0, w=0),
            3: MockNeuron(3, x=2, y=0, z=0, w=0),
        }
        self.model.get_areas = Mock(return_value=[
            {
                "name": "hippocampus",
                "coord_ranges": {
                    "x": [0, 2],
                    "y": [0, 2],
                    "z": [0, 2],
                    "w": [0, 2],
                }
            }
        ])
    
    def test_init(self):
        """Test initialization."""
        replay = MemoryReplay(self.model)
        
        assert replay.model == self.model
        assert replay.replay_area == "hippocampus"
        assert replay.max_patterns == 100
        assert replay.replay_speed == 10.0
        assert len(replay.replay_patterns) == 0
        assert len(replay.replay_count) == 0
    
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        replay = MemoryReplay(
            self.model,
            replay_area="custom_area",
            max_patterns=50,
            replay_speed=5.0
        )
        
        assert replay.replay_area == "custom_area"
        assert replay.max_patterns == 50
        assert replay.replay_speed == 5.0
    
    def test_record_pattern(self):
        """Test recording a pattern."""
        replay = MemoryReplay(self.model)
        activity = np.array([1.0, 2.0, 3.0])
        
        replay.record_pattern(activity, "episode_1", importance=1.5)
        
        assert len(replay.replay_patterns) == 1
        assert replay.replay_patterns[0]["id"] == "episode_1"
        assert np.array_equal(replay.replay_patterns[0]["activity"], activity)
        assert replay.replay_patterns[0]["importance"] == 1.5
    
    def test_record_pattern_max_capacity(self):
        """Test recording patterns at max capacity."""
        replay = MemoryReplay(self.model, max_patterns=3)
        
        # Record patterns up to capacity
        for i in range(3):
            replay.record_pattern(np.array([i]), f"pattern_{i}", importance=i)
        
        assert len(replay.replay_patterns) == 3
        
        # Record one more - should evict least important
        replay.record_pattern(np.array([10.0]), "pattern_new", importance=5.0)
        
        assert len(replay.replay_patterns) == 3
        # Pattern with importance 0 should be removed
        pattern_ids = [p["id"] for p in replay.replay_patterns]
        assert "pattern_0" not in pattern_ids
        assert "pattern_new" in pattern_ids
    
    def test_replay_pattern_success(self):
        """Test successful pattern replay."""
        replay = MemoryReplay(self.model)
        activity = np.array([1.0, 2.0])
        
        replay.record_pattern(activity, "test_pattern")
        
        # Store initial inputs
        initial_inputs = {nid: n.external_input for nid, n in self.model.neurons.items()}
        
        result = replay.replay_pattern("test_pattern", noise_level=0.0)
        
        assert result is True
        assert replay.replay_count["test_pattern"] == 1
        
        # Check that neurons received input
        final_inputs = {nid: n.external_input for nid, n in self.model.neurons.items()}
        assert any(final_inputs[nid] > initial_inputs[nid] for nid in [1, 2])
    
    def test_replay_pattern_nonexistent(self):
        """Test replay of non-existent pattern."""
        replay = MemoryReplay(self.model)
        
        result = replay.replay_pattern("nonexistent")
        
        assert result is False
        assert "nonexistent" not in replay.replay_count
    
    def test_replay_pattern_with_noise(self):
        """Test pattern replay with noise."""
        replay = MemoryReplay(self.model)
        activity = np.array([1.0, 2.0])
        
        replay.record_pattern(activity, "test")
        
        # Set seed for reproducibility
        np.random.seed(42)
        result = replay.replay_pattern("test", noise_level=0.1)
        
        assert result is True
        # With noise, the exact values will vary
    
    def test_replay_pattern_increment_count(self):
        """Test that replay count increments correctly."""
        replay = MemoryReplay(self.model)
        replay.record_pattern(np.array([1.0]), "test")
        
        replay.replay_pattern("test")
        assert replay.replay_count["test"] == 1
        
        replay.replay_pattern("test")
        assert replay.replay_count["test"] == 2
    
    def test_replay_sequence_all_valid(self):
        """Test replaying a sequence of valid patterns."""
        replay = MemoryReplay(self.model)
        
        replay.record_pattern(np.array([1.0]), "pattern1")
        replay.record_pattern(np.array([2.0]), "pattern2")
        replay.record_pattern(np.array([3.0]), "pattern3")
        
        count = replay.replay_sequence(["pattern1", "pattern2", "pattern3"])
        
        assert count == 3
        assert replay.replay_count["pattern1"] == 1
        assert replay.replay_count["pattern2"] == 1
        assert replay.replay_count["pattern3"] == 1
    
    def test_replay_sequence_some_invalid(self):
        """Test replaying sequence with some invalid patterns."""
        replay = MemoryReplay(self.model)
        
        replay.record_pattern(np.array([1.0]), "pattern1")
        replay.record_pattern(np.array([2.0]), "pattern2")
        
        count = replay.replay_sequence(["pattern1", "nonexistent", "pattern2"])
        
        assert count == 2
        assert replay.replay_count["pattern1"] == 1
        assert replay.replay_count["pattern2"] == 1
        assert "nonexistent" not in replay.replay_count
    
    def test_prioritized_replay_empty(self):
        """Test prioritized replay with no patterns."""
        replay = MemoryReplay(self.model)
        
        count = replay.prioritized_replay(n_patterns=5)
        
        assert count == 0
    
    def test_prioritized_replay_success(self):
        """Test successful prioritized replay."""
        replay = MemoryReplay(self.model)
        
        # Record patterns with different importances
        replay.record_pattern(np.array([1.0]), "low", importance=0.5)
        replay.record_pattern(np.array([2.0]), "high", importance=2.0)
        replay.record_pattern(np.array([3.0]), "medium", importance=1.0)
        
        np.random.seed(42)
        count = replay.prioritized_replay(n_patterns=2)
        
        assert count == 2
        # At least some patterns should have been replayed
        assert sum(replay.replay_count.values()) == 2
    
    def test_prioritized_replay_temperature(self):
        """Test prioritized replay with different temperatures."""
        replay = MemoryReplay(self.model)
        
        replay.record_pattern(np.array([1.0]), "pattern1", importance=1.0)
        replay.record_pattern(np.array([2.0]), "pattern2", importance=1.0)
        
        np.random.seed(42)
        count = replay.prioritized_replay(n_patterns=1, temperature=0.5)
        
        assert count == 1
    
    def test_get_replay_stats_empty(self):
        """Test getting replay stats with no replays."""
        replay = MemoryReplay(self.model)
        
        stats = replay.get_replay_stats()
        
        assert stats["num_patterns"] == 0
        assert stats["total_replays"] == 0
        assert stats["most_replayed"] is None
    
    def test_get_replay_stats(self):
        """Test getting replay statistics."""
        replay = MemoryReplay(self.model)
        
        replay.record_pattern(np.array([1.0]), "pattern1")
        replay.record_pattern(np.array([2.0]), "pattern2")
        
        replay.replay_pattern("pattern1")
        replay.replay_pattern("pattern1")
        replay.replay_pattern("pattern2")
        
        stats = replay.get_replay_stats()
        
        assert stats["num_patterns"] == 2
        assert stats["total_replays"] == 3
        assert stats["most_replayed"] == "pattern1"
        assert stats["replay_counts"]["pattern1"] == 2
        assert stats["replay_counts"]["pattern2"] == 1


class TestSleepLikeState:
    """Tests for SleepLikeState class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = Mock()
        self.model.neurons = {
            1: MockNeuron(1, x=0, y=0, z=0, w=0),
            2: MockNeuron(2, x=1, y=0, z=0, w=0),
            3: MockNeuron(3, x=2, y=0, z=0, w=0),
        }
        self.model.synapses = [
            MockSynapse(1, 2, weight=1.0),
            MockSynapse(2, 3, weight=0.5),
        ]
        self.model.get_areas = Mock(return_value=[
            {
                "name": "hippocampus",
                "coord_ranges": {
                    "x": [0, 2],
                    "y": [0, 2],
                    "z": [0, 2],
                    "w": [0, 2],
                }
            }
        ])
        
        self.consolidator = MemoryConsolidation(self.model)
        self.replay = MemoryReplay(self.model)
    
    def test_init(self):
        """Test initialization."""
        sleep = SleepLikeState(self.model, self.consolidator, self.replay)
        
        assert sleep.model == self.model
        assert sleep.consolidator == self.consolidator
        assert sleep.replay == self.replay
        assert sleep.baseline_activity == 0.5
        assert sleep.is_sleeping is False
        assert sleep.sleep_depth == 0.0
        assert sleep.sleep_duration == 0
    
    def test_init_custom_baseline(self):
        """Test initialization with custom baseline activity."""
        sleep = SleepLikeState(
            self.model,
            self.consolidator,
            self.replay,
            baseline_activity=0.7
        )
        
        assert sleep.baseline_activity == 0.7
    
    def test_enter_sleep(self):
        """Test entering sleep state."""
        sleep = SleepLikeState(self.model, self.consolidator, self.replay)
        
        # Set initial neuron inputs
        for neuron in self.model.neurons.values():
            neuron.external_input = 1.0
        
        sleep.enter_sleep(depth=0.8)
        
        assert sleep.is_sleeping is True
        assert sleep.sleep_depth == 0.8
        assert sleep.sleep_duration == 0
        
        # Check that activity was reduced
        for neuron in self.model.neurons.values():
            assert neuron.external_input < 1.0
    
    def test_enter_sleep_depth_clipping(self):
        """Test that sleep depth is clipped to valid range."""
        sleep = SleepLikeState(self.model, self.consolidator, self.replay)
        
        sleep.enter_sleep(depth=1.5)
        assert sleep.sleep_depth == 1.0
        
        sleep.enter_sleep(depth=-0.5)
        assert sleep.sleep_depth == 0.0
    
    def test_sleep_step_not_sleeping(self):
        """Test sleep step when not in sleep state."""
        sleep = SleepLikeState(self.model, self.consolidator, self.replay)
        
        result = sleep.sleep_step()
        
        assert "error" in result
        assert result["error"] == "Not in sleep state"
    
    def test_sleep_step_increments_duration(self):
        """Test that sleep step increments duration."""
        sleep = SleepLikeState(self.model, self.consolidator, self.replay)
        sleep.enter_sleep(depth=0.5)
        
        assert sleep.sleep_duration == 0
        
        sleep.sleep_step()
        assert sleep.sleep_duration == 1
        
        sleep.sleep_step()
        assert sleep.sleep_duration == 2
    
    def test_sleep_step_with_replay(self):
        """Test sleep step with replay."""
        sleep = SleepLikeState(self.model, self.consolidator, self.replay)
        
        # Add some patterns to replay
        self.replay.record_pattern(np.array([1.0, 2.0]), "pattern1")
        
        sleep.enter_sleep(depth=0.8)
        
        np.random.seed(42)
        result = sleep.sleep_step()
        
        assert "sleep_duration" in result
        assert "sleep_depth" in result
        assert "replays" in result
        assert "consolidations" in result
        assert result["sleep_depth"] == 0.8
    
    def test_sleep_step_with_consolidation(self):
        """Test sleep step with consolidation."""
        sleep = SleepLikeState(self.model, self.consolidator, self.replay)
        
        # Add pattern to consolidate
        self.consolidator.store_pattern(np.array([1.0, 2.0]), "pattern1")
        
        sleep.enter_sleep(depth=0.9)
        
        np.random.seed(42)
        result = sleep.sleep_step()
        
        assert "consolidations" in result
        assert result["consolidations"] >= 0
    
    def test_sleep_step_synaptic_scaling(self):
        """Test synaptic scaling during deep sleep."""
        sleep = SleepLikeState(self.model, self.consolidator, self.replay)
        
        initial_weights = [s.weight for s in self.model.synapses]
        
        sleep.enter_sleep(depth=0.9)  # Deep sleep triggers scaling
        sleep.sleep_step()
        
        final_weights = [s.weight for s in self.model.synapses]
        
        # Weights should be slightly reduced (scaled down)
        for initial, final in zip(initial_weights, final_weights):
            assert final <= initial
    
    def test_sleep_step_no_scaling_light_sleep(self):
        """Test no synaptic scaling during light sleep."""
        sleep = SleepLikeState(self.model, self.consolidator, self.replay)
        
        initial_weights = [s.weight for s in self.model.synapses]
        
        sleep.enter_sleep(depth=0.3)  # Light sleep, no scaling
        sleep.sleep_step()
        
        final_weights = [s.weight for s in self.model.synapses]
        
        # Weights should remain the same (no scaling in light sleep)
        for initial, final in zip(initial_weights, final_weights):
            assert final == initial
    
    def test_exit_sleep(self):
        """Test exiting sleep state."""
        sleep = SleepLikeState(self.model, self.consolidator, self.replay)
        
        sleep.enter_sleep(depth=0.7)
        sleep.sleep_step()
        sleep.sleep_step()
        
        stats = sleep.exit_sleep()
        
        assert sleep.is_sleeping is False
        assert sleep.sleep_depth == 0.0
        assert stats["total_sleep_duration"] == 2
        assert stats["sleep_depth"] == 0.7
    
    def test_exit_sleep_restores_activity(self):
        """Test that exiting sleep restores activity levels."""
        sleep = SleepLikeState(self.model, self.consolidator, self.replay, baseline_activity=0.5)
        
        sleep.enter_sleep(depth=0.8)
        sleep.exit_sleep()
        
        # All neurons should have baseline activity restored
        for neuron in self.model.neurons.values():
            assert neuron.external_input == 0.5
    
    def test_multiple_sleep_cycles(self):
        """Test multiple sleep cycles."""
        sleep = SleepLikeState(self.model, self.consolidator, self.replay)
        
        # First cycle
        sleep.enter_sleep(depth=0.5)
        sleep.sleep_step()
        stats1 = sleep.exit_sleep()
        
        assert stats1["total_sleep_duration"] == 1
        
        # Second cycle
        sleep.enter_sleep(depth=0.7)
        sleep.sleep_step()
        sleep.sleep_step()
        stats2 = sleep.exit_sleep()
        
        assert stats2["total_sleep_duration"] == 2
        assert stats2["sleep_depth"] == 0.7
    
    def test_synaptic_scaling_factor(self):
        """Test synaptic scaling with specific factor."""
        sleep = SleepLikeState(self.model, self.consolidator, self.replay)
        
        # Set initial weights
        self.model.synapses[0].weight = 1.0
        self.model.synapses[1].weight = 2.0
        
        sleep._synaptic_scaling(scale_factor=0.5)
        
        assert self.model.synapses[0].weight == 0.5
        assert self.model.synapses[1].weight == 1.0
