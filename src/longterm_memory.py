"""Long-term memory consolidation and replay mechanisms.

This module provides:
- Memory consolidation (transfer from working to long-term memory)
- Replay mechanisms for memory strengthening
- Sleep-like states for offline learning
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    try:
        from .brain_model import BrainModel
    except ImportError:
        from brain_model import BrainModel


class MemoryConsolidation:
    """Manages consolidation of memories from short-term to long-term storage.
    
    Consolidation is the process by which initially labile memories are
    transformed into stable, long-lasting forms. This typically involves:
    - Transfer from hippocampus to cortical areas
    - Strengthening of relevant synapses
    - Weakening of irrelevant connections
    
    Attributes:
        model: The brain model
        short_term_area: Brain area for short-term/working memory
        long_term_area: Brain area for consolidated long-term memory
        consolidation_threshold: Activity threshold for consolidation
        
    Example:
        >>> consolidator = MemoryConsolidation(model)
        >>> # Store a pattern in short-term memory
        >>> consolidator.store_pattern(pattern, "visual_object")
        >>> # Later, consolidate to long-term memory
        >>> consolidator.consolidate("visual_object")
    """
    
    def __init__(
        self,
        model: "BrainModel",
        short_term_area: str = "hippocampus",
        long_term_area: str = "temporal_cortex",
        consolidation_threshold: float = 0.5,
        consolidation_rate: float = 0.01,
    ):
        """Initialize memory consolidation system.
        
        Args:
            model: Brain model
            short_term_area: Area for initial memory encoding
            long_term_area: Area for long-term storage
            consolidation_threshold: Minimum activity for consolidation
            consolidation_rate: Rate of synaptic strength transfer
        """
        self.model = model
        self.short_term_area = short_term_area
        self.long_term_area = long_term_area
        self.consolidation_threshold = consolidation_threshold
        self.consolidation_rate = consolidation_rate
        
        # Track patterns stored in short-term memory
        self.short_term_patterns: Dict[str, np.ndarray] = {}
        self.pattern_neurons: Dict[str, List[int]] = {}
        self.consolidation_history: List[Dict[str, Any]] = []
    
    def store_pattern(
        self,
        pattern: np.ndarray,
        pattern_id: str,
    ) -> bool:
        """Store a pattern in short-term memory area.
        
        Args:
            pattern: Pattern to store
            pattern_id: Identifier for the pattern
            
        Returns:
            True if storage successful
        """
        # Get neurons in short-term area
        st_neurons = self._get_area_neurons(self.short_term_area)
        
        if len(st_neurons) < len(pattern):
            return False
        
        # Store pattern by activating corresponding neurons
        selected_neurons = st_neurons[:len(pattern)]
        self.pattern_neurons[pattern_id] = selected_neurons
        self.short_term_patterns[pattern_id] = pattern.copy()
        
        # Activate neurons according to pattern
        for i, neuron_id in enumerate(selected_neurons):
            if neuron_id in self.model.neurons:
                self.model.neurons[neuron_id].external_input += pattern[i]
        
        return True
    
    def consolidate(
        self,
        pattern_id: str,
        strength_multiplier: float = 1.0,
    ) -> bool:
        """Consolidate a pattern from short-term to long-term memory.
        
        Transfers the pattern representation by:
        1. Creating connections from short-term to long-term area
        2. Strengthening these connections over time
        3. Gradually reducing dependence on short-term area
        
        Args:
            pattern_id: Identifier of pattern to consolidate
            strength_multiplier: Factor to amplify consolidation strength
            
        Returns:
            True if consolidation initiated successfully
        """
        if pattern_id not in self.short_term_patterns:
            return False
        
        pattern = self.short_term_patterns[pattern_id]
        st_neurons = self.pattern_neurons.get(pattern_id, [])
        
        # Get neurons in long-term area
        lt_neurons = self._get_area_neurons(self.long_term_area)
        
        if not st_neurons or not lt_neurons:
            return False
        
        # Create connections from short-term to long-term neurons
        n_connections = min(len(st_neurons), len(lt_neurons))
        
        for i in range(n_connections):
            st_id = st_neurons[i]
            lt_id = lt_neurons[i % len(lt_neurons)]
            
            # Create or strengthen synapse
            weight = pattern[i % len(pattern)] * self.consolidation_rate * strength_multiplier
            
            try:
                self.model.add_synapse(st_id, lt_id, weight=abs(weight))
            except (ValueError, KeyError):
                # Synapse might already exist, try to strengthen it
                for synapse in self.model.synapses:
                    if synapse.pre_id == st_id and synapse.post_id == lt_id:
                        synapse.weight += abs(weight) * 0.1
                        break
        
        # Record consolidation event
        self.consolidation_history.append({
            "pattern_id": pattern_id,
            "timestamp": len(self.consolidation_history),
            "num_connections": n_connections,
            "strength": strength_multiplier,
        })
        
        return True
    
    def consolidate_all(self, min_age: int = 0) -> int:
        """Consolidate all patterns in short-term memory.
        
        Args:
            min_age: Minimum age (in storage time) for consolidation
            
        Returns:
            Number of patterns consolidated
        """
        count = 0
        for pattern_id in list(self.short_term_patterns.keys()):
            if self.consolidate(pattern_id):
                count += 1
        return count
    
    def _get_area_neurons(self, area_name: str) -> List[int]:
        """Get neuron IDs in a brain area.
        
        Args:
            area_name: Name of brain area
            
        Returns:
            List of neuron IDs in the area
        """
        areas = self.model.get_areas()
        area = next((a for a in areas if a["name"] == area_name), None)
        
        if area is None:
            return []
        
        ranges = area["coord_ranges"]
        area_neurons = []
        
        for neuron_id, neuron in self.model.neurons.items():
            in_area = (
                ranges["x"][0] <= neuron.x <= ranges["x"][1] and
                ranges["y"][0] <= neuron.y <= ranges["y"][1] and
                ranges["z"][0] <= neuron.z <= ranges["z"][1] and
                ranges["w"][0] <= neuron.w <= ranges["w"][1]
            )
            if in_area:
                area_neurons.append(neuron_id)
        
        return area_neurons
    
    def get_consolidation_stats(self) -> Dict[str, Any]:
        """Get statistics about consolidation process.
        
        Returns:
            Dictionary with consolidation statistics
        """
        return {
            "num_patterns_stored": len(self.short_term_patterns),
            "num_consolidations": len(self.consolidation_history),
            "patterns": list(self.short_term_patterns.keys()),
        }


class MemoryReplay:
    """Implements memory replay mechanisms for strengthening memories.
    
    Replay involves reactivating previously experienced patterns, typically
    during rest or sleep. This helps:
    - Consolidate memories
    - Extract statistical regularities
    - Plan and simulate future scenarios
    
    Attributes:
        model: The brain model
        replay_patterns: Stored patterns for replay
        replay_speed: Speed multiplier for replay (>1 = faster than real-time)
        
    Example:
        >>> replay = MemoryReplay(model)
        >>> # Record an experience pattern
        >>> replay.record_pattern(neural_activity, "episode_1")
        >>> # Later, replay the pattern
        >>> replay.replay_pattern("episode_1", speed=10.0)
    """
    
    def __init__(
        self,
        model: "BrainModel",
        replay_area: str = "hippocampus",
        max_patterns: int = 100,
        replay_speed: float = 10.0,
    ):
        """Initialize memory replay system.
        
        Args:
            model: Brain model
            replay_area: Brain area for replay generation
            max_patterns: Maximum number of patterns to store
            replay_speed: Speed multiplier for replay
        """
        self.model = model
        self.replay_area = replay_area
        self.max_patterns = max_patterns
        self.replay_speed = replay_speed
        
        self.replay_patterns: List[Dict[str, Any]] = []
        self.replay_count: Dict[str, int] = {}
    
    def record_pattern(
        self,
        neural_activity: np.ndarray,
        pattern_id: str,
        importance: float = 1.0,
    ) -> None:
        """Record a neural activity pattern for later replay.
        
        Args:
            neural_activity: Activity levels of neurons
            pattern_id: Identifier for the pattern
            importance: Importance weight for prioritizing replay
        """
        pattern = {
            "id": pattern_id,
            "activity": neural_activity.copy(),
            "importance": importance,
            "record_time": len(self.replay_patterns),
        }
        
        # Add pattern to storage
        self.replay_patterns.append(pattern)
        
        # Remove oldest if at capacity (FIFO with importance)
        if len(self.replay_patterns) > self.max_patterns:
            # Remove least important pattern
            self.replay_patterns.sort(key=lambda x: x["importance"])
            self.replay_patterns.pop(0)
    
    def replay_pattern(
        self,
        pattern_id: str,
        noise_level: float = 0.1,
    ) -> bool:
        """Replay a stored pattern by reactivating neurons.
        
        Args:
            pattern_id: Identifier of pattern to replay
            noise_level: Amount of noise to add (0-1)
            
        Returns:
            True if replay successful
        """
        # Find pattern
        pattern = next(
            (p for p in self.replay_patterns if p["id"] == pattern_id),
            None
        )
        
        if pattern is None:
            return False
        
        activity = pattern["activity"]
        
        # Get neurons in replay area
        area_neurons = self._get_area_neurons(self.replay_area)
        
        if not area_neurons:
            return False
        
        # Replay pattern with noise
        for i, neuron_id in enumerate(area_neurons[:len(activity)]):
            if neuron_id in self.model.neurons:
                # Add activity with noise
                activation = activity[i % len(activity)]
                noise = np.random.randn() * noise_level
                self.model.neurons[neuron_id].external_input += activation + noise
        
        # Track replay count
        self.replay_count[pattern_id] = self.replay_count.get(pattern_id, 0) + 1
        
        return True
    
    def replay_sequence(
        self,
        pattern_ids: List[str],
        inter_pattern_delay: int = 5,
    ) -> int:
        """Replay a sequence of patterns.
        
        Args:
            pattern_ids: List of pattern IDs to replay in order
            inter_pattern_delay: Steps to wait between patterns
            
        Returns:
            Number of patterns successfully replayed
        """
        count = 0
        for pattern_id in pattern_ids:
            if self.replay_pattern(pattern_id):
                count += 1
        return count
    
    def prioritized_replay(
        self,
        n_patterns: int = 10,
        temperature: float = 1.0,
    ) -> int:
        """Replay patterns with priority based on importance.
        
        Uses softmax sampling to select patterns, weighted by importance.
        
        Args:
            n_patterns: Number of patterns to replay
            temperature: Temperature for softmax (higher = more random)
            
        Returns:
            Number of patterns replayed
        """
        if not self.replay_patterns:
            return 0
        
        # Compute sampling probabilities
        importances = np.array([p["importance"] for p in self.replay_patterns])
        probs = np.exp(importances / temperature)
        probs = probs / np.sum(probs)
        
        # Sample patterns
        n_to_replay = min(n_patterns, len(self.replay_patterns))
        indices = np.random.choice(
            len(self.replay_patterns),
            size=n_to_replay,
            replace=False,
            p=probs
        )
        
        count = 0
        for idx in indices:
            pattern = self.replay_patterns[idx]
            if self.replay_pattern(pattern["id"]):
                count += 1
        
        return count
    
    def _get_area_neurons(self, area_name: str) -> List[int]:
        """Get neuron IDs in a brain area."""
        areas = self.model.get_areas()
        area = next((a for a in areas if a["name"] == area_name), None)
        
        if area is None:
            return []
        
        ranges = area["coord_ranges"]
        area_neurons = []
        
        for neuron_id, neuron in self.model.neurons.items():
            in_area = (
                ranges["x"][0] <= neuron.x <= ranges["x"][1] and
                ranges["y"][0] <= neuron.y <= ranges["y"][1] and
                ranges["z"][0] <= neuron.z <= ranges["z"][1] and
                ranges["w"][0] <= neuron.w <= ranges["w"][1]
            )
            if in_area:
                area_neurons.append(neuron_id)
        
        return area_neurons
    
    def get_replay_stats(self) -> Dict[str, Any]:
        """Get statistics about replay activity.
        
        Returns:
            Dictionary with replay statistics
        """
        return {
            "num_patterns": len(self.replay_patterns),
            "total_replays": sum(self.replay_count.values()),
            "replay_counts": self.replay_count.copy(),
            "most_replayed": max(self.replay_count.items(), key=lambda x: x[1])[0]
                             if self.replay_count else None,
        }


class SleepLikeState:
    """Implements sleep-like states for offline learning.
    
    During sleep-like states:
    - Global activity is reduced
    - Replay mechanisms are enhanced
    - Synaptic homeostasis occurs
    - Consolidation is prioritized
    
    Attributes:
        model: The brain model
        is_sleeping: Whether currently in sleep state
        sleep_depth: Depth of sleep (0=awake, 1=deep sleep)
        
    Example:
        >>> sleep = SleepLikeState(model, consolidator, replay)
        >>> # Enter sleep state
        >>> sleep.enter_sleep(depth=0.8)
        >>> # Run sleep cycles
        >>> for _ in range(100):
        ...     sleep.sleep_step()
        >>> sleep.exit_sleep()
    """
    
    def __init__(
        self,
        model: "BrainModel",
        consolidator: MemoryConsolidation,
        replay: MemoryReplay,
        baseline_activity: float = 0.5,
    ):
        """Initialize sleep-like state system.
        
        Args:
            model: Brain model
            consolidator: Memory consolidation system
            replay: Memory replay system
            baseline_activity: Activity level during wakefulness
        """
        self.model = model
        self.consolidator = consolidator
        self.replay = replay
        self.baseline_activity = baseline_activity
        
        self.is_sleeping = False
        self.sleep_depth = 0.0
        self.sleep_duration = 0
    
    def enter_sleep(self, depth: float = 0.7) -> None:
        """Enter sleep-like state.
        
        Args:
            depth: Sleep depth (0=light, 1=deep)
        """
        self.is_sleeping = True
        self.sleep_depth = np.clip(depth, 0.0, 1.0)
        self.sleep_duration = 0
        
        # Reduce global activity
        for neuron in self.model.neurons.values():
            neuron.external_input *= (1.0 - self.sleep_depth * 0.5)
    
    def sleep_step(self) -> Dict[str, Any]:
        """Execute one step of sleep processing.
        
        Returns:
            Dictionary with sleep step statistics
        """
        if not self.is_sleeping:
            return {"error": "Not in sleep state"}
        
        self.sleep_duration += 1
        
        # Perform memory replay (more frequent in deep sleep)
        replay_prob = self.sleep_depth * 0.3
        n_replays = 0
        
        if np.random.random() < replay_prob:
            n_replays = self.replay.prioritized_replay(
                n_patterns=int(5 * self.sleep_depth),
                temperature=1.0 / (self.sleep_depth + 0.1)
            )
        
        # Perform consolidation (especially in deep sleep)
        consolidation_prob = self.sleep_depth * 0.2
        n_consolidated = 0
        
        if np.random.random() < consolidation_prob:
            n_consolidated = self.consolidator.consolidate_all()
        
        # Synaptic homeostasis (scale down strong synapses)
        if self.sleep_depth > 0.5:
            self._synaptic_scaling(scale_factor=0.99)
        
        return {
            "sleep_duration": self.sleep_duration,
            "sleep_depth": self.sleep_depth,
            "replays": n_replays,
            "consolidations": n_consolidated,
        }
    
    def exit_sleep(self) -> Dict[str, Any]:
        """Exit sleep state and return to wakefulness.
        
        Returns:
            Dictionary with sleep session statistics
        """
        stats = {
            "total_sleep_duration": self.sleep_duration,
            "sleep_depth": self.sleep_depth,
        }
        
        self.is_sleeping = False
        self.sleep_depth = 0.0
        
        # Restore normal activity levels
        for neuron in self.model.neurons.values():
            neuron.external_input = self.baseline_activity
        
        return stats
    
    def _synaptic_scaling(self, scale_factor: float = 0.99) -> None:
        """Apply synaptic scaling for homeostasis.
        
        Scales down all synaptic weights slightly to prevent runaway
        strengthening and maintain network stability.
        
        Args:
            scale_factor: Multiplicative factor for scaling (<1)
        """
        for synapse in self.model.synapses:
            synapse.weight *= scale_factor
