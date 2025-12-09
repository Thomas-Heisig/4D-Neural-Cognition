"""Working memory module with persistent activity and attractor networks.

This module provides:
- Persistent activity patterns
- Attractor networks
- Memory gating mechanisms
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    try:
        from .brain_model import BrainModel
    except ImportError:
        from brain_model import BrainModel


class PersistentActivityManager:
    """Manages persistent activity patterns in working memory."""

    def __init__(
        self,
        model: "BrainModel",
        memory_area: str = "prefrontal_cortex",
        maintenance_current: float = 0.5,
        decay_rate: float = 0.01,
    ):
        """Initialize persistent activity manager.

        Args:
            model: Brain model.
            memory_area: Name of brain area for working memory.
            maintenance_current: Current to maintain activity.
            decay_rate: Rate of activity decay without maintenance.
        """
        self.model = model
        self.memory_area = memory_area
        self.maintenance_current = maintenance_current
        self.decay_rate = decay_rate
        self.active_patterns: Dict[str, np.ndarray] = {}
        self.pattern_neurons: Dict[str, List[int]] = {}

    def encode_pattern(self, pattern_id: str, input_vector: np.ndarray) -> None:
        """Encode a pattern into persistent activity.

        Args:
            pattern_id: Identifier for the pattern.
            input_vector: Input pattern to encode.
        """
        # Get neurons in memory area
        memory_neurons = self._get_memory_neurons()

        if not memory_neurons:
            return

        # Store pattern
        self.active_patterns[pattern_id] = input_vector.copy()
        self.pattern_neurons[pattern_id] = memory_neurons[:len(input_vector)]

        # Apply input to neurons
        for i, neuron_id in enumerate(self.pattern_neurons[pattern_id]):
            if i < len(input_vector):
                self.model.neurons[neuron_id].external_input += input_vector[i]

    def maintain_activity(self, pattern_id: str) -> None:
        """Maintain persistent activity for a pattern.

        Args:
            pattern_id: Identifier for the pattern to maintain.
        """
        if pattern_id not in self.pattern_neurons:
            return

        neurons = self.pattern_neurons[pattern_id]
        pattern = self.active_patterns[pattern_id]

        # Apply maintenance current
        for i, neuron_id in enumerate(neurons):
            if neuron_id in self.model.neurons and i < len(pattern):
                # Add current proportional to pattern value
                current = self.maintenance_current * pattern[i]
                self.model.neurons[neuron_id].external_input += current

    def decay_activity(self, pattern_id: str) -> None:
        """Apply decay to persistent activity.

        Args:
            pattern_id: Identifier for the pattern.
        """
        if pattern_id not in self.active_patterns:
            return

        # Decay the stored pattern
        self.active_patterns[pattern_id] *= (1 - self.decay_rate)

    def retrieve_pattern(self, pattern_id: str) -> Optional[np.ndarray]:
        """Retrieve the current state of a pattern.

        Args:
            pattern_id: Identifier for the pattern.

        Returns:
            Current pattern state or None if not found.
        """
        if pattern_id not in self.pattern_neurons:
            return None

        neurons = self.pattern_neurons[pattern_id]
        pattern_state = []

        for neuron_id in neurons:
            if neuron_id in self.model.neurons:
                # Get current membrane potential
                v = self.model.neurons[neuron_id].v if hasattr(self.model.neurons[neuron_id], 'v') else 0.0
                pattern_state.append(v)
            else:
                pattern_state.append(0.0)

        return np.array(pattern_state)

    def clear_pattern(self, pattern_id: str) -> None:
        """Clear a stored pattern.

        Args:
            pattern_id: Identifier for the pattern.
        """
        if pattern_id in self.active_patterns:
            del self.active_patterns[pattern_id]
        if pattern_id in self.pattern_neurons:
            del self.pattern_neurons[pattern_id]

    def clear_all(self) -> None:
        """Clear all stored patterns."""
        self.active_patterns.clear()
        self.pattern_neurons.clear()

    def _get_memory_neurons(self) -> List[int]:
        """Get neurons in the memory area.

        Returns:
            List of neuron IDs in memory area.
        """
        areas = self.model.get_areas()
        memory_area = next((a for a in areas if a["name"] == self.memory_area), None)

        if memory_area is None:
            # Return arbitrary neurons if area not found
            return list(self.model.neurons.keys())[:100]

        # Get neurons in area
        ranges = memory_area["coord_ranges"]
        neurons = []

        for neuron_id, neuron in self.model.neurons.items():
            if (
                ranges["x"][0] <= neuron.x <= ranges["x"][1]
                and ranges["y"][0] <= neuron.y <= ranges["y"][1]
                and ranges["z"][0] <= neuron.z <= ranges["z"][1]
                and ranges["w"][0] <= neuron.w <= ranges["w"][1]
            ):
                neurons.append(neuron_id)

        return neurons


class AttractorNetwork:
    """Implements attractor network dynamics for content-addressable memory."""

    def __init__(
        self,
        size: int,
        num_attractors: int = 5,
        learning_rate: float = 0.1,
    ):
        """Initialize attractor network.

        Args:
            size: Number of units in the network.
            num_attractors: Maximum number of attractor states.
            learning_rate: Learning rate for Hebbian learning.
        """
        self.size = size
        self.num_attractors = num_attractors
        self.learning_rate = learning_rate

        # Initialize weight matrix (symmetric)
        self.weights = np.zeros((size, size))
        self.stored_patterns: List[np.ndarray] = []
        self.state = np.zeros(size)

    def store_pattern(self, pattern: np.ndarray) -> None:
        """Store a pattern as an attractor.

        Args:
            pattern: Binary pattern to store (values should be -1 or 1).
        """
        if len(pattern) != self.size:
            raise ValueError(f"Pattern size {len(pattern)} does not match network size {self.size}")

        # Store pattern
        if len(self.stored_patterns) < self.num_attractors:
            self.stored_patterns.append(pattern.copy())

            # Update weights using Hebbian rule
            self.weights += self.learning_rate * np.outer(pattern, pattern)

            # Zero diagonal (no self-connections)
            np.fill_diagonal(self.weights, 0)

    def set_state(self, state: np.ndarray) -> None:
        """Set the current state of the network.

        Args:
            state: State vector.
        """
        if len(state) != self.size:
            raise ValueError(f"State size {len(state)} does not match network size {self.size}")

        self.state = state.copy()

    def update_async(self, num_updates: int = 1) -> np.ndarray:
        """Update network state asynchronously.

        Args:
            num_updates: Number of random unit updates.

        Returns:
            Updated state.
        """
        for _ in range(num_updates):
            # Pick random unit
            i = np.random.randint(self.size)

            # Compute input
            h = np.dot(self.weights[i], self.state)

            # Update state with sign function
            self.state[i] = 1.0 if h >= 0 else -1.0

        return self.state.copy()

    def update_sync(self) -> np.ndarray:
        """Update all units synchronously.

        Returns:
            Updated state.
        """
        # Compute inputs for all units
        h = np.dot(self.weights, self.state)

        # Update states
        self.state = np.sign(h)
        self.state[self.state == 0] = 1  # Handle zeros

        return self.state.copy()

    def recall(
        self,
        cue: np.ndarray,
        max_iterations: int = 100,
        convergence_threshold: float = 0.01,
    ) -> Tuple[np.ndarray, bool]:
        """Recall a pattern from a cue.

        Args:
            cue: Partial or noisy pattern.
            max_iterations: Maximum update iterations.
            convergence_threshold: Threshold for convergence detection.

        Returns:
            Tuple of (recalled_pattern, converged).
        """
        self.set_state(cue)
        converged = False

        for _ in range(max_iterations):
            prev_state = self.state.copy()
            self.update_async(num_updates=self.size)

            # Check convergence
            change = np.mean(np.abs(self.state - prev_state))
            if change < convergence_threshold:
                converged = True
                break

        return self.state.copy(), converged

    def compute_energy(self) -> float:
        """Compute Hopfield energy of current state.

        Returns:
            Energy value.
        """
        energy = -0.5 * np.dot(self.state, np.dot(self.weights, self.state))
        return float(energy)

    def find_nearest_attractor(self, state: np.ndarray) -> Optional[int]:
        """Find the stored pattern nearest to given state.

        Args:
            state: State vector.

        Returns:
            Index of nearest stored pattern or None.
        """
        if not self.stored_patterns:
            return None

        distances = [
            np.sum(np.abs(state - pattern))
            for pattern in self.stored_patterns
        ]

        return int(np.argmin(distances))


class MemoryGate:
    """Gating mechanism for controlling memory access."""

    def __init__(
        self,
        gate_threshold: float = 0.5,
        update_strength: float = 1.0,
    ):
        """Initialize memory gate.

        Args:
            gate_threshold: Threshold for gate opening.
            update_strength: Strength of gated updates.
        """
        self.gate_threshold = gate_threshold
        self.update_strength = update_strength
        self.gate_state = 0.0
        self.gated_value: Optional[np.ndarray] = None

    def update_gate(self, control_signal: float) -> None:
        """Update gate state based on control signal.

        Args:
            control_signal: Control signal (0 to 1).
        """
        self.gate_state = max(0.0, min(1.0, control_signal))

    def is_open(self) -> bool:
        """Check if gate is open.

        Returns:
            True if gate is open.
        """
        return self.gate_state >= self.gate_threshold

    def apply_gate(self, input_value: np.ndarray, memory_value: np.ndarray) -> np.ndarray:
        """Apply gating to blend input and memory.

        Args:
            input_value: New input value.
            memory_value: Current memory value.

        Returns:
            Gated output value.
        """
        if self.is_open():
            # Gate open: update memory with input
            output = (1 - self.gate_state) * memory_value + self.gate_state * input_value
            self.gated_value = output
        else:
            # Gate closed: maintain memory
            output = memory_value
            self.gated_value = output

        return output

    def reset(self) -> None:
        """Reset gate state."""
        self.gate_state = 0.0
        self.gated_value = None


class WorkingMemoryBuffer:
    """High-level working memory buffer with multiple slots."""

    def __init__(self, num_slots: int = 7, slot_size: int = 50):
        """Initialize working memory buffer.

        Args:
            num_slots: Number of memory slots (default: 7 for Miller's law).
            slot_size: Size of each memory slot.
        """
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.slots: List[Optional[np.ndarray]] = [None] * num_slots
        self.slot_ages: List[int] = [0] * num_slots
        self.gates = [MemoryGate() for _ in range(num_slots)]

    def store(self, item: np.ndarray, slot_index: Optional[int] = None) -> int:
        """Store an item in working memory.

        Args:
            item: Item to store.
            slot_index: Specific slot to use, or None for automatic.

        Returns:
            Slot index where item was stored.
        """
        if len(item) != self.slot_size:
            # Resize item to fit slot
            if len(item) < self.slot_size:
                item = np.pad(item, (0, self.slot_size - len(item)), mode='constant')
            else:
                item = item[:self.slot_size]

        if slot_index is not None:
            if 0 <= slot_index < self.num_slots:
                self.slots[slot_index] = item.copy()
                self.slot_ages[slot_index] = 0
                return slot_index
            else:
                raise ValueError(f"Invalid slot index {slot_index}")

        # Find empty slot or replace oldest
        empty_slots = [i for i, slot in enumerate(self.slots) if slot is None]
        if empty_slots:
            slot_index = empty_slots[0]
        else:
            slot_index = int(np.argmax(self.slot_ages))

        self.slots[slot_index] = item.copy()
        self.slot_ages[slot_index] = 0

        return slot_index

    def retrieve(self, slot_index: int) -> Optional[np.ndarray]:
        """Retrieve an item from working memory.

        Args:
            slot_index: Index of slot to retrieve.

        Returns:
            Retrieved item or None if slot is empty.
        """
        if 0 <= slot_index < self.num_slots:
            return self.slots[slot_index].copy() if self.slots[slot_index] is not None else None
        return None

    def update(self, slot_index: int, new_value: np.ndarray, control_signal: float) -> None:
        """Update a memory slot with gating.

        Args:
            slot_index: Index of slot to update.
            new_value: New value to potentially write.
            control_signal: Control signal for gating (0 to 1).
        """
        if not (0 <= slot_index < self.num_slots):
            return

        self.gates[slot_index].update_gate(control_signal)

        if self.slots[slot_index] is not None:
            gated_value = self.gates[slot_index].apply_gate(new_value, self.slots[slot_index])
            self.slots[slot_index] = gated_value
        else:
            self.slots[slot_index] = new_value.copy()

        self.slot_ages[slot_index] = 0

    def clear_slot(self, slot_index: int) -> None:
        """Clear a memory slot.

        Args:
            slot_index: Index of slot to clear.
        """
        if 0 <= slot_index < self.num_slots:
            self.slots[slot_index] = None
            self.slot_ages[slot_index] = 0
            self.gates[slot_index].reset()

    def clear_all(self) -> None:
        """Clear all memory slots."""
        for i in range(self.num_slots):
            self.clear_slot(i)

    def age_memory(self) -> None:
        """Increment age of all memory slots."""
        for i in range(self.num_slots):
            if self.slots[i] is not None:
                self.slot_ages[i] += 1

    def get_occupancy(self) -> float:
        """Get fraction of occupied slots.

        Returns:
            Occupancy ratio (0 to 1).
        """
        occupied = sum(1 for slot in self.slots if slot is not None)
        return occupied / self.num_slots

    def search_content(self, query: np.ndarray, top_k: int = 1) -> List[Tuple[int, float]]:
        """Search for items similar to query.

        Args:
            query: Query pattern.
            top_k: Number of top matches to return.

        Returns:
            List of (slot_index, similarity) tuples.
        """
        if len(query) != self.slot_size:
            if len(query) < self.slot_size:
                query = np.pad(query, (0, self.slot_size - len(query)), mode='constant')
            else:
                query = query[:self.slot_size]

        similarities = []
        for i, slot in enumerate(self.slots):
            if slot is not None:
                # Compute cosine similarity
                dot_product = np.dot(query, slot)
                norm_product = np.linalg.norm(query) * np.linalg.norm(slot)
                similarity = dot_product / (norm_product + 1e-8)
                similarities.append((i, float(similarity)))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]


class AttentionMechanism:
    """Implements attention mechanisms for selective processing.
    
    Provides both top-down (goal-directed) and bottom-up (stimulus-driven)
    attention control. Attention modulates neural activity to prioritize
    relevant information while suppressing irrelevant inputs.
    
    Attributes:
        model: The brain model
        attention_weights: Current attention weights for each neuron
        saliency_map: Bottom-up saliency scores
        
    Example:
        >>> attention = AttentionMechanism(model)
        >>> # Apply top-down attention to prefrontal area
        >>> attention.apply_topdown_attention("prefrontal_cortex", strength=1.5)
        >>> # Compute bottom-up saliency from visual input
        >>> saliency = attention.compute_saliency("vision_area")
    """
    
    def __init__(
        self,
        model: "BrainModel",
        default_attention: float = 1.0,
    ):
        """Initialize attention mechanism.
        
        Args:
            model: Brain model to apply attention to
            default_attention: Default attention weight (1.0 = no modulation)
        """
        self.model = model
        self.default_attention = default_attention
        self.attention_weights: Dict[int, float] = {}
        self.saliency_map: Dict[int, float] = {}
        
        # Initialize all neurons with default attention
        for neuron_id in self.model.neurons.keys():
            self.attention_weights[neuron_id] = default_attention
    
    def apply_topdown_attention(
        self,
        target_area: str,
        strength: float = 1.5,
        decay_rate: float = 0.1,
    ) -> None:
        """Apply top-down attention to a brain area.
        
        Top-down attention represents goal-directed focus on a particular
        brain area or set of neurons. This enhances processing in the
        attended region while potentially suppressing unattended regions.
        
        Args:
            target_area: Name of brain area to attend to
            strength: Attention strength multiplier (>1 enhances, <1 suppresses)
            decay_rate: Rate at which attention decays over time
            
        Example:
            >>> # Focus attention on visual area for visual search task
            >>> attention.apply_topdown_attention("vision_area", strength=2.0)
        """
        areas = self.model.get_areas()
        target = next((a for a in areas if a["name"] == target_area), None)
        
        if target is None:
            return
        
        # Get neurons in target area
        ranges = target["coord_ranges"]
        
        for neuron_id, neuron in self.model.neurons.items():
            # Check if neuron is in target area
            in_area = (
                ranges["x"][0] <= neuron.x <= ranges["x"][1] and
                ranges["y"][0] <= neuron.y <= ranges["y"][1] and
                ranges["z"][0] <= neuron.z <= ranges["z"][1] and
                ranges["w"][0] <= neuron.w <= ranges["w"][1]
            )
            
            if in_area:
                # Enhance attention in target area
                self.attention_weights[neuron_id] = strength
            else:
                # Decay attention in non-target areas
                current = self.attention_weights.get(neuron_id, self.default_attention)
                self.attention_weights[neuron_id] = current * (1 - decay_rate)
    
    def compute_saliency(
        self,
        sensory_area: str,
        use_temporal_change: bool = True,
    ) -> Dict[int, float]:
        """Compute bottom-up saliency map from sensory input.
        
        Bottom-up attention is driven by stimulus properties. Salient features
        (e.g., high contrast, rapid change, unique features) automatically
        capture attention.
        
        Args:
            sensory_area: Name of sensory area to compute saliency for
            use_temporal_change: Whether to include temporal change in saliency
            
        Returns:
            Dictionary mapping neuron IDs to saliency scores
            
        Example:
            >>> # Compute visual saliency from current neural activity
            >>> saliency = attention.compute_saliency("vision_area")
            >>> # Most salient neuron
            >>> max_salient = max(saliency.items(), key=lambda x: x[1])
        """
        areas = self.model.get_areas()
        area = next((a for a in areas if a["name"] == sensory_area), None)
        
        if area is None:
            return {}
        
        ranges = area["coord_ranges"]
        saliency = {}
        
        # Get neurons in sensory area
        area_neurons = []
        for neuron_id, neuron in self.model.neurons.items():
            in_area = (
                ranges["x"][0] <= neuron.x <= ranges["x"][1] and
                ranges["y"][0] <= neuron.y <= ranges["y"][1] and
                ranges["z"][0] <= neuron.z <= ranges["z"][1] and
                ranges["w"][0] <= neuron.w <= ranges["w"][1]
            )
            if in_area:
                area_neurons.append((neuron_id, neuron))
        
        if not area_neurons:
            return {}
        
        # Compute activity statistics
        activities = np.array([n.membrane_potential for _, n in area_neurons])
        mean_activity = np.mean(activities)
        std_activity = np.std(activities)
        
        # Compute saliency for each neuron
        for neuron_id, neuron in area_neurons:
            # Saliency based on deviation from mean (center-surround)
            deviation = abs(neuron.membrane_potential - mean_activity)
            saliency_score = deviation / (std_activity + 1e-8)
            
            # Add temporal change component if requested
            if use_temporal_change and hasattr(neuron, 'previous_potential'):
                temporal_change = abs(
                    neuron.membrane_potential - neuron.previous_potential
                )
                saliency_score += temporal_change * 0.5
            
            saliency[neuron_id] = float(saliency_score)
        
        self.saliency_map = saliency
        return saliency
    
    def apply_attention_modulation(self) -> None:
        """Apply attention weights to modulate neural processing.
        
        Multiplies external input to each neuron by its attention weight.
        This enhances processing of attended stimuli and suppresses
        unattended stimuli.
        
        Should be called each simulation step after sensory input is provided.
        """
        for neuron_id, weight in self.attention_weights.items():
            if neuron_id in self.model.neurons:
                neuron = self.model.neurons[neuron_id]
                # Modulate external input by attention weight
                neuron.external_input *= weight
    
    def winner_take_all(
        self,
        area_name: str,
        top_k: int = 1,
    ) -> List[int]:
        """Implement winner-take-all selection in a brain area.
        
        Selects the most active neurons and suppresses all others.
        This creates focused, competitive selection of neural populations.
        
        Args:
            area_name: Name of brain area to apply WTA
            top_k: Number of "winners" to select
            
        Returns:
            List of winning neuron IDs
            
        Example:
            >>> # Select single most active neuron in motor area
            >>> winners = attention.winner_take_all("motor_cortex", top_k=1)
        """
        areas = self.model.get_areas()
        area = next((a for a in areas if a["name"] == area_name), None)
        
        if area is None:
            return []
        
        ranges = area["coord_ranges"]
        
        # Collect neurons in area with their activity
        area_neurons = []
        for neuron_id, neuron in self.model.neurons.items():
            in_area = (
                ranges["x"][0] <= neuron.x <= ranges["x"][1] and
                ranges["y"][0] <= neuron.y <= ranges["y"][1] and
                ranges["z"][0] <= neuron.z <= ranges["z"][1] and
                ranges["w"][0] <= neuron.w <= ranges["w"][1]
            )
            if in_area:
                area_neurons.append((neuron_id, neuron.membrane_potential))
        
        if not area_neurons:
            return []
        
        # Sort by activity level
        area_neurons.sort(key=lambda x: x[1], reverse=True)
        
        # Select top-k winners
        winners = [nid for nid, _ in area_neurons[:top_k]]
        
        # Suppress non-winners
        for neuron_id, _ in area_neurons[top_k:]:
            if neuron_id in self.model.neurons:
                self.model.neurons[neuron_id].external_input *= 0.1
        
        return winners
    
    def reset_attention(self) -> None:
        """Reset all attention weights to default."""
        for neuron_id in self.attention_weights.keys():
            self.attention_weights[neuron_id] = self.default_attention
        self.saliency_map.clear()
