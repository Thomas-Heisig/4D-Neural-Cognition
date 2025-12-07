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
        if None in self.slots:
            slot_index = self.slots.index(None)
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
